import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import time 
import random 
import logging
import re
import config
import user_agents
import supabase_utils
from markdownify import markdownify as md
import json
import uuid
import math
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urlparse
import relist_tracking
from lane_catalog import canonical_lane_slug
from scrape_configuration import (
    LinkedInSearchExecution,
    ScrapeConfiguration,
    build_search_executions,
    load_scrape_configuration,
)
from linkedin_source_policy import (
    DurableLinkedInRequestGate,
    LinkedInCircuitOpen,
    LinkedInGrantRejected,
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_relist_detail_fetches_used = 0
_relist_detail_fetches_lock = threading.Lock()
_linkedin_scrape_state = threading.local()


class LinkedInAccessDenied(RuntimeError):
    """Stop the complete run when LinkedIn explicitly denies access."""


class LinkedInRequestFailed(RuntimeError):
    """Stop the complete run when required LinkedIn coverage is incomplete."""


class LinkedInQueryJobs(list):
    """List-compatible query result that carries operational completeness."""

    def __init__(
        self,
        jobs=(),
        *,
        processing_complete: bool = True,
        incomplete_reason: str | None = None,
    ):
        super().__init__(jobs)
        self.processing_complete = processing_complete
        self.incomplete_reason = incomplete_reason


class _LinkedInDetailUnavailable:
    """Falsey marker for a confirmed terminal LinkedIn detail response."""

    confirmed_terminal_unavailable = True

    def __init__(self, status_code: int = 404, confirmations: int = 2):
        self.status_code = status_code
        self.confirmations = confirmations

    def __bool__(self) -> bool:
        return False


LINKEDIN_DETAIL_UNAVAILABLE = _LinkedInDetailUnavailable()


class LinkedInRequestLimiter:
    """Apply one polite request cadence across search and detail endpoints."""

    def __init__(self, minimum_interval_ms: int, jitter_ms: int = 1_500):
        self.minimum_interval_seconds = max(0, minimum_interval_ms) / 1000
        self.jitter_seconds = max(0, jitter_ms) / 1000
        self._last_request_started_at: float | None = None
        self._lock = threading.Lock()
        self.request_count = 0
        self.total_wait_seconds = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            target_interval = self.minimum_interval_seconds + random.uniform(
                0, self.jitter_seconds
            )
            elapsed = (
                target_interval
                if self._last_request_started_at is None
                else now - self._last_request_started_at
            )
            wait_seconds = max(0.0, target_interval - elapsed)
            if wait_seconds:
                logging.info(
                    "Waiting %.2f seconds for global LinkedIn request pacing...",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
            self._last_request_started_at = time.monotonic()
            self.request_count += 1
            self.total_wait_seconds += wait_seconds
            return wait_seconds


def resolve_linkedin_lookback_hours(
    last_success_at: str | None,
    now: datetime | None = None,
    configured_hours: int | None = None,
    overlap_hours: int | None = None,
    max_hours: int | None = None,
) -> int:
    configured_hours = configured_hours or config.LINKEDIN_LOOKBACK_HOURS
    overlap_hours = config.LINKEDIN_LOOKBACK_OVERLAP_HOURS if overlap_hours is None else overlap_hours
    max_hours = max_hours or config.LINKEDIN_MAX_LOOKBACK_HOURS
    if not last_success_at:
        return configured_hours

    try:
        parsed = datetime.fromisoformat(last_success_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        logging.warning("Invalid LinkedIn scrape watermark %r; using %s hours", last_success_at, configured_hours)
        return configured_hours

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elapsed_hours = max(0, math.ceil((now - parsed).total_seconds() / 3600))
    return min(
        max_hours,
        max(configured_hours, elapsed_hours + overlap_hours),
    )


def _resolve_archetype_config(
    archetype: str | None,
    runtime_profile: dict | None = None,
) -> tuple[str, dict]:
    resolved_archetype = archetype or config.DEFAULT_ARCHETYPE
    archetype_config = (
        runtime_profile
        if runtime_profile is not None
        else config.ARCHETYPE_CONFIGS.get(resolved_archetype)
    )
    if archetype_config is None:
        raise ValueError(f"Unknown archetype '{resolved_archetype}'. Check config.ARCHETYPE_CONFIGS.")

    missing_keys = [key for key in ("filter_profile",) if not archetype_config.get(key)]
    if missing_keys:
        missing_keys_text = ", ".join(missing_keys)
        raise ValueError(
            f"Archetype '{resolved_archetype}' is missing required config key(s): {missing_keys_text}."
        )

    return resolved_archetype, archetype_config


def _lane_runtime_archetype_config(execution: LinkedInSearchExecution) -> dict:
    """Translate typed lane context to the existing filter interface."""
    return _career_lane_runtime_profile(execution.lane, execution.geography.location)


def _career_lane_runtime_profile(lane, location: str) -> dict:
    base_profile = config.ARCHETYPE_CONFIGS.get(lane.archetype)
    if base_profile is None and lane.archetype == "technology_delivery":
        base_profile = config.ARCHETYPE_CONFIGS.get("software_tpm")
    profile = dict(base_profile or {})
    profile.update({
        "provider": "linkedin",
        "location": location,
        "filter_profile": f"{lane.archetype}_v1",
        "definition": lane.description,
        "route_when": lane.routing_guidance,
        "title_context": list(lane.title_include),
        "description_context": list(lane.description_include),
        "positive_signals": list(lane.description_include),
        "exclude_signals": [
            *lane.title_exclude,
            *lane.description_exclude,
        ],
        "title_blocklist": list(lane.title_exclude),
        "desc_blocklist": list(lane.description_exclude),
    })
    for key in ("company_blocklist", "title_blocklist", "title_entry_level_blocklist", "desc_blocklist"):
        profile.setdefault(key, [])
    return profile

def _parse_salary_fields(text: str) -> dict:
    if not text:
        return {"salary_text": None, "salary_min": None, "salary_max": None, "salary_currency": None}

    match = re.search(
        r'(?:(CAD|USD)\s*)?'
        r'(\$?\s*\d[\d,]*(?:\.\d+)?\s*[kK]?)'
        r'\s*(?:-|–|—|to|à)\s*'
        r'(\$?\s*\d[\d,]*(?:\.\d+)?\s*[kK]?)'
        r'\s*(CAD|USD)?',
        text,
        re.IGNORECASE,
    )
    if not match:
        return {"salary_text": None, "salary_min": None, "salary_max": None, "salary_currency": None}

    leading_currency, raw_min, raw_max, trailing_currency = match.groups()

    def parse_amount(value: str) -> int:
        normalized = value.replace("$", "").replace(",", "").replace(" ", "")
        multiplier = 1000 if normalized.lower().endswith("k") else 1
        normalized = normalized.rstrip("kK")
        return int(float(normalized) * multiplier)

    salary_min = parse_amount(raw_min)
    salary_max = parse_amount(raw_max)
    context = text[max(0, match.start() - 100):match.end() + 100]
    has_pay_context = re.search(
        r'\b(salary|pay|compensation|wage|rate|remuneration|rémunération|salaire|'
        r'annual|annually|yearly|hourly|bi-weekly|weekly|monthly)\b|'
        r'/(?:hr|hour|year)|per\s+(?:hour|year|annum)',
        context,
        re.IGNORECASE,
    )
    has_salary_marker = bool(
        (leading_currency or trailing_currency)
        or "$" in raw_min
        or "$" in raw_max
        or raw_min.strip().lower().endswith("k")
        or raw_max.strip().lower().endswith("k")
    )
    if not has_salary_marker or not has_pay_context or salary_min < 1000 or salary_max < 1000 or salary_max < salary_min:
        return {"salary_text": None, "salary_min": None, "salary_max": None, "salary_currency": None}

    currency = leading_currency or trailing_currency
    return {
        "salary_text": match.group(0).strip(),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": currency.upper() if currency else None,
    }

def _extract_recruiter_identifier(profile_url: str | None) -> str | None:
    if not profile_url:
        return None
    path = urlparse(profile_url).path.strip("/")
    return path.removeprefix("in/") if path.startswith("in/") else (path or None)

def _extract_linkedin_detail_metadata(soup: BeautifulSoup) -> dict:
    result = {
        "applicant_count": None,
        "applicant_count_text": None,
        "applicant_count_type": None,
        "salary_text": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "recruiter_name": None,
        "recruiter_profile_url": None,
        "recruiter_identifier": None,
    }

    applicant_caption = soup.find(["span", "figcaption"], {"class": re.compile(r"num-applicants__caption")})
    if applicant_caption:
        text = applicant_caption.get_text(" ", strip=True)
        result["applicant_count_text"] = text
        match = re.search(r'(\d+)', text.replace(",", ""))
        if match:
            result["applicant_count"] = int(match.group(1))
            lowered = text.lower()
            if "over" in lowered or "+" in lowered:
                result["applicant_count_type"] = "lower_bound"
            elif "among the first" in lowered or "fewer than" in lowered:
                result["applicant_count_type"] = "upper_bound"
            else:
                result["applicant_count_type"] = "exact"

    recruiter_section = soup.find(class_=re.compile(r"hirer-card|message-the-recruiter"))
    recruiter_link = (recruiter_section or soup).find("a", href=re.compile(r"linkedin\.com/in/"))
    if recruiter_link:
        result["recruiter_profile_url"] = recruiter_link.get("href")
        result["recruiter_identifier"] = _extract_recruiter_identifier(result["recruiter_profile_url"])
        text = recruiter_link.get_text(" ", strip=True)
        if text and len(text.split()) <= 5 and "message the hiring team" not in text.lower():
            result["recruiter_name"] = text

    description_div = soup.find("div", {"class": "show-more-less-html__markup"})
    description_text = description_div.get_text(" ", strip=True) if description_div else ""
    result.update(_parse_salary_fields(description_text))
    return result

def _extract_linkedin_search_cards(job_elements: list) -> list[dict]:
    results = []
    seen = set()
    for job_element in job_elements:
        base_card = job_element.find("div", {"class": "base-card"})
        job_urn = base_card.get("data-entity-urn") if base_card else None
        if not job_urn or "jobPosting:" not in job_urn:
            continue

        try:
            job_id = job_urn.split(":")[3]
        except IndexError:
            logging.warning(f"Could not parse job ID from URN: {job_urn}")
            continue

        if job_id in seen:
            continue
        seen.add(job_id)

        time_el = job_element.find("time")
        posted_at = time_el.get("datetime").strip() if time_el and time_el.get("datetime") else None
        posted_relative_text = time_el.get_text(" ", strip=True) if time_el else None
        title_el = job_element.select_one(".base-search-card__title")
        company_el = job_element.select_one(".base-search-card__subtitle")
        location_el = job_element.select_one(".job-search-card__location")

        results.append({
            "job_id": job_id,
            "posted_at": posted_at,
            "posted_relative_text": posted_relative_text,
            "job_title": title_el.get_text(" ", strip=True) if title_el else None,
            "company": company_el.get_text(" ", strip=True) if company_el else None,
            "location": location_el.get_text(" ", strip=True) if location_el else None,
        })
    return results

# Convert HTML description to Markdown
def convert_html_to_markdown(html: str) -> str | None:
    """
    Convert HTML to clean Markdown using BeautifulSoup (to strip unwanted tags)
    and markdownify (to convert the cleaned HTML to Markdown).
    No LLM API calls are made — this is entirely local.
    """
    if not html or not html.strip():
        logging.info("Received empty HTML for Markdown conversion, returning empty string.")
        return ""

    try:
        # Clean the HTML: remove scripts, styles, nav, and other non-content tags
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
            tag.decompose()

        cleaned_html = str(soup)

        # Convert cleaned HTML to Markdown
        markdown_text = md(
            cleaned_html,
            heading_style="ATX",
            bullets="-",
            strip=['img'],
        )

        # Clean up excessive blank lines
        lines = markdown_text.splitlines()
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if not line.strip():
                if not prev_blank:
                    cleaned_lines.append('')
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False
        markdown_text = '\n'.join(cleaned_lines).strip()

        logging.info("Successfully converted HTML to Markdown.")
        return markdown_text if markdown_text else ""
    except Exception as e:
        logging.error(f"Error during HTML to Markdown conversion: {e}")
        return None

def _get_careers_future_job_company_name(job_item: dict) -> str | None:
    """Helper to extract company name, preferring hiringCompany."""
    if not isinstance(job_item, dict):
        return None
    
    hiring_company = job_item.get('hiringCompany')
    if isinstance(hiring_company, dict) and hiring_company.get('name'):
        return hiring_company['name']
    
    posted_company = job_item.get('postedCompany')
    if isinstance(posted_company, dict) and posted_company.get('name'):
        return posted_company['name']
        
    return None

# --- LinkedIn Scraping Logic ---
def _fetch_linkedin_job_ids(
    search_query: str,
    location: str,
    posting_date_filter: str | None = None,
    geo_id: int | None = None,
    max_start: int | None = None,
    job_type: str | None = None,
    work_types: str | None = None,
    geo_id_is_explicit: bool = False,
    request_delay_ms: int | None = None,
    request_limiter: LinkedInRequestLimiter | None = None,
    durable_gate: DurableLinkedInRequestGate | None = None,
    user_agent: str | None = None,
) -> list:
    """Fetches job IDs from LinkedIn search results pages with delays, rotating user agents, and retries."""

    coverage = {
        "pages_attempted": 0,
        "pages_completed": 0,
        "page_coverage": [],
    }
    _linkedin_scrape_state.coverage = coverage
    scraped_cards = []
    start = 0
    max_start = config.LINKEDIN_MAX_START if max_start is None else max_start
    posting_date_filter = posting_date_filter or config.LINKEDIN_JOB_POSTING_DATE
    resolved_geo_id = geo_id
    if geo_id is None and not geo_id_is_explicit and location == config.LINKEDIN_LOCATION:
        resolved_geo_id = config.LINKEDIN_GEO_ID
    job_type = job_type or config.LINKEDIN_JOB_TYPE
    work_types = work_types or config.LINKEDIN_F_WT


    logging.info(f"--- Starting Phase 1: Scraping Job IDs (Max Start: {max_start}) ---")
    while start <= max_start:
        coverage["pages_attempted"] += 1
        query_parameters = {
            "keywords": search_query,
            "location": location,
            "f_TPR": posting_date_filter,
            "f_JT": job_type,
            "f_WT": work_types,
            "start": start,
        }
        if resolved_geo_id is not None:
            query_parameters["geoId"] = resolved_geo_id
        target_url = (
            "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?"
            + urlencode(query_parameters)
        )

        if request_limiter is not None:
            request_limiter.wait()
        elif start > 0:
            sleep_time = (
                request_delay_ms / 1000
                if request_delay_ms is not None
                else random.uniform(5.0, 15.0)
            )
            logging.info(f"Waiting for {sleep_time:.2f} seconds before next request...")
            time.sleep(sleep_time)

        request_user_agent = user_agent or random.choice(user_agents.USER_AGENTS)
        headers = {'User-Agent': request_user_agent}
    
        logging.info(f"Using User-Agent: {request_user_agent}")

    
        logging.info(f"Scraping URL: {target_url}")

        res = None 
        retries = 0
        while retries <= config.MAX_RETRIES:
            grant = None
            try:
                if durable_gate is not None:
                    grant = durable_gate.acquire(
                        "search", f"legacy:{search_query}:{location}:{start}:{retries}"
                    )
                res = requests.get(target_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
                if _linkedin_response_is_challenge(res):
                    if durable_gate is not None and grant is not None:
                        durable_gate.open_circuit(
                            grant, "LinkedIn denied or challenged legacy search", res.status_code
                        )
                    raise LinkedInAccessDenied(
                        f"LinkedIn denied or challenged search access (status={res.status_code})"
                    )
                res.raise_for_status()
                if durable_gate is not None and grant is not None:
                    durable_gate.finish(grant, "complete", res.status_code)
                break
            except LinkedInAccessDenied:
                raise
            except requests.exceptions.HTTPError as e:
                if durable_gate is not None and grant is not None:
                    durable_gate.finish(grant, "http_error", e.response.status_code)
                if e.response.status_code == 429 and retries < config.MAX_RETRIES:
                    retries += 1
                    wait_time = max(
                        _retry_after_seconds(e.response) or 0,
                        config.RETRY_DELAY_SECONDS + random.uniform(0, 5),
                    )
                    
                    logging.warning(f"Error 429: Too Many Requests. Retrying attempt {retries}/{config.MAX_RETRIES} after {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

                    logging.info("Retrying search request after LinkedIn cooldown")
                    if request_limiter is not None:
                        request_limiter.wait()
                    continue
                if e.response.status_code == 429:
                    raise LinkedInRequestFailed(
                        "LinkedIn search throttling exhausted the retry budget"
                    ) from e
                else:
                    
                    logging.error(f"HTTP Error fetching search results page: {e}")
                    res = None 
                    break
            except requests.exceptions.RequestException as e:
                if durable_gate is not None and grant is not None:
                    durable_gate.finish(grant, "transport_error", None)
                raise LinkedInRequestFailed(
                    f"LinkedIn search request failed: {e}"
                ) from e

        
        if res is None:
            raise LinkedInRequestFailed(
                f"Failed to fetch {target_url} after {retries} retries"
            )

        if not res.text:
             coverage["page_coverage"].append({
                 "page": start // 10 + 1,
                 "start": start,
                 "elements": 0,
                 "cards": 0,
                 "new_source_ids": 0,
                 "result": "empty_response",
             })
             raise LinkedInRequestFailed(
                 f"LinkedIn search returned an empty response at start={start}"
             )

        coverage["pages_completed"] += 1

        soup = BeautifulSoup(res.text, 'html.parser')
        all_jobs_on_this_page = soup.find_all('li')

        if not all_jobs_on_this_page:
             coverage["page_coverage"].append({
                 "page": start // 10 + 1,
                 "start": start,
                 "elements": 0,
                 "cards": 0,
                 "new_source_ids": 0,
                 "result": "no_results",
             })
             logging.info(f"No job listings ('li' elements) found on page at start={start}, stopping.")
             break

    
        logging.info(f"Found {len(all_jobs_on_this_page)} potential job elements on this page.")

        page_cards = _extract_linkedin_search_cards(all_jobs_on_this_page)
        if not page_cards:
            coverage["page_coverage"].append({
                "page": start // 10 + 1,
                "start": start,
                "elements": len(all_jobs_on_this_page),
                "cards": 0,
                "new_source_ids": 0,
                "result": "parser_failure",
            })
            raise LinkedInRequestFailed(
                f"LinkedIn search parser extracted zero cards from "
                f"{len(all_jobs_on_this_page)} list elements at start={start}"
            )
        existing_ids = {card["job_id"] for card in scraped_cards}
        new_page_cards = [card for card in page_cards if card["job_id"] not in existing_ids]
        scraped_cards.extend(new_page_cards)
        jobs_found_this_iteration = len(new_page_cards)

        coverage["page_coverage"].append({
            "page": start // 10 + 1,
            "start": start,
            "elements": len(all_jobs_on_this_page),
            "cards": len(page_cards),
            "new_source_ids": jobs_found_this_iteration,
            "result": "complete",
        })

    
        logging.info(f"Added {jobs_found_this_iteration} unique job IDs from this page.")

        if jobs_found_this_iteration == 0 and len(all_jobs_on_this_page) > 0:
            logging.info(
                "Found list items but no new job IDs at start=%s; continuing to the configured bound.",
                start,
            )

        start += 10


    logging.info(f"--- Finished Phase 1: Found {len(scraped_cards)} unique job IDs during scraping ---")
    return scraped_cards


def _linkedin_max_start_for_pages(max_pages: int) -> int:
    """Translate an exact page count into LinkedIn's 10-result start offset."""
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    return (max_pages - 1) * 10


def _retry_after_seconds(response) -> float | None:
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _linkedin_response_is_challenge(response) -> bool:
    if getattr(response, "status_code", None) in (403, 999):
        return True
    url = str(getattr(response, "url", "") or "").lower()
    if "/checkpoint/" in url or "/challenge/" in url:
        return True
    text = (getattr(response, "text", "") or "")[:2_000].lower()
    return (
        "<title>security verification" in text
        or 'id="challenge-page"' in text
        or "id='challenge-page'" in text
    )


def _fetch_linkedin_job_details(
    job_id: str,
    search_card: dict | None = None,
    request_limiter: LinkedInRequestLimiter | None = None,
    durable_gate: DurableLinkedInRequestGate | None = None,
    user_agent: str | None = None,
    request_key: str | None = None,
    physical_attempts: list[int] | None = None,
    physical_limit: int | None = None,
    required_fields: tuple[str, ...] = (),
    deadline: float | None = None,
) -> tuple[dict, dict] | _LinkedInDetailUnavailable | None:
    """Fetch job content and detail metadata, returned as separate dictionaries."""

    job_detail_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

    logging.info(f"Preparing to fetch details for job ID: {job_id}")

    if durable_gate is None and request_limiter is None:
        sleep_time = random.uniform(3.0, 10.0)
        logging.info(f"Waiting for {sleep_time:.2f} seconds before fetching details...")
        time.sleep(sleep_time)
    elif durable_gate is None:
        request_limiter.wait()

    user_agent = user_agent or random.choice(user_agents.USER_AGENTS)
    headers = {'User-Agent': user_agent}

    logging.info(f"Using User-Agent for details: {user_agent}")


    logging.info(f"Fetching details from: {job_detail_url}")

    resp = None 
    retries = 0
    while retries <= config.MAX_RETRIES:
        grant = None
        try:
            if physical_attempts is not None:
                if physical_limit is not None and physical_attempts[0] >= physical_limit:
                    raise LinkedInRequestFailed("LinkedIn detail physical-attempt budget exhausted")
                physical_attempts[0] += 1
            if durable_gate is not None:
                grant = durable_gate.acquire(
                    "detail", f"{request_key or job_id}:{retries}", deadline=deadline
                )
            request_timeout = config.REQUEST_TIMEOUT
            if deadline is not None:
                request_timeout = min(
                    request_timeout, max(0.1, deadline - time.monotonic())
                )
            resp = requests.get(job_detail_url, headers=headers, timeout=request_timeout)
            if _linkedin_response_is_challenge(resp):
                if durable_gate is not None and grant is not None:
                    durable_gate.open_circuit(
                        grant,
                        f"LinkedIn denied or challenged detail access for {job_id}",
                        resp.status_code,
                    )
                error_type = LinkedInCircuitOpen if durable_gate is not None else LinkedInAccessDenied
                raise error_type(
                    f"LinkedIn denied or challenged detail access (status={resp.status_code})"
                )
            resp.raise_for_status()
            break
        except (LinkedInAccessDenied, LinkedInCircuitOpen):
            raise
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 404 and retries < 1:
                if durable_gate is not None and grant is not None:
                    durable_gate.finish(grant, "not_found_unconfirmed", status_code)
                retries += 1
                if deadline is not None and time.monotonic() + config.RETRY_DELAY_SECONDS >= deadline:
                    raise LinkedInRequestFailed(
                        f"LinkedIn detail deadline exhausted for job {job_id}"
                    ) from e
                time.sleep(config.RETRY_DELAY_SECONDS)
                continue
            if status_code in (404, 410):
                if durable_gate is not None and grant is not None:
                    durable_gate.finish(grant, "terminal_unavailable", status_code)
                logging.info(
                    "LinkedIn detail is no longer available for job ID %s (status=%s)",
                    job_id,
                    status_code,
                )
                if status_code == 404:
                    return LINKEDIN_DETAIL_UNAVAILABLE
                return _LinkedInDetailUnavailable(status_code=410, confirmations=1)
            if durable_gate is not None and grant is not None:
                durable_gate.finish(grant, "http_error", status_code)
            if status_code in (429, 500, 502, 503, 504) and retries < config.MAX_RETRIES:
                retries += 1
                wait_time = max(
                    _retry_after_seconds(e.response) or 0,
                    config.RETRY_DELAY_SECONDS + random.uniform(0, 5),
                )
                
                logging.warning(
                    "HTTP %s for job ID %s. Retrying attempt %s/%s after %.2f seconds...",
                    status_code,
                    job_id,
                    retries,
                    config.MAX_RETRIES,
                    wait_time,
                )
                if deadline is not None and time.monotonic() + wait_time >= deadline:
                    raise LinkedInRequestFailed(
                        f"LinkedIn detail deadline exhausted for job {job_id}"
                    ) from e
                time.sleep(wait_time)
                logging.info(
                    "Retrying job %s after LinkedIn cooldown with the same request identity",
                    job_id,
                )
                if request_limiter is not None:
                    request_limiter.wait()
                continue
            if status_code in (429, 500, 502, 503, 504):
                raise LinkedInRequestFailed(
                    f"LinkedIn detail HTTP {status_code} exhausted retries for job {job_id}"
                ) from e
            raise LinkedInRequestFailed(
                f"LinkedIn detail HTTP {status_code} failed for job {job_id}"
            ) from e
        except requests.exceptions.RequestException as e:
            if durable_gate is not None and grant is not None:
                durable_gate.finish(grant, "transport_error", None)
            raise LinkedInRequestFailed(
                f"LinkedIn detail request failed for job {job_id}: {e}"
            ) from e

    
    if resp is None:
         logging.error(f"Failed to fetch details for job ID {job_id} after {retries} retries (unexpected state).")
         return None

    try:
        soup = BeautifulSoup(resp.text, 'html.parser')
        job_details = {"job_id": job_id}

        # --- Extract Company ---
        try:
            company_img = soup.find("div",{"class":"top-card-layout__card"}).find("a").find("img")
            if company_img:
                job_details["company"] = company_img.get('alt').strip()
            if not job_details.get("company"):
                 company_link = soup.find("a", {"class": "topcard__org-name-link"})
                 if company_link:
                      job_details["company"] = company_link.text.strip()
                 else:
                      sub_title_span = soup.find("span", {"class": "topcard__flavor"})
                      if sub_title_span:
                           job_details["company"] = sub_title_span.text.strip()

            if not job_details.get("company"):
                 job_details["company"] = None
                 print(f"Warning: Could not extract company for job ID {job_id}")
        except Exception as e:
            print(f"Error extracting company for job ID {job_id}: {e}")
            job_details["company"] = None

        # --- Extract Job Title ---
        try:
            title_link = soup.find("div",{"class":"top-card-layout__entity-info"}).find("a")
            job_details["job_title"] = title_link.text.strip() if title_link else None
            if not job_details["job_title"]:
                 title_h1 = soup.find("h1", {"class": "top-card-layout__title"})
                 if title_h1:
                      job_details["job_title"] = title_h1.text.strip()
        except Exception as e: 
            print(f"Error extracting job title for job ID {job_id}: {e}")
            job_details["job_title"] = None

        # --- Extract Seniority Level ---
        try:
            # Find all criteria items
            criteria_items = soup.find("ul",{"class":"description__job-criteria-list"}).find_all("li")
            job_details["level"] = None 
            for item in criteria_items:
                header = item.find("h3", {"class": "description__job-criteria-subheader"})
                if header and "Seniority level" in header.text:
                    level_text = item.find("span", {"class": "description__job-criteria-text"})
                    if level_text:
                        job_details["level"] = level_text.text.strip()
                        break 
        except Exception as e: 
            print(f"Error extracting seniority level for job ID {job_id}: {e}")
            job_details["level"] = None

        # --- Extract Location ---
        try:
           
            location_span = soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"})
            if location_span:
                job_details["location"] = location_span.text.strip()
            else:
                
                subtitle_div = soup.find("div", {"class": "topcard__flavor-row"})
                if subtitle_div:
                    location_span_fallback = subtitle_div.find("span", {"class": "topcard__flavor"})
                    if location_span_fallback:
                         job_details["location"] = location_span_fallback.text.strip()

            if not job_details.get("location"): 
                 job_details["location"] = None
                 print(f"Warning: Could not extract location for job ID {job_id}")
        except Exception as e:
            print(f"Error extracting location for job ID {job_id}: {e}")
            job_details["location"] = None

        # --- Extract Description ---
        description_html = "" 
        try:
            description_div = soup.find("div", {"class": "show-more-less-html__markup"})
            if description_div:
                description_html = str(description_div)
            else:
                logging.warning(f"Could not find description div for job ID {job_id}")
        except Exception as e:
                logging.error(f"Error extracting description HTML for job ID {job_id}: {e}")
                description_html = ""

        if description_html.strip():
            job_details["description"] = convert_html_to_markdown(description_html)
        else:
            job_details["description"] = None 
            logging.warning(f"Description HTML was empty for job ID {job_id}. Skipping conversion.") 

        detail_metadata = _extract_linkedin_detail_metadata(soup)
        detail_metadata["detail_metadata_checked_at"] = datetime.now().isoformat()

        # --- Set Provider ---
        job_details["provider"] = "linkedin"

        if search_card:
            job_details["posted_at"] = search_card.get("posted_at")
            job_details["posted_relative_text"] = search_card.get("posted_relative_text")
        missing_fields = [field for field in required_fields if not job_details.get(field)]
        if missing_fields:
            raise ValueError(
                "LinkedIn detail response omitted required fields: "
                + ", ".join(missing_fields)
            )
        if durable_gate is not None and grant is not None:
            durable_gate.finish(grant, "complete", resp.status_code)
        return job_details, detail_metadata

    except LinkedInGrantRejected:
         raise
    except Exception as e:
         if durable_gate is not None and grant is not None:
              durable_gate.finish(grant, "parser_error", getattr(resp, "status_code", None))
         
         logging.error(f"General Error processing details for job ID {job_id} after successful fetch: {e}")
         return None

def process_linkedin_query(
    search_query: str,
    location: str,
    limit: int = None,
    archetype: str | None = None,
    filter_profile: str | None = None,
    posting_date_filter: str | None = None,
    query_id: str | None = None,
    query_kind: str | None = None,
    query_language: str | None = None,
    lane: str | None = None,
    location_scope: str | None = None,
    geography_id: str | None = None,
    geo_id: int | None = None,
    max_start: int | None = None,
    job_type: str | None = None,
    work_types: str | None = None,
    geo_id_is_explicit: bool = False,
    request_delay_ms: int | None = None,
    fetch_descriptions: bool = True,
    runtime_profile: dict | None = None,
    relist_budget: int | None = None,
    run_context: supabase_utils.CanonicalRunContext | None = None,
    request_limiter: LinkedInRequestLimiter | None = None,
    durable_gate: DurableLinkedInRequestGate | None = None,
    user_agent: str | None = None,
) -> list[dict]:
    """
    Orchestrates scraping and detail fetching for a single query,
    filtering against existing jobs in Supabase BEFORE fetching details.
    Returns flat job dictionaries ready for persistence. The lower-level detail
    fetch keeps content and metadata separate for targeted callers; this
    orchestration boundary merges both dictionaries into each returned job.
    """

    global _relist_detail_fetches_used
    resolved_archetype, archetype_config = _resolve_archetype_config(archetype, runtime_profile)
    resolved_filter_profile = filter_profile or archetype_config["filter_profile"]
    run_id = str(uuid.uuid4())
    query_scope = json.dumps({
        "archetype": resolved_archetype,
        "lane": lane or canonical_lane_slug(resolved_archetype),
        "filter_profile": resolved_filter_profile,
        "location": location,
        # This serialized scope is provenance, not a jobs.location_scope write.
        "location_scope": location_scope,
        "geography_id": geography_id,
        "posting_date_filter": posting_date_filter or config.LINKEDIN_JOB_POSTING_DATE,
        "query_id": query_id,
        "query_kind": query_kind,
        "language": query_language,
        "search_query": search_query,
    }, sort_keys=True, separators=(",", ":"))
    tracking_enabled = getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    if tracking_enabled:
        supabase_utils.start_ingestion_run(
            run_id,
            provider="linkedin",
            search_query=search_query,
            archetype=resolved_archetype,
            filter_profile=resolved_filter_profile,
            query_scope=query_scope,
        )

    fetch_options = {"posting_date_filter": posting_date_filter}
    if geo_id is not None:
        fetch_options["geo_id"] = geo_id
    if max_start is not None:
        fetch_options["max_start"] = max_start
    if job_type is not None:
        fetch_options["job_type"] = job_type
    if work_types is not None:
        fetch_options["work_types"] = work_types
    if geo_id_is_explicit:
        fetch_options["geo_id_is_explicit"] = True
    if request_delay_ms is not None:
        fetch_options["request_delay_ms"] = request_delay_ms
    if request_limiter is not None:
        fetch_options["request_limiter"] = request_limiter
    if durable_gate is not None:
        fetch_options["durable_gate"] = durable_gate
    if user_agent is not None:
        fetch_options["user_agent"] = user_agent
    empty_coverage = {
        "pages_attempted": 0,
        "pages_completed": 0,
        "page_coverage": [],
    }
    _linkedin_scrape_state.coverage = empty_coverage.copy()
    try:
        scraped_cards = _fetch_linkedin_job_ids(search_query, location, **fetch_options)
    except (LinkedInAccessDenied, LinkedInRequestFailed) as exc:
        coverage = getattr(
            _linkedin_scrape_state,
            "coverage",
            empty_coverage,
        )
        if tracking_enabled:
            supabase_utils.finish_ingestion_run(
                run_id,
                status="failed",
                pages_attempted=coverage["pages_attempted"],
                pages_completed=coverage["pages_completed"],
                cards_seen=0,
                detail_budget_used=0,
                coverage_complete=False,
                coverage_reason=str(exc),
                page_coverage=coverage["page_coverage"],
            )
        raise
    coverage = getattr(
        _linkedin_scrape_state,
        "coverage",
        empty_coverage,
    )
    if not scraped_cards:
        if tracking_enabled:
            supabase_utils.finish_ingestion_run(
                run_id,
                status="incomplete",
                pages_attempted=coverage["pages_attempted"],
                pages_completed=coverage["pages_completed"],
                cards_seen=0,
                coverage_complete=False,
                coverage_reason="zero cards; empty result or parser/request failure",
                page_coverage=coverage["page_coverage"],
            )
        logging.info("No job IDs found in Phase 1. Skipping detail fetching.")
        return LinkedInQueryJobs(
            processing_complete=False,
            incomplete_reason="zero cards; empty result or parser/request failure",
        )

    normalized_cards = []
    for card in scraped_cards:
        if isinstance(card, dict):
            normalized_cards.append(card)
        else:
            normalized_cards.append({"job_id": str(card)})

    card_by_job_id = {card['job_id']: card for card in normalized_cards}
    unique_linkedin_job_ids = list(card_by_job_id.keys())

    tracking_context = (
        supabase_utils.get_listing_tracking_context(
            "linkedin",
            unique_linkedin_job_ids,
            canonical_by_source=(
                run_context.canonical_ids_for_sources(
                    "linkedin",
                    unique_linkedin_job_ids,
                )
                if run_context is not None
                else None
            ),
        )
        if tracking_enabled
        else {}
    )
    relist_candidates = []
    min_forward_days = getattr(config, "LINKEDIN_RELIST_MIN_FORWARD_DAYS", 2)
    stable_observations = getattr(config, "LINKEDIN_RELIST_STABLE_OBSERVATIONS", 2)
    for job_id in unique_linkedin_job_ids:
        card = card_by_job_id[job_id]
        prior = tracking_context.get(str(job_id)) or {}
        observations = list(prior.get("observations") or [])
        observation_already_recorded = any(
            str(item.get("ingestion_run_id") or item.get("scrape_run_id") or "") == run_id
            for item in observations
        )
        if not observation_already_recorded:
            observations.append({
                "posted_at": card.get("posted_at"),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "ingestion_run_id": run_id,
            })
        fold = relist_tracking.fold_observations(
            observations,
            min_forward_days=min_forward_days,
            stable_observations=stable_observations,
        )
        prior_trusted_date = supabase_utils._date_part(
            prior.get("latest_trusted_posted_date")
            or (prior.get("observations") or [{}])[-1].get("posted_at")
        )
        pending_event = relist_tracking.latest_pending_event(
            fold, prior.get("accepted_relist_dates")
        )
        if (
            observation_already_recorded
            and pending_event
            and str(pending_event.get("ingestion_run_id") or "") == run_id
        ):
            pending_event = None
        pending_date = relist_tracking.date_part(prior.get("pending_relist_on"))
        if pending_event or pending_date:
            event = pending_event or {
                "relisted_on": pending_date,
                "algorithm_version": relist_tracking.ALGORITHM_VERSION,
                "pending_state_recovery": True,
            }
            card["same_id_relist_candidate"] = True
            card["same_id_relist_date"] = event["relisted_on"]
            card["same_id_relist_evidence"] = event
            card["trigger_evidence"] = {"classification": "relist_candidate", **event}
            relist_candidates.append(str(job_id))
        elif fold["corrections"]:
            card["trigger_evidence"] = {
                "classification": "correction",
                **fold["corrections"][-1],
            }
        else:
            card["trigger_evidence"] = {"classification": "unchanged"}

    if tracking_enabled:
        canonical_by_source = {
            str(job_id): (tracking_context.get(str(job_id)) or {}).get("canonical_job_id")
            for job_id in unique_linkedin_job_ids
        }
        supabase_utils.save_listing_observations(
            normalized_cards,
            run_id=run_id,
            provider="linkedin",
            query_scope=query_scope,
            canonical_by_source=canonical_by_source,
        )
        supabase_utils.save_listing_states(
            normalized_cards,
            tracking_context,
            canonical_by_source=canonical_by_source,
            provider="linkedin",
        )
    logging.info(f"Found {len(scraped_cards)} raw job cards, {len(unique_linkedin_job_ids)} unique IDs after scraping.")


    logging.info("\n--- Starting Filtering Step: Checking against Supabase ---")
    if run_context is None:
        job_ids_set, company_title_set = supabase_utils.get_existing_jobs_from_supabase()
    else:
        job_ids_set, company_title_set = run_context.existing_indexes("linkedin")
    incomplete_metadata_ids = supabase_utils.get_incomplete_linkedin_metadata_ids(unique_linkedin_job_ids)

    new_job_ids_to_process = [
        str(job_id) for job_id in unique_linkedin_job_ids 
        if str(job_id) not in job_ids_set
    ]
    enrichment_limit = getattr(config, "LINKEDIN_METADATA_ENRICH_LIMIT_PER_QUERY", 10)
    query_relist_limit = getattr(config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY", 3)
    if relist_budget is None:
        with _relist_detail_fetches_lock:
            run_relist_remaining = max(
                0,
                getattr(config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN", 20)
                - _relist_detail_fetches_used,
            )
    else:
        run_relist_remaining = max(0, relist_budget)
    relist_limit = min(query_relist_limit, run_relist_remaining)
    relist_job_ids_to_process = [
        job_id for job_id in relist_candidates if job_id in job_ids_set
    ][:relist_limit]
    metadata_job_ids_to_process = [
        str(job_id) for job_id in unique_linkedin_job_ids
        if str(job_id) in incomplete_metadata_ids
    ][:enrichment_limit]
    if limit is not None:
        reserved = min(limit, len(relist_job_ids_to_process))
        new_job_ids_to_process = new_job_ids_to_process[:max(0, limit - reserved)]
    job_ids_to_process = list(dict.fromkeys(
        relist_job_ids_to_process + new_job_ids_to_process + metadata_job_ids_to_process
    ))


    logging.info(f"Found {len(unique_linkedin_job_ids)} unique scraped IDs.")

    logging.info(f"Found {len(job_ids_set)} existing IDs in Supabase.")

    logging.info(f"Identified {len(new_job_ids_to_process)} new job IDs to fetch details for.")

    # Search matches are lane evidence even when the canonical job was saved by
    # an earlier query or lane. Persist these before the detail/new-ID early
    # return; canonical saves below handle new IDs and reposts.
    known_source_ids = [job_id for job_id in unique_linkedin_job_ids if job_id in job_ids_set]
    if not known_source_ids:
        canonical_ids = {}
    elif run_context is None:
        canonical_ids = supabase_utils.get_canonical_job_ids_for_sources(known_source_ids)
    else:
        canonical_ids = run_context.canonical_ids_for_sources("linkedin", known_source_ids)
    membership_job = {
        "lane": lane or canonical_lane_slug(resolved_archetype),
        "archetype": resolved_archetype,
        "search_query": search_query,
        "search_query_id": query_id,
        "search_query_type": query_kind,
        "search_query_language": query_language,
        "search_location_scope": location_scope,
        "geography_id": geography_id,
        "query_scope": query_scope,
    }
    for canonical_id in sorted(set(canonical_ids.values())):
        supabase_utils.upsert_job_archetype_membership(canonical_id, membership_job)

    if not job_ids_to_process:
        if tracking_enabled:
            supabase_utils.finish_ingestion_run(
                run_id,
                status="complete",
                pages_attempted=coverage["pages_attempted"],
                pages_completed=coverage["pages_completed"],
                cards_seen=len(normalized_cards),
                detail_budget_used=0,
                coverage_complete=False,
                coverage_reason="LinkedIn guest recent-window search cannot prove absence",
                page_coverage=coverage["page_coverage"],
            )
        logging.info("No new job IDs to process after filtering.")
        return LinkedInQueryJobs()

    if limit is not None and len(job_ids_to_process) > limit:
        logging.info(f"Truncating job_ids_to_process from {len(job_ids_to_process)} to {limit} to stay within source limit.")
        job_ids_to_process = job_ids_to_process[:limit]

    logging.info(f"\n--- Starting Phase 2: Fetching Job Details for {len(job_ids_to_process)} IDs ---")
    detailed_new_jobs = []
    processed_count = 0
    terminal_unavailable_count = 0

    ids_to_fetch = job_ids_to_process
    for job_id in ids_to_fetch:
        detail_fetch_options = {"search_card": card_by_job_id.get(job_id)}
        if request_limiter is not None:
            detail_fetch_options["request_limiter"] = request_limiter
        if durable_gate is not None:
            detail_fetch_options["durable_gate"] = durable_gate
        if user_agent is not None:
            detail_fetch_options["user_agent"] = user_agent
        try:
            detail_result = _fetch_linkedin_job_details(job_id, **detail_fetch_options)
        except (LinkedInAccessDenied, LinkedInRequestFailed) as exc:
            if tracking_enabled:
                supabase_utils.finish_ingestion_run(
                    run_id,
                    status="failed",
                    pages_attempted=coverage["pages_attempted"],
                    pages_completed=coverage["pages_completed"],
                    cards_seen=len(normalized_cards),
                    detail_budget_used=processed_count,
                    coverage_complete=False,
                    coverage_reason=str(exc),
                    page_coverage=coverage["page_coverage"],
                )
            raise
        if detail_result is LINKEDIN_DETAIL_UNAVAILABLE:
            terminal_unavailable_count += 1
        elif detail_result:
            details, detail_metadata = detail_result
            details = {**details, **detail_metadata}
            if not fetch_descriptions:
                details["description"] = None
            details["search_query"] = search_query
            if query_id is not None:
                details["search_query_id"] = query_id
            if query_kind is not None:
                details["search_query_type"] = query_kind
            if query_language is not None:
                details["search_query_language"] = query_language
            if lane is not None:
                details["lane"] = lane
            if geography_id is not None:
                details["geography_id"] = geography_id
            if any(value is not None for value in (query_id, query_kind, query_language, lane, location_scope, geography_id)):
                details["query_scope"] = query_scope
            details["archetype"] = resolved_archetype
            details["filter_profile"] = resolved_filter_profile
            if location_scope:
                details["search_location_scope"] = location_scope
            details["scrape_run_id"] = run_id
            if job_id in relist_job_ids_to_process:
                details["same_id_relist_candidate"] = bool(
                    getattr(config, "ENABLE_LINKEDIN_RELIST_EFFECTS", True)
                )
                details["same_id_relist_date"] = card_by_job_id[job_id].get("same_id_relist_date")
                details["same_id_relist_evidence"] = card_by_job_id[job_id].get("same_id_relist_evidence")
                if relist_budget is None:
                    with _relist_detail_fetches_lock:
                        _relist_detail_fetches_used += 1
            description = details.get('description')
            if not fetch_descriptions or (description and description.strip()):
                if 'job_id' in details and details['job_id'] is not None:
                    detailed_new_jobs.append(details)
                    processed_count += 1
                else:
                    
                    logging.warning(f"Fetched details for {job_id} but missing 'job_id' key. Skipping.")
            else:
                
                logging.warning(f"Skipping job ID {job_id} due to missing or empty description.") 
        else:
            
            logging.warning(f"Skipping job ID {job_id} as detail fetching failed or returned no data.") 


    completed_detail_count = processed_count + terminal_unavailable_count
    logging.info(f"--- Finished Phase 2: Successfully fetched details for {processed_count} new job(s) ---")
    if tracking_enabled:
        supabase_utils.finish_ingestion_run(
            run_id,
            status="complete" if completed_detail_count == len(ids_to_fetch) else "incomplete",
            pages_attempted=coverage["pages_attempted"],
            pages_completed=coverage["pages_completed"],
            cards_seen=len(normalized_cards),
            detail_budget_used=len(ids_to_fetch),
            coverage_complete=False,
            coverage_reason=(
                "LinkedIn guest recent-window search cannot prove absence"
                if completed_detail_count == len(ids_to_fetch)
                else f"detail validation accepted {completed_detail_count} of {len(ids_to_fetch)} selected IDs"
            ),
            page_coverage=coverage["page_coverage"],
        )
    processing_complete = completed_detail_count == len(ids_to_fetch)
    return LinkedInQueryJobs(
        detailed_new_jobs,
        processing_complete=processing_complete,
        incomplete_reason=(
            None
            if processing_complete
            else f"detail validation accepted {completed_detail_count} of {len(ids_to_fetch)} selected IDs"
        ),
    )

def _fetch_careers_future_jobs(search_query: str) -> list:
    """
    Fetches job items from CareersFuture based on the provided search query.
    This involves:
    1. Getting skill suggestions based on the search query.
    2. Using these skill UUIDs to search for jobs.
    3. Handling pagination to retrieve all job results.
    4. Returning a list of all collected job item dictionaries.

    Args:
        search_query (str): The job title or keywords to search for.

    Returns:
        list: A list of job item dictionaries. Returns an empty list if an error occurs
              or if no jobs are found.
    """


    careers_future_suggestions_api_url = "https://api.mycareersfuture.gov.sg/v2/skills/suggestions"
    careers_future_search_api_base_url =  "https://api.mycareersfuture.gov.sg/v2/search"

    skillUuids = []

    # --- 1. Get Skill Suggestions ---
    skills_suggestions_payload = {'jobTitle': search_query}

    try:
        logging.info(f"Fetching skill suggestions for query: '{search_query}' from {careers_future_suggestions_api_url}")
        skills_suggestions_response = requests.post(
            careers_future_suggestions_api_url, 
            data=skills_suggestions_payload,
            timeout=config.REQUEST_TIMEOUT
            )

        skills_suggestions_response.raise_for_status()
        skills_data = skills_suggestions_response.json()
        skills_list = skills_data.get('skills', [])
        skillUuids = [skill_dict['uuid'] for skill_dict in skills_list if 'uuid' in skill_dict]
        logging.info(f"Successfully retrieved {len(skillUuids)} skill UUIDs for '{search_query}'.")
        if not skillUuids:
            logging.warning(f"No skill UUIDs found for query '{search_query}'. Job search will proceed without specific skill filtering.")


    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else 'N/A'
        response_text = http_err.response.text if http_err.response is not None else 'N/A'
        logging.error(f"HTTP error during skill suggestions: {http_err} - Status: {status_code}")
        logging.debug(f"Skill suggestions error response content: {response_text[:500]}") 
        return []
    except requests.exceptions.RequestException as req_err: 
        logging.error(f"Request exception during skill suggestions: {req_err}")
        return []
    except json.JSONDecodeError:
        content_for_log = skills_suggestions_response.text if 'skills_suggestions_response' in locals() and skills_suggestions_response else "N/A"
        logging.error(f"Could not decode JSON response for skill suggestions. Content: {content_for_log[:500]}")
        return []

    # --- 2. Search for Jobs and Handle Pagination ---
    all_job_items = []
    total_api_calls_for_search = 0

    # Initial search URL with default limit and page
    current_search_url = f"{careers_future_search_api_base_url}?limit=100&page=0"
    search_payload = {
        'sessionId':"",
        'search': search_query,
        'categories':config.CAREERS_FUTURE_SEARCH_CATEGORIES,
        'employmentTypes': config.CAREERS_FUTURE_SEARCH_EMPLOYMENT_TYPES,
        'postingCompany' : [],
        'sortBy': ["new_posting_date"],
        'skillUuids': skillUuids,

    }

    try:
        while current_search_url:
            total_api_calls_for_search += 1
            logging.info(f"Job search API call {total_api_calls_for_search}: POST to {current_search_url}")
        
            search_response = requests.post(current_search_url, json=search_payload)
            search_response.raise_for_status()
            search_results_data  = search_response.json()

            current_page_jobs = search_results_data.get('results', [])
            all_job_items.extend(current_page_jobs)

            logging.info(f"Retrieved {len(current_page_jobs)} job items from this page. Total items collected: {len(all_job_items)}.")

            # Log total results reported by API 
            if 'total' in search_results_data and total_api_calls_for_search == 1:
                logging.info(f"API reports total potential jobs matching criteria: {search_results_data['total']}")
            
            # Get the next page URL. The API provides a full URL.
            next_page_link_info = search_results_data.get("_links", {}).get("next", {})
            current_search_url = next_page_link_info.get("href") if next_page_link_info else None 

            if current_search_url:
                logging.debug(f"Next page URL for job search: {current_search_url}")
            else:
                logging.info("No more job pages to fetch.")

        logging.info(f"Completed job search. Total API calls made for search: {total_api_calls_for_search}.")
    
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else 'N/A'
        response_text = http_err.response.text if http_err.response is not None else 'N/A'
        logging.error(f"HTTP error during job search: {http_err} - Status: {status_code}")
        logging.debug(f"Job search error response content: {response_text[:500]}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request exception during job search: {req_err}")
    except json.JSONDecodeError:
        content_for_log = search_response.text if 'search_response' in locals() and search_response else "N/A"
        logging.error(f"Could not decode JSON response during job search. Content: {content_for_log[:500]}")

    # --- 3. Return all collected job items ---
    if not all_job_items:
        logging.info(f"No job items were collected for query '{search_query}'.")
        return [] 

    logging.info(f"Returning {len(all_job_items)} total job items for query '{search_query}'.")
    return all_job_items

def _fetch_careers_future_job_details(job_id: str) -> dict | None:
    """
    Fetch job details from CareersFuture based on the provided job ID.

    Args:
        job_id (str): The UUID of the job to fetch details for.

    Returns:
        dict | None: A dictionary containing the job details if successful,
                      None otherwise.
    """
    if not job_id:
        logging.warning("Job ID is missing or empty. Cannot fetch details.")
        return None

    api_url = f"https://api.mycareersfuture.gov.sg/v2/jobs/{job_id}"
    
    logging.info(f"Attempting to fetch job details for ID: {job_id} from URL: {api_url}")

    try:
        response = requests.get(api_url, timeout=config.REQUEST_TIMEOUT) 

        response.raise_for_status()

        job_data = response.json()
        logging.info(f"Successfully fetched and parsed job details for ID: {job_id}")

        raw_description_html = job_data.get('description', '')
        # Convert HTML description directly to Markdown (no LLM needed)
        markdown_description = None 
        if raw_description_html.strip(): 
            markdown_description = convert_html_to_markdown(raw_description_html)
        else:
            logging.warning(f"Raw description was empty for Careers Future job ID {job_id}. Skipping conversion.") 

        job_details = {
            'job_id': job_data.get('uuid'),
            'company': _get_careers_future_job_company_name(job_data),
            'job_title': job_data.get('title'),
            'location': 'Singapore',
            'level': job_data.get('positionLevels', [{'position': 'Not applicable'}])[0].get('position', 'Not applicable'),
            'provider': 'careers_future',
            'description': markdown_description, 
            'posted_at': job_data.get('metadata', {}).get('createdAt', ''),
        }

        return job_details

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else 'N/A'
        response_text = http_err.response.text if http_err.response is not None else 'N/A'
        if status_code == 404:
            logging.warning(f"Job details not found (404) for ID: {job_id} at {api_url}.")
        else:
            logging.error(f"HTTP error occurred while fetching job details for ID '{job_id}': {http_err} - Status: {status_code}")
            logging.debug(f"Error response content: {response_text[:500]}") 
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred while fetching job details for ID '{job_id}': {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred while fetching job details for ID '{job_id}': {timeout_err}")
    except requests.exceptions.RequestException as req_err: 
        logging.error(f"An error occurred during the request for job details for ID '{job_id}': {req_err}")
    except json.JSONDecodeError:
        content_for_log = response.text if 'response' in locals() and response else "N/A"
        logging.error(f"Failed to decode JSON response for job details for ID '{job_id}'. Content: {content_for_log[:500]}")
    
    return None # Return None in case of any error

def process_careers_future_query(search_query: str, limit: int = None) -> list:
    """
    Fetch jobs from CareersFuture and return them as a list of dictionaries.
    """
    # 1. Fetch all potential job items from CareersFuture search
    careers_future_jobs = _fetch_careers_future_jobs(search_query)
    if not careers_future_jobs:
        print("No job items found in Phase 1. Skipping detail fetching.")
        return []

    # 2. Fetch existing job identifiers from Supabase
    logging.info("Phase 2: Fetching existing job identifiers from Supabase...")
    try:
        job_ids_set_supabase, company_title_set_supabase = supabase_utils.get_existing_jobs_from_supabase()
        logging.info(f"Phase 2: Supabase returned {len(job_ids_set_supabase)} existing IDs and {len(company_title_set_supabase)} company/title pairs.")
    except Exception as e:
        logging.error(f"Failed to fetch existing jobs from Supabase: {e}")
        logging.warning("Proceeding without Supabase data; all fetched jobs will be considered new.")
        job_ids_set_supabase = set()
        company_title_set_supabase = set()

    # 3. Filter the fetched jobs
    logging.info("Phase 3: Filtering fetched jobs against Supabase data...")
    new_job_ids_to_process = []
    skipped_by_id_count = 0
    skipped_by_combo_count = 0

    for job_item in careers_future_jobs:
        if not isinstance(job_item, dict):
            logging.warning(f"Skipping invalid job item (not a dict): {str(job_item)[:100]}")
            continue

        job_uuid = str(job_item.get('uuid'))
        
        # Check 1: Does the UUID already exist in Supabase?
        if job_uuid and job_uuid in job_ids_set_supabase:
            logging.debug(f"Skipping job (ID exists in Supabase): UUID='{job_uuid}', Title='{job_item.get('title', 'N/A')}'")
            skipped_by_id_count += 1
            continue # Skip this job

        # Prepare for Check 2: Company & Title combination
        company_name = _get_careers_future_job_company_name(job_item)
        job_title = job_item.get('title')

        normalized_company = None
        normalized_title = None

        if company_name:
            normalized_company = company_name.strip().lower()
        if job_title:
            normalized_title = job_title.strip().lower()
        
        if normalized_company and normalized_title:
            company_title_key = (normalized_company, normalized_title)
            if company_title_key in company_title_set_supabase:
                logging.debug(f"Skipping job (Company/Title combo exists in Supabase): UUID='{job_uuid}', Company='{normalized_company}', Title='{normalized_title}'")
                skipped_by_combo_count +=1
                continue 
        elif job_uuid: 
            logging.debug(f"Job UUID='{job_uuid}' has no company/title for combo check. Will be added if ID is new.")
        else: 
             logging.warning(f"Job item has no UUID and insufficient company/title for matching: {str(job_item)[:100]}")


        new_job_ids_to_process.append(job_uuid) 

    # 4. Fetch details ONLY for the genuinely new job IDs
    if limit is not None and len(new_job_ids_to_process) > limit:
        logging.info(f"Truncating new_job_ids_to_process from {len(new_job_ids_to_process)} to {limit} to stay within source limit.")
        new_job_ids_to_process = new_job_ids_to_process[:limit]

    print(f"\n--- Phase 4: Fetching Job Details for {len(new_job_ids_to_process)} New Jobs ---")
    detailed_new_jobs = []
    processed_count = 0

    for job_id in new_job_ids_to_process:
        details = _fetch_careers_future_job_details(job_id)
        if details:
            # --- NEW: Check for description before adding ---
            description = details.get('description')
            if description and description.strip(): # Ensure it's not None or an empty/whitespace string
                if 'job_id' in details and details['job_id'] is not None:
                    detailed_new_jobs.append(details)
                    processed_count += 1
                else:
                    
                    logging.warning(f"Fetched details for {job_id} but missing 'job_id' key. Skipping.")
            else:
                
                logging.warning(f"Skipping job ID {job_id} due to missing or empty description.") 
        else:
            
            logging.warning(f"Skipping job ID {job_id} as detail fetching failed or returned no data.") 



    logging.info(f"--- Finished Phase 4: Successfully fetched details for {processed_count} new job(s) ---")
    return detailed_new_jobs


def _run_database_configured_linkedin(
    scrape_config: ScrapeConfiguration,
    saved_job_ids: list[str],
) -> bool:
    settings = scrape_config.settings
    if not settings.scraping_enabled:
        logging.info("LinkedIn scraping disabled by scrape_settings.scraping_enabled")
        return False

    if os.getenv("LINKEDIN_DISCOVERY_MODE", "adaptive_queue").strip().lower() == "adaptive_queue":
        import linkedin_discovery

        archetype_override = os.getenv("SCRAPE_ARCHETYPE")
        executions = build_search_executions(
            scrape_config,
            archetype_override=archetype_override,
        )
        execution_locations = {
            execution.lane.archetype: execution.geography.location
            for execution in executions
        }
        runtime_profiles = {
            lane.archetype: _career_lane_runtime_profile(
                lane, execution_locations.get(lane.archetype, "")
            )
            for lane in scrape_config.lanes
        }
        run_context = supabase_utils.CanonicalRunContext()
        previous_repost_dedup = config.ENABLE_REPOST_DEDUP
        config.ENABLE_REPOST_DEDUP = settings.deduplicate_jobs

        def save_adaptive_details(task, worker_id, job):
            if not settings.fetch_descriptions:
                job = {**job, "description": None}
            lane = job.get("lane") or job.get("archetype")
            if lane not in runtime_profiles:
                raise ValueError(f"adaptive detail has unknown lane {lane!r}")
            return supabase_utils.apply_linkedin_discovery_task_canonical(
                task,
                worker_id,
                job,
                run_context=run_context,
                runtime_profiles=runtime_profiles,
            )

        try:
            result = linkedin_discovery.run_discovery(
                scrape_config,
                executions,
                parse_cards=_extract_linkedin_search_cards,
                detail_fetch=_fetch_linkedin_job_details,
                save_details=save_adaptive_details,
                partial=bool(archetype_override and archetype_override.strip()),
            )
        finally:
            config.ENABLE_REPOST_DEDUP = previous_repost_dedup
        saved_job_ids.extend(result.canonical_job_ids)
        os.environ["LINKEDIN_DISCOVERY_CYCLE_ID"] = str(result.cycle_id)
        output_path = os.getenv("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as output:
                output.write(f"discovery_cycle_id={result.cycle_id}\n")
                output.write(f"discovery_sequence={result.discovery_sequence}\n")
        return result.advance_watermark

    global _relist_detail_fetches_used
    last_success_at = supabase_utils.get_last_successful_scrape_at() or config.LINKEDIN_LAST_SUCCESS_AT
    configured_lookback_hours = max(
        settings.lookback_days * 24,
        config.LINKEDIN_LOOKBACK_HOURS,
    )
    lookback_hours = resolve_linkedin_lookback_hours(
        last_success_at,
        configured_hours=configured_lookback_hours,
        overlap_hours=0,
        max_hours=max(config.LINKEDIN_MAX_LOOKBACK_HOURS, configured_lookback_hours),
    )
    posting_date_filter = f"r{lookback_hours * 3600}"
    logging.info(
        "LinkedIn configured search: version=%s lookback=%s hours scopes=%s",
        scrape_config.version,
        lookback_hours,
        ",".join(sorted({scope.value for lane in scrape_config.lanes for scope in lane.locations})),
    )

    # The legacy persistence helper reads this setting. Restore it so invoking
    # orchestration in-process cannot leak database configuration to later work.
    previous_repost_dedup = config.ENABLE_REPOST_DEDUP
    config.ENABLE_REPOST_DEDUP = settings.deduplicate_jobs
    _relist_detail_fetches_used = 0
    run_context = supabase_utils.CanonicalRunContext()
    request_limiter = LinkedInRequestLimiter(
        minimum_interval_ms=max(
            2_500,
            int(settings.options.get("global_request_interval_ms", settings.request_delay_ms)),
        ),
        jitter_ms=int(settings.options.get("request_jitter_ms", 1_500)),
    )
    durable_gate = DurableLinkedInRequestGate("legacy-scraper")
    user_agent = os.getenv("LINKEDIN_USER_AGENT") or user_agents.USER_AGENTS[0]
    archetype_override = os.getenv("SCRAPE_ARCHETYPE")
    executions = build_search_executions(
        scrape_config,
        archetype_override=archetype_override,
    )
    if archetype_override and archetype_override.strip():
        logging.info(
            "LinkedIn scrape restricted by SCRAPE_ARCHETYPE=%s",
            executions[0].lane.archetype,
        )
    query_relist_limit = getattr(config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY", 3)
    run_relist_remaining = getattr(config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN", 20)
    work_items = []
    for execution in executions:
        runtime_profile = _lane_runtime_archetype_config(execution)
        relist_budget = None
        if settings.concurrent_queries > 1:
            relist_budget = min(query_relist_limit, max(0, run_relist_remaining))
            run_relist_remaining -= relist_budget
        work_items.append((execution, runtime_profile, relist_budget))

    def fetch_execution(work_item):
        execution, runtime_profile, relist_budget = work_item
        lane_slug = execution.lane.archetype
        logging.info(
            "Processing LinkedIn lane=%s query_id=%s query_kind=%s location_scope=%s geography=%s query=%r",
            lane_slug,
            execution.query.query_id,
            execution.query.query_type.value,
            execution.geography.location_scope.value,
            execution.geography.geography_id,
            execution.query.query,
        )
        jobs = process_linkedin_query(
            search_query=execution.query.query,
            location=execution.geography.location,
            limit=settings.max_jobs_per_query,
            archetype=lane_slug,
            filter_profile=f"{lane_slug}_v1",
            posting_date_filter=posting_date_filter,
            query_id=execution.query.query_id,
            query_kind=execution.query.query_type.value,
            query_language=execution.query.language,
            lane=lane_slug,
            location_scope=execution.geography.location_scope.value,
            geography_id=execution.geography.geography_id,
            geo_id=execution.geography.geo_id,
            max_start=_linkedin_max_start_for_pages(settings.max_pages_per_query),
            request_delay_ms=settings.request_delay_ms,
            fetch_descriptions=settings.fetch_descriptions,
            geo_id_is_explicit=True,
            runtime_profile=runtime_profile,
            relist_budget=relist_budget,
            run_context=run_context,
            request_limiter=request_limiter,
            durable_gate=durable_gate,
            user_agent=user_agent,
        )
        return execution, runtime_profile, jobs

    executor = None
    try:
        # Supabase's synchronous client owns a shared HTTP/2 connection pool and
        # is not safe to drive concurrently from these query workers. Keep the
        # bounded executor for provider-only/test integrations, but serialize
        # production fetches that also perform tracking reads and writes.
        use_concurrent_fetches = (
            settings.concurrent_queries > 1
            and process_linkedin_query.__module__ != __name__
        )
        if not use_concurrent_fetches:
            fetched_results = map(fetch_execution, work_items)
        else:
            executor = ThreadPoolExecutor(max_workers=settings.concurrent_queries)
            # Finish all provider work before deterministic serialized writes.
            fetched_results = list(executor.map(fetch_execution, work_items))

        # Canonical persistence and lane-state writes remain serialized.
        incomplete_queries = []
        for execution, runtime_profile, jobs in fetched_results:
            if not getattr(jobs, "processing_complete", True):
                incomplete_queries.append(
                    (
                        execution.query.query_id,
                        getattr(jobs, "incomplete_reason", None) or "query processing incomplete",
                    )
                )
            if not jobs:
                continue
            lane_slug = execution.lane.archetype
            save_result = supabase_utils.save_linkedin_jobs_canonicalized_with_mapping(
                jobs,
                run_context=run_context,
            )
            saved_job_ids.extend(save_result.canonical_ids)
            for job, canonical_id in zip(jobs, save_result.canonical_ids_by_input):
                if canonical_id:
                    supabase_utils.persist_lane_filter_state(
                        canonical_id,
                        lane_slug,
                        job,
                        runtime_profile=runtime_profile,
                    )
        if incomplete_queries:
            summary = "; ".join(
                f"{query_id}: {reason}" for query_id, reason in incomplete_queries
            )
            raise LinkedInRequestFailed(
                f"LinkedIn configured query coverage incomplete: {summary}"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        logging.info(
            "LinkedIn request pacing summary: requests=%s wait_seconds=%.2f",
            request_limiter.request_count,
            request_limiter.total_wait_seconds,
        )
        config.ENABLE_REPOST_DEDUP = previous_repost_dedup
    return not bool(archetype_override and archetype_override.strip())


def main() -> list[str]:
    """Run configured scrapers and return the canonical job IDs that were saved."""
    saved_job_ids: list[str] = []
    complete_linkedin_coverage = False

    # Get jobs from LinkedIn
    if "linkedin" in config.SCRAPING_SOURCES:
        logging.info("\n--- Starting LinkedIn Job Scraping ---")
        scrape_config = load_scrape_configuration(db=supabase_utils.supabase)
        complete_linkedin_coverage = _run_database_configured_linkedin(
            scrape_config, saved_job_ids
        )
    else:
        logging.info("\n--- Skipping LinkedIn Job Scraping per config ---")

    # Get jobs from Careers Future
    if "careers_future" in config.SCRAPING_SOURCES:
        logging.info(f"\n--- Starting Careers Future Job Scraping ---")
        max_jobs_per_search = config.MAX_JOBS_PER_SEARCH.get("careers_future", getattr(config, 'DEFAULT_MAX_JOBS_PER_SEARCH', 10))
        for query in config.CAREERS_FUTURE_SEARCH_QUERIES:
            logging.info(f"\n{'='*20} Processing Careers Future Search Query: '{query}' {'='*20}")

            # 1. Process the query: Scrape IDs, filter, fetch new details
            new_careers_future_job_details = process_careers_future_query(query, limit=max_jobs_per_search)

            # 2. Save the NEW scraped data to Supabase
            if new_careers_future_job_details:
                logging.info(f"\n--- Saving {len(new_careers_future_job_details)} new job(s) for query '{query}' ---")
                saved_job_ids.extend(
                    supabase_utils.save_jobs_canonicalized(
                        new_careers_future_job_details
                    )
                )
            else:
                logging.info(f"\nNo new job details were fetched or processed for query '{query}'.")
    else:
        logging.info("\n--- Skipping Careers Future Job Scraping per config ---")

    # --- End of Script ---
    logging.info(f"\n{'='*20} Job scraping script finished {'='*20}")
    logging.info(f"Total new jobs saved across all queries: {len(saved_job_ids)}")
    adaptive_linkedin = (
        "linkedin" in config.SCRAPING_SOURCES
        and os.getenv("LINKEDIN_DISCOVERY_MODE", "adaptive_queue").strip().lower() == "adaptive_queue"
    )
    if adaptive_linkedin:
        logging.info("Discovery cycle sealing owns the LinkedIn operational watermark")
    elif "linkedin" not in config.SCRAPING_SOURCES or complete_linkedin_coverage:
        if not supabase_utils.record_scrape_success():
            raise RuntimeError("Failed to persist scrape success watermark")
    else:
        logging.info(
            "Skipped global scrape watermark for partial SCRAPE_ARCHETYPE recovery"
        )
    return saved_job_ids


# --- Main Execution ---
if __name__ == "__main__":
    main()
