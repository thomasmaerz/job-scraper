import hashlib
import html
import json
import math
import random
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from bs4 import BeautifulSoup

import config


Category = Literal[
    "software_engineering", "backend", "frontend", "fullstack", "mobile",
    "devops", "sre", "network_engineering", "data_engineering", "data_science",
    "data_analytics", "ml_ai", "ai_engineering", "qa", "security", "hardware",
    "embedded", "blockchain", "architecture", "design", "engineering_design",
    "product", "project_management", "management", "marketing", "sales",
    "support", "business_analysis", "solutions_engineering", "developer_relations",
    "technical_writing", "recruiting", "hr", "finance", "legal", "operations",
    "customer_success", "other",
]
Seniority = Literal["", "intern", "junior", "middle", "senior", "lead", "staff", "principal", "c_level"]

REMOTE_RE = re.compile(r"(?<![A-Za-z0-9_])remote(?![A-Za-z0-9_])", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]*>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class FreehireClassificationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    category: Category
    seniority: Seniority
    confidence: float = Field(ge=0.0, le=1.0)


class FreehireClassificationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[FreehireClassificationItem]


@dataclass
class ClassificationOutcome:
    results: dict[str, FreehireClassificationItem] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    requests: int = 0
    retries: int = 0
    splits: int = 0
    result_models: dict[str, str] = field(default_factory=dict)
    global_error: str | None = None


SYSTEM_PROMPT = f"""Classify LinkedIn jobs for the pinned Freehire vocabulary.

Return each requested job_id exactly once. Select exactly one category from:
{', '.join(sorted(config.FREEHIRE_CATEGORIES))}

Select seniority from: {', '.join(repr(value) for value in sorted(config.FREEHIRE_SENIORITY_LEVELS))}.
Use other when no narrower category is defensible. Use an empty seniority rather than guessing.
Classify records independently. Do not invent IDs or labels.
"""


def normalize_visible_text(value) -> str:
    text = str(value or "")
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = unicodedata.normalize("NFKC", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(
        ["script", "style", "template", "noscript", "iframe", "svg", "canvas"]
    ):
        tag.decompose()
    for tag in soup.find_all(True):
        style = re.sub(r"\s+", "", str(tag.get("style") or "")).casefold()
        if (
            tag.has_attr("hidden")
            or str(tag.get("aria-hidden") or "").casefold() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        ):
            tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split()).casefold()


def classify_remote(job: dict) -> tuple[bool, dict | None]:
    for field_name in ("job_title", "location", "description"):
        visible = normalize_visible_text(job.get(field_name))
        match = REMOTE_RE.search(visible)
        if match:
            return True, {
                "field": field_name,
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
    return False, None


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_classification_hash(job: dict) -> str:
    return _canonical_hash({
        "description": normalize_visible_text(job.get("description")),
        "job_id": str(job.get("job_id") or ""),
        "job_title": normalize_visible_text(job.get("job_title")),
        "level": normalize_visible_text(job.get("level")),
        "location": normalize_visible_text(job.get("location")),
        "schema_version": config.FREEHIRE_COMPAT_SCHEMA_VERSION,
    })


SOURCE_SNAPSHOT_FIELDS = (
    "job_id", "latest_job_id", "company", "job_title", "level", "location",
    "description", "posted_at", "scraped_at", "first_seen_at", "last_seen_at",
    "last_seen_posted_at", "last_checked", "detail_metadata_checked_at", "salary_text",
    "salary_min", "salary_max", "salary_currency", "applicant_count",
    "applicant_count_text", "applicant_count_type", "recruiter_name",
    "recruiter_profile_url", "recruiter_identifier", "original_job_id", "seen_count",
    "posting_wave_count", "repost_count", "same_id_relist_count", "listing_instances",
    "archetype", "search_query", "filter_profile", "is_filtered",
    "is_entry_level_filtered", "filter_reason", "description_fingerprint",
)


def source_snapshot(job: dict) -> dict:
    """Database-source evidence that must remain unchanged for a worker lease."""
    return {field: job.get(field) for field in SOURCE_SNAPSHOT_FIELDS}


def compute_import_hash(
    job: dict,
    category: str | None = None,
    seniority: str | None = None,
    is_remote: bool | None = None,
) -> str:
    if is_remote is None:
        is_remote, _ = classify_remote(job)
    published_fields = SOURCE_SNAPSHOT_FIELDS + (
        "freehire_remote_evidence", "freehire_compat_confidence",
        "freehire_compat_classified_at", "freehire_compat_model",
        "freehire_compat_prompt_version", "freehire_compat_schema_version",
        "freehire_compat_provenance",
    )
    payload = {field: job.get(field) for field in published_fields}
    payload.update({
        "freehire_category": category if category is not None else job.get("freehire_category"),
        "freehire_seniority": seniority if seniority is not None else (job.get("freehire_seniority") or ""),
        "is_remote": bool(is_remote),
        "live_listing_id": str(job.get("latest_job_id") or job.get("job_id") or ""),
    })
    return _canonical_hash(payload)


def normalize_category(value) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in config.FREEHIRE_CATEGORIES else None


def normalize_seniority(value) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in config.FREEHIRE_SENIORITY_LEVELS else None


def source_seniority(level) -> str:
    normalized = normalize_visible_text(level).replace("-", " ").replace("_", " ")
    exact = {
        "intern": "intern",
        "internship": "intern",
        "junior": "junior",
        "middle": "middle",
        "senior": "senior",
        "lead": "lead",
        "staff": "staff",
        "principal": "principal",
        "c level": "c_level",
    }
    return exact.get(normalized, "")


def model_name(client=None) -> str:
    if client is not None:
        used = getattr(client, "last_model_used", None)
        if used:
            return used
        chain = getattr(client, "model_chain", None)
        if chain:
            return chain[0]
        if getattr(client, "model", None):
            return client.model
    return config.FREEHIRE_CLASSIFY_MODEL_CHAIN[0]


def _estimate_tokens(text: str, model: str | None = None) -> int:
    if model:
        try:
            import litellm

            return int(litellm.token_counter(model=model, text=text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / config.FREEHIRE_CHARS_PER_TOKEN))


def build_job_block(job: dict) -> str:
    description = normalize_visible_text(job.get("description"))
    description = description[:config.FREEHIRE_DESCRIPTION_MAX_CHARS]
    source_level = source_seniority(job.get("level"))
    return "\n".join((
        f"Job ID: {job.get('job_id', '')}",
        f"Title: {normalize_visible_text(job.get('job_title'))}",
        f"Location: {normalize_visible_text(job.get('location'))}",
        f"LinkedIn level: {normalize_visible_text(job.get('level'))}",
        f"Exact source seniority (authoritative when non-empty): {source_level}",
        f"Description: {description}",
    ))


def pack_batches(
    jobs: list[dict],
    token_budget: int | None = None,
    max_jobs: int | None = None,
    model: str | None = None,
) -> list[list[dict]]:
    token_budget = token_budget or config.FREEHIRE_INPUT_TOKEN_BUDGET
    max_jobs = min(max_jobs or config.FREEHIRE_MAX_BATCH_JOBS, 50)
    fixed_tokens = _estimate_tokens(SYSTEM_PROMPT, model=model) + 100
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = fixed_tokens
    for job in jobs:
        block_tokens = _estimate_tokens(build_job_block(job), model=model)
        proposed_count = len(current) + 1
        proposed_tokens = current_tokens + block_tokens + (
            proposed_count * config.FREEHIRE_OUTPUT_TOKENS_PER_JOB
        )
        if current and (proposed_count > max_jobs or proposed_tokens > token_budget):
            batches.append(current)
            current = []
            current_tokens = fixed_tokens
        current.append(job)
        current_tokens += block_tokens
    if current:
        batches.append(current)
    return batches


def _parse_response(raw_response: str, expected_ids: set[str]) -> tuple[dict[str, FreehireClassificationItem], list[str]]:
    payload = json.loads(raw_response)
    raw_jobs = payload if isinstance(payload, list) else payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("Classification response must contain a jobs list")

    by_id: dict[str, list[FreehireClassificationItem]] = {}
    errors = []
    for raw_item in raw_jobs:
        try:
            item = FreehireClassificationItem.model_validate(raw_item)
        except Exception as exc:
            raw_id = str(raw_item.get("job_id")) if isinstance(raw_item, dict) and raw_item.get("job_id") is not None else "unknown"
            errors.append(f"invalid job_id={raw_id}: {exc}")
            continue
        if item.job_id not in expected_ids:
            errors.append(f"unexpected job_id={item.job_id}")
            continue
        by_id.setdefault(item.job_id, []).append(item)

    parsed = {}
    for job_id, items in by_id.items():
        if len(items) != 1:
            errors.append(f"duplicate job_id={job_id}")
            continue
        parsed[job_id] = items[0]
    return parsed, errors


def _request_prompt(jobs: list[dict]) -> str:
    return "Classify every job below.\n\n" + "\n\n".join(build_job_block(job) for job in jobs)


def classify_batch(
    jobs: list[dict],
    client=None,
    max_retries: int | None = None,
    max_requests: int | None = None,
    sleep_fn=time.sleep,
) -> ClassificationOutcome:
    if client is None:
        from llm_client import freehire_classify_client

        client = freehire_classify_client
    max_retries = max_retries or config.FREEHIRE_CLASSIFY_MAX_RETRIES
    max_requests = max_requests or config.FREEHIRE_CLASSIFY_REQUEST_BUDGET
    expected = {str(job["job_id"]): job for job in jobs if job.get("job_id") is not None}
    request_state = {"used": 0}

    def global_failure(exc: Exception) -> bool:
        value = f"{type(exc).__name__}: {exc}".casefold()
        return any(token in value for token in (
            "authentication", "authorization", "permission", "api key", "invalid key",
            "quota", "rate limit", "429", "resource_exhausted", "connection",
            "transport", "timeout", "timed out", "dns", "503", "502", "500",
        ))

    def classify_subset(pending: dict[str, dict]) -> ClassificationOutcome:
        outcome = ClassificationOutcome()
        last_error = "missing classification result"
        remaining = dict(pending)
        for attempt in range(max_retries):
            if not remaining:
                return outcome
            if request_state["used"] >= max_requests:
                last_error = f"request budget exhausted ({max_requests})"
                break
            try:
                request_state["used"] += 1
                raw = client.generate_content(
                    prompt=_request_prompt(list(remaining.values())),
                    system_prompt=SYSTEM_PROMPT,
                    reasoning_effort="low",
                    response_format=FreehireClassificationBatch,
                    max_api_attempts=1,
                )
                outcome.requests += 1
                parsed, errors = _parse_response(raw, set(remaining))
                outcome.results.update(parsed)
                used_model = model_name(client)
                outcome.result_models.update({job_id: used_model for job_id in parsed})
                remaining = {job_id: job for job_id, job in remaining.items() if job_id not in parsed}
                last_error = "; ".join(errors) or (
                    "missing job_ids: " + ", ".join(sorted(remaining))
                )
            except Exception as exc:
                outcome.requests += 1
                last_error = str(exc)
                if global_failure(exc):
                    outcome.global_error = last_error
                    outcome.failures.update({job_id: last_error for job_id in remaining})
                    return outcome
            if remaining and attempt < max_retries - 1:
                outcome.retries += 1
                delay = config.FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS * (2 ** attempt)
                sleep_fn(delay + random.uniform(0, delay if delay else 0))

        if len(remaining) > 1 and request_state["used"] < max_requests:
            values = list(remaining.values())
            midpoint = len(values) // 2
            outcome.splits += 1
            halves = (values[:midpoint], values[midpoint:])
            for half_index, half in enumerate(halves):
                child = classify_subset({str(job["job_id"]): job for job in half})
                outcome.results.update(child.results)
                outcome.result_models.update(child.result_models)
                outcome.failures.update(child.failures)
                outcome.requests += child.requests
                outcome.retries += child.retries
                outcome.splits += child.splits
                if child.global_error:
                    outcome.global_error = child.global_error
                    for unprocessed_half in halves[half_index + 1:]:
                        for unprocessed in unprocessed_half:
                            outcome.failures.setdefault(str(unprocessed["job_id"]), child.global_error)
                    break
        else:
            for job_id in remaining:
                outcome.failures[job_id] = last_error
        return outcome

    return classify_subset(expected)


def build_current_payload(
    job: dict,
    classification: FreehireClassificationItem,
    client=None,
    batch_id: str | None = None,
    attempts: int = 1,
    result_model: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    is_remote, evidence = classify_remote(job)
    seniority = source_seniority(job.get("level")) or classification.seniority
    category = classification.category
    model = result_model or model_name(client)
    payload = {
        "freehire_category": category,
        "freehire_seniority": seniority,
        "is_remote": is_remote,
        "freehire_remote_evidence": evidence,
        "freehire_compat_status": "current",
        "freehire_compat_input_hash": compute_classification_hash(job),
        "freehire_compat_model": model,
        "freehire_compat_prompt_version": config.FREEHIRE_COMPAT_PROMPT_VERSION,
        "freehire_compat_schema_version": config.FREEHIRE_COMPAT_SCHEMA_VERSION,
        "freehire_compat_confidence": classification.confidence,
        "freehire_compat_classified_at": now,
        "freehire_compat_error": None,
        "freehire_compat_attempts": attempts,
        "freehire_compat_claimed_at": None,
        "freehire_compat_claimed_by": None,
        "freehire_compat_next_retry_at": None,
        "freehire_compat_provenance": {
            "batch_id": batch_id or str(uuid.uuid4()),
            "classified_at": now,
            "model": model,
            "prompt_version": config.FREEHIRE_COMPAT_PROMPT_VERSION,
            "schema_version": config.FREEHIRE_COMPAT_SCHEMA_VERSION,
        },
    }
    payload["freehire_compat_import_hash"] = compute_import_hash(
        {**job, **payload}, category=category, seniority=seniority, is_remote=is_remote
    )
    return payload


def build_failure_payload(job: dict, error: str, client=None, attempts: int = 1) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    model = model_name(client)
    cooldown = min(
        config.FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MINUTES * (2 ** max(0, attempts - 1)),
        config.FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MAX_MINUTES,
    )
    return {
        "freehire_compat_status": "failed",
        "freehire_compat_error": error[:4000],
        "freehire_compat_attempts": attempts,
        "freehire_compat_model": model,
        "freehire_compat_prompt_version": config.FREEHIRE_COMPAT_PROMPT_VERSION,
        "freehire_compat_schema_version": config.FREEHIRE_COMPAT_SCHEMA_VERSION,
        "freehire_compat_claimed_at": None,
        "freehire_compat_claimed_by": None,
        "freehire_compat_next_retry_at": (
            datetime.now(timezone.utc) + timedelta(minutes=cooldown)
        ).isoformat(),
        "freehire_compat_provenance": {
            "failed_at": now,
            "error": error[:4000],
            "model": model,
            "prompt_version": config.FREEHIRE_COMPAT_PROMPT_VERSION,
            "schema_version": config.FREEHIRE_COMPAT_SCHEMA_VERSION,
        },
    }
