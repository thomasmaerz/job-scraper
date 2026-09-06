from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Sequence
from urllib.parse import urlencode, urlparse

import requests
from bs4 import BeautifulSoup

import config
import supabase_utils
import user_agents
from linkedin_source_policy import (
    DurableLinkedInRequestGate,
    LinkedInCircuitOpen,
    LinkedInGrantRejected,
    LinkedInRequestDeadlineExceeded,
)


CLASSIFIER_VERSION = "linkedin-guest-search-v3"
SCOPE_VERSION = "linkedin-scope-v1"
EXPECTED_PAGE_SIZE = 10
NO_RESULTS_SELECTORS = (
    ".jobs-search-no-results-banner",
    ".jobs-search-no-results-banner__image",
    "section.no-results",
)
NO_RESULTS_RESPONSE_BODIES = (
    b"<!DOCTYPE html>\n\n<!---->  ",
)


class DiscoveryError(RuntimeError):
    pass


class RetryableDiscoveryInterruption(DiscoveryError):
    pass


@dataclass(frozen=True)
class DiscoveryResult:
    cycle_id: int
    discovery_sequence: int
    canonical_job_ids: tuple[str, ...]
    advance_watermark: bool


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def scope_definition(execution: Any, job_type: str, work_types: str) -> dict[str, Any]:
    return {
        "version": SCOPE_VERSION,
        "provider": "linkedin",
        "endpoint": "jobs-guest-search-v1",
        "lane": execution.lane.archetype,
        "query": _normalized_text(execution.query.query),
        "query_type": execution.query.query_type.value,
        "language": execution.query.language,
        "location": execution.geography.location,
        "location_scope": execution.geography.location_scope.value,
        "geography_id": execution.geography.geography_id,
        "geo_id": execution.geography.geo_id,
        "geography_mapping_version": "static-v1",
        "job_type": job_type,
        "work_types": sorted(set(filter(None, re.split(r"[,;]", work_types or "")))),
        "partition_filters": {},
    }


def scope_key(definition: dict[str, Any]) -> str:
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "linkedin:v1:" + hashlib.sha256(encoded.encode()).hexdigest()


def configuration_hash(configuration: Any) -> str:
    encoded = json.dumps(
        configuration.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def adaptive_options(settings: Any) -> dict[str, int]:
    options = settings.options

    def integer_option(name: str, default: int) -> int:
        value = options.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise DiscoveryError(f"options.{name} must be an integer")
        return value

    minimum = integer_option("min_pages_per_query", settings.max_pages_per_query)
    soft = integer_option("soft_max_pages_per_query", max(minimum, min(10, minimum + 4)))
    hard = integer_option("hard_max_pages_per_query", max(soft, 20))
    extra = integer_option("max_adaptive_extra_requests", 20)
    detail = integer_option("max_detail_tasks_per_run", settings.max_jobs_per_query * 6)
    attempts = integer_option("max_source_http_attempts_per_run", 800)
    detail_attempts = integer_option("max_detail_http_attempts_per_run", 800)
    minimum_window = integer_option("minimum_recent_window_hours", 3)
    overlap = integer_option("indexing_overlap_hours", 6)
    maximum_window = integer_option("maximum_normal_window_hours", 24)
    recovery_cap = integer_option("outage_recovery_cap_hours", 168)
    search_runtime = integer_option("max_search_runtime_seconds", 1_620)
    detail_runtime = integer_option("max_detail_runtime_seconds", 300)
    if not 1 <= minimum <= soft <= hard <= 100:
        raise DiscoveryError("adaptive page limits must satisfy 1 <= minimum <= soft <= hard <= 100")
    if (extra < 0 or detail < 0 or detail > 10_000 or attempts < minimum
            or detail_attempts < 0 or (detail > 0 and detail_attempts == 0)):
        raise DiscoveryError("adaptive request budgets are invalid")
    if not 1 <= minimum_window <= maximum_window <= recovery_cap <= 8_760 or overlap < 0:
        raise DiscoveryError("adaptive lookback options are invalid")
    if (not 60 <= search_runtime <= 1_920 or not 0 <= detail_runtime <= 1_200
            or search_runtime + detail_runtime > 1_920):
        raise DiscoveryError("adaptive runtime budgets are invalid")
    return {
        "minimum": minimum,
        "soft": soft,
        "hard": hard,
        "extra": extra,
        "detail": detail,
        "attempts": attempts,
        "detail_attempts": detail_attempts,
        "minimum_window": minimum_window,
        "overlap": overlap,
        "maximum_window": maximum_window,
        "recovery_cap": recovery_cap,
        "search_runtime": search_runtime,
        "detail_runtime": detail_runtime,
    }


def _is_challenge(response: requests.Response) -> bool:
    if response.status_code in (403, 999):
        return True
    parsed = urlparse(str(response.url or ""))
    text = (response.text or "")[:2000].lower()
    return (
        "/checkpoint/" in parsed.path
        or "/challenge/" in parsed.path
        or "security verification" in text
        or "id=\"challenge-page\"" in text
        or "id='challenge-page'" in text
    )


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
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


def classify_search_response(response: requests.Response) -> tuple[str, BeautifulSoup, list[Any]]:
    if _is_challenge(response):
        return "challenge", BeautifulSoup("", "html.parser"), []
    if response.status_code != 200:
        return "http_error", BeautifulSoup("", "html.parser"), []
    parsed_url = urlparse(str(response.url or ""))
    if parsed_url.hostname not in {"www.linkedin.com", "linkedin.com"}:
        raise DiscoveryError("LinkedIn search redirected to an unexpected host")
    content_type = (response.headers.get("Content-Type") or "text/html").lower()
    if "html" not in content_type or not response.text:
        raise DiscoveryError("LinkedIn search returned an empty or non-HTML response")
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.find_all("li")
    if elements:
        return "cards", soup, elements
    if response.content in NO_RESULTS_RESPONSE_BODIES:
        return "no_results", soup, []
    if any(soup.select_one(selector) is not None for selector in NO_RESULTS_SELECTORS):
        return "no_results", soup, []
    raise DiscoveryError("LinkedIn returned unrecognized zero-card HTML")


def _position_cards(cards: Sequence[dict], page_number: int, page_start: int) -> list[dict]:
    positioned: list[dict] = []
    seen: set[str] = set()
    for position, card in enumerate(cards):
        source_id = str(card.get("job_id") or "").strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        positioned.append({
            **card,
            "job_id": source_id,
            "page_number": page_number,
            "page_start": page_start,
            "position_on_page": position,
            "position_in_scope": page_start + position,
        })
    return positioned


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _request_page(
    scope: dict[str, Any],
    page_number: int,
    *,
    user_agent: str,
    gate: DurableLinkedInRequestGate,
    parse_cards: Callable[[Sequence[Any]], list[dict]],
    physical_attempts: list[int],
    physical_limit: int,
    maximum_lookback_seconds: int | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    page_start = (page_number - 1) * EXPECTED_PAGE_SIZE
    anchor = datetime.fromisoformat(scope["source_window_earliest_at"].replace("Z", "+00:00"))
    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        if physical_attempts[0] >= physical_limit:
            raise RetryableDiscoveryInterruption(
                "LinkedIn physical HTTP-attempt budget exhausted"
            )
        grant = gate.acquire(
            "search",
            f"{scope['scope_key']}:{page_number}:{attempt}",
            deadline=deadline,
        )
        lookback_seconds = max(1, math.ceil((grant.started_at - anchor).total_seconds()))
        if (maximum_lookback_seconds is not None
                and lookback_seconds > maximum_lookback_seconds):
            gate.finish(grant, "window_expired", None)
            raise DiscoveryError(
                "persisted discovery window exceeds the supported recovery cap"
            )
        effective_earliest = grant.started_at - timedelta(seconds=lookback_seconds)
        params = {
            "keywords": scope["query"],
            "location": scope["location"],
            "f_TPR": f"r{lookback_seconds}",
            "f_JT": scope["job_type"],
            "f_WT": scope["work_types"],
            "start": page_start,
        }
        if scope.get("geo_id") is not None:
            params["geoId"] = scope["geo_id"]
        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?" + urlencode(params)
        physical_attempts[0] += 1
        started = time.monotonic()
        response = None
        try:
            request_timeout = config.REQUEST_TIMEOUT
            if deadline is not None:
                request_timeout = min(
                    request_timeout, max(0.1, deadline - time.monotonic())
                )
            response = requests.get(
                url, headers={"User-Agent": user_agent}, timeout=request_timeout
            )
            kind, _soup, elements = classify_search_response(response)
            if kind == "challenge":
                gate.open_circuit(grant, "LinkedIn denied or challenged search access", response.status_code)
                raise LinkedInCircuitOpen("LinkedIn denied or challenged search access")
            if kind == "http_error":
                gate.finish(grant, "http_error", response.status_code)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = DiscoveryError(f"LinkedIn search HTTP {response.status_code}")
                    if attempt < config.MAX_RETRIES:
                        retry_delay = max(
                            _retry_after_seconds(response) or 0,
                            config.RETRY_DELAY_SECONDS,
                        )
                        if deadline is not None and time.monotonic() + retry_delay >= deadline:
                            raise RetryableDiscoveryInterruption(str(last_error))
                        time.sleep(retry_delay)
                        continue
                error = f"LinkedIn search HTTP {response.status_code}"
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    raise RetryableDiscoveryInterruption(error)
                raise DiscoveryError(error)
            cards = [] if kind == "no_results" else _position_cards(
                parse_cards(elements), page_number, page_start
            )
            if kind == "cards" and not cards:
                raise DiscoveryError("LinkedIn search parser extracted zero valid cards")
            gate.finish(grant, kind, response.status_code)
            return {
                "kind": kind,
                "cards": cards,
                "elements": len(elements),
                "page_number": page_number,
                "page_start": page_start,
                "requested_at": grant.started_at.isoformat(),
                "source_window_earliest_at": effective_earliest.isoformat(),
                "source_window_latest_at": grant.started_at.isoformat(),
                "lookback_seconds": lookback_seconds,
                "request_attempts": attempt + 1,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "classifier_version": CLASSIFIER_VERSION,
                "response_fingerprint": hashlib.sha256(response.content).hexdigest(),
                "membership_fingerprint": _fingerprint([
                    (card["job_id"], card["position_on_page"]) for card in cards
                ]),
            }
        except LinkedInCircuitOpen:
            raise
        except LinkedInGrantRejected:
            raise
        except LinkedInRequestDeadlineExceeded as exc:
            raise RetryableDiscoveryInterruption(str(exc)) from exc
        except RetryableDiscoveryInterruption:
            raise
        except requests.RequestException as exc:
            gate.finish(grant, "transport_error", None)
            last_error = exc
            if attempt < config.MAX_RETRIES:
                if deadline is not None and time.monotonic() + config.RETRY_DELAY_SECONDS >= deadline:
                    raise RetryableDiscoveryInterruption(str(exc)) from exc
                time.sleep(config.RETRY_DELAY_SECONDS)
                continue
            raise RetryableDiscoveryInterruption(
                f"LinkedIn search transport failed: {exc}"
            ) from exc
        except Exception:
            gate.finish(grant, "parser_error", getattr(response, "status_code", None))
            raise
    raise DiscoveryError(str(last_error or "LinkedIn page request failed"))


def _scope_manifest(configuration: Any, executions: Sequence[Any], options: dict[str, int]) -> list[dict]:
    now = datetime.now(timezone.utc)
    definitions = [
        scope_definition(execution, config.LINKEDIN_JOB_TYPE, config.LINKEDIN_F_WT)
        for execution in executions
    ]
    keys = [scope_key(definition) for definition in definitions]
    recovery_floor = now - timedelta(hours=options["recovery_cap"])
    prepared = supabase_utils.prepare_linkedin_discovery_scope_state(
        keys, recovery_floor.isoformat()
    )
    prior_states = prepared["states"]
    pending_debt = prepared["debt"]
    configured_hours = min(
        options["recovery_cap"],
        max(configuration.settings.lookback_days * 24, config.LINKEDIN_LOOKBACK_HOURS),
    )
    manual_recovery = os.getenv("LINKEDIN_RECOVERY_LOOKBACK_HOURS")
    manual_recovery_hours = None
    if manual_recovery:
        try:
            manual_recovery_hours = min(
                options["recovery_cap"], max(1, int(manual_recovery))
            )
            configured_hours = manual_recovery_hours
        except ValueError as exc:
            raise DiscoveryError("LINKEDIN_RECOVERY_LOOKBACK_HOURS must be an integer") from exc
    manifests = []
    for execution, definition, key in zip(executions, definitions, keys):
        prior = prior_states.get(key) or {}
        debt = pending_debt.get(key)
        last_success_value = prior.get("last_operational_success_at")
        truncated_earliest = None
        truncated_latest = None
        expired_earliest = None
        expired_latest = None
        if last_success_value:
            try:
                last_success = datetime.fromisoformat(
                    str(last_success_value).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise DiscoveryError(f"scope {key} has an invalid success watermark") from exc
            elapsed_hours = max(0, math.ceil((now - last_success).total_seconds() / 3600))
            desired_hours = max(
                options["minimum_window"], elapsed_hours + options["overlap"]
            )
            if manual_recovery_hours is not None:
                desired_hours = max(desired_hours, manual_recovery_hours)
            if desired_hours > options["recovery_cap"]:
                expired_earliest = (now - timedelta(hours=desired_hours)).isoformat()
                expired_latest = recovery_floor.isoformat()
            window_hours = min(options["recovery_cap"], desired_hours)
            desired_earliest = now - timedelta(hours=desired_hours)
            selected_earliest = now - timedelta(hours=window_hours)
        else:
            window_hours = configured_hours
        earliest = now - timedelta(hours=window_hours)
        if debt:
            try:
                debt_earliest = datetime.fromisoformat(
                    str(debt["source_window_earliest_at"]).replace("Z", "+00:00")
                )
            except (KeyError, ValueError) as exc:
                raise DiscoveryError(f"scope {key} has invalid coverage debt") from exc
            earliest = max(recovery_floor, min(earliest, debt_earliest))
        recommended_pages = prior.get("recommended_pages")
        target_pages = options["hard"] if debt else options["soft"]
        if not debt and isinstance(recommended_pages, int) and not isinstance(recommended_pages, bool):
            target_pages = min(target_pages, max(options["minimum"], recommended_pages))
        manifests.append({
            "scope_key": key,
            "scope_definition_hash": key.rsplit(":", 1)[-1],
            "scope_definition": definition,
            "archetype": execution.lane.archetype,
            "query_id": execution.query.query_id,
            "query": execution.query.query,
            "query_type": execution.query.query_type.value,
            "language": execution.query.language,
            "location": execution.geography.location,
            "location_scope": execution.geography.location_scope.value,
            "geography_id": execution.geography.geography_id,
            "geo_id": execution.geography.geo_id,
            "job_type": config.LINKEDIN_JOB_TYPE,
            "work_types": config.LINKEDIN_F_WT,
            "query_scope": json.dumps({
                "scope_key": key,
                "archetype": execution.lane.archetype,
                "query_id": execution.query.query_id,
                "geography_id": execution.geography.geography_id,
            }, sort_keys=True, separators=(",", ":")),
            "request_anchor_at": now.isoformat(),
            "source_window_earliest_at": earliest.isoformat(),
            "source_window_latest_at": now.isoformat(),
            "truncated_window_earliest_at": truncated_earliest,
            "truncated_window_latest_at": truncated_latest,
            "expired_window_earliest_at": expired_earliest,
            "expired_window_latest_at": expired_latest,
            "minimum_pages": options["minimum"],
            "target_pages": target_pages,
            "coverage_debt_created_at": debt.get("created_at") if debt else None,
            "last_deep_sweep_at": prior.get("last_deep_sweep_at"),
        })
    return manifests


def _resumable_scope_manifest(scope: dict[str, Any]) -> dict[str, Any]:
    definition = scope.get("scope_definition")
    if not isinstance(definition, dict):
        raise DiscoveryError("resumable discovery scope omitted its definition")
    work_types = definition.get("work_types")
    if not isinstance(work_types, list):
        raise DiscoveryError("resumable discovery scope has invalid work types")
    return {
        **scope,
        "scope_definition": definition,
        "archetype": definition["lane"],
        "query_id": json.loads(scope["query_scope"])["query_id"],
        "query": definition["query"],
        "query_type": definition["query_type"],
        "language": definition["language"],
        "location": definition["location"],
        "location_scope": definition["location_scope"],
        "geography_id": definition["geography_id"],
        "geo_id": definition.get("geo_id"),
        "job_type": definition["job_type"],
        "work_types": ",".join(str(value) for value in work_types),
        "next_page": int(scope.get("next_page") or 1),
        "status": "exhausted" if scope.get("latest_page_result") == "no_results" else scope.get("status", "running"),
        "tail_yield": 0,
    }


def _drain_tasks(
    cycle_id: int,
    limit: int,
    *,
    user_agent: str,
    detail_fetch: Callable[..., Any],
    save_details: Callable[[dict, str, dict], str],
    physical_limit: int | None = None,
    deadline: float | None = None,
) -> list[str]:
    if limit <= 0 or (physical_limit is not None and physical_limit <= 0):
        return []
    worker_id = f"{os.getenv('GITHUB_RUN_ID') or 'local'}:{uuid.uuid4()}"
    saved: list[str] = []
    gate = DurableLinkedInRequestGate("adaptive-detail")
    remaining = limit
    newest_remaining = min(limit, 1 + math.ceil(max(0, limit - 1) * 0.2))
    physical_attempts = [0]
    while remaining > 0:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if physical_limit is not None and physical_attempts[0] >= physical_limit:
            break
        order_mode = "newest" if newest_remaining > 0 else "oldest"
        claim_limit = 1
        tasks = supabase_utils.claim_linkedin_discovery_tasks(
            worker_id, limit=claim_limit, order_mode=order_mode
        )
        if not tasks:
            if order_mode == "newest":
                newest_remaining = 0
                continue
            break
        for task in tasks:
            try:
                result = detail_fetch(
                    str(task["source_job_id"]),
                    search_card=task.get("search_card") or {},
                    durable_gate=gate,
                    user_agent=user_agent,
                    request_key=f"task:{task['id']}:{task['lease_token']}:{task['source_job_id']}",
                    physical_attempts=physical_attempts,
                    physical_limit=physical_limit,
                    required_fields=("company", "job_title"),
                    deadline=deadline,
                )
                if result is None:
                    supabase_utils.transition_linkedin_discovery_task(
                        task["id"], worker_id, task["lease_token"], "failed_retryable",
                        error_code="detail_invalid",
                    )
                    continue
                if getattr(result, "confirmed_terminal_unavailable", False):
                    status_code = int(getattr(result, "status_code", 0) or 0)
                    confirmations = int(getattr(result, "confirmations", 0) or 0)
                    supabase_utils.transition_linkedin_discovery_task(
                        task["id"], worker_id, task["lease_token"], "terminal_unavailable",
                        error_code=(
                            "source_unavailable_404_confirmed"
                            if status_code == 404 and confirmations >= 2
                            else "source_unavailable_410"
                            if status_code == 410 and confirmations >= 1
                            else "source_unavailable_unconfirmed"
                        ),
                    )
                    continue
                if not result:
                    supabase_utils.transition_linkedin_discovery_task(
                        task["id"], worker_id, task["lease_token"], "failed_retryable",
                        error_code="detail_empty",
                    )
                    continue
                supabase_utils.heartbeat_linkedin_discovery_task(
                    task["id"], worker_id, task["lease_token"]
                )
                details, metadata = result
                job = {**details, **metadata, **(task.get("provenance") or {})}
                job["scrape_run_id"] = task.get("first_ingestion_run_id")
                canonical_id = save_details(task, worker_id, job)
                if not isinstance(canonical_id, str) or not canonical_id:
                    raise DiscoveryError("canonical persistence did not return exactly one job ID")
                saved.append(canonical_id)
            except LinkedInCircuitOpen:
                raise
            except LinkedInGrantRejected:
                raise
            except (
                supabase_utils.CanonicalTaskApplyAmbiguous,
                supabase_utils.CanonicalTaskReceiptConflict,
            ):
                raise
            except supabase_utils.CanonicalTaskLeaseLost:
                logging.warning("Lost adaptive discovery task lease for task %s", task["id"])
                continue
            except Exception as exc:
                try:
                    supabase_utils.transition_linkedin_discovery_task(
                        task["id"], worker_id, task["lease_token"], "failed_retryable",
                        error_code=type(exc).__name__[:100],
                    )
                except supabase_utils.CanonicalTaskLeaseLost:
                    logging.warning(
                        "Lost adaptive discovery task lease while recording failure for task %s",
                        task["id"],
                    )
                    continue
                except Exception as cleanup_error:
                    raise exc from cleanup_error
                if physical_limit is not None and physical_attempts[0] >= physical_limit:
                    return saved
        remaining -= len(tasks)
        if order_mode == "newest":
            newest_remaining = max(0, newest_remaining - len(tasks))
        if len(tasks) < claim_limit and order_mode == "oldest":
            break
    return saved


def run_discovery(
    configuration: Any,
    executions: Sequence[Any],
    *,
    parse_cards: Callable[[Sequence[Any]], list[dict]],
    detail_fetch: Callable[..., Any],
    save_details: Callable[[dict, str, dict], str],
    partial: bool,
) -> DiscoveryResult:
    run_started = time.monotonic()
    options = adaptive_options(configuration.settings)
    user_agent = os.getenv("LINKEDIN_USER_AGENT") or user_agents.USER_AGENTS[0]
    requested_scope_keys = [
        scope_key(scope_definition(execution, config.LINKEDIN_JOB_TYPE, config.LINKEDIN_F_WT))
        for execution in executions
    ]
    resumable = supabase_utils.get_resumable_linkedin_discovery_cycle(
        partial=partial,
        scope_keys=requested_scope_keys if partial else None,
    )
    if resumable:
        cycle = resumable
        user_agent = str(cycle["pinned_user_agent"])
        manifests = [
            _resumable_scope_manifest(scope) for scope in cycle["scopes"]
        ]
    else:
        manifests = _scope_manifest(configuration, executions, options)
        if options["attempts"] < len(manifests) * options["minimum"]:
            raise DiscoveryError(
                "max_source_http_attempts_per_run cannot fund every required minimum page"
            )
        for scope in manifests:
            query_scope = json.loads(scope["query_scope"])
            query_scope["partial"] = partial
            scope["query_scope"] = json.dumps(
                query_scope, sort_keys=True, separators=(",", ":")
            )
    execution_key = (
        os.getenv("LINKEDIN_DISCOVERY_EXECUTION_ID")
        or os.getenv("GITHUB_RUN_ID")
        or str(uuid.uuid4())
    )
    try:
        execution_id = str(uuid.UUID(execution_key))
    except ValueError:
        execution_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"linkedin-discovery:{execution_key}"))
    if not resumable:
        config_hash = configuration_hash(configuration)
        for _ in range(10):
            cycle = supabase_utils.create_linkedin_discovery_cycle(
                execution_id=execution_id,
                configuration_revision=configuration.revision,
                configuration_hash=config_hash,
                user_agent=user_agent,
                scopes=manifests,
            )
            if cycle.get("search_status", "running") != "failed":
                break
            execution_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"linkedin-discovery-recovery:{execution_id}:{cycle['cycle_id']}",
            ))
        else:
            raise DiscoveryError("discovery recovery chain exceeded 10 failed executions")
    cycle_id = cycle["cycle_id"]
    sequence = int(cycle["discovery_sequence"])
    cycle_status = cycle.get("search_status", "running")
    returned_scopes = {row["scope_key"]: row for row in cycle["scopes"]}
    for scope in manifests:
        persisted = returned_scopes[scope["scope_key"]]
        scope.update(persisted)
        scope["next_page"] = int(scope.get("next_page") or 1)
        scope["status"] = returned_scopes[scope["scope_key"]].get("status", "running")
        if scope.get("latest_page_result") == "no_results":
            scope["status"] = "exhausted"
        scope["tail_yield"] = 0
    gate = DurableLinkedInRequestGate("adaptive-discovery")
    physical_attempts = [0]

    def fetch_commit(scope: dict[str, Any]) -> dict[str, Any]:
        page = _request_page(
            scope,
            scope["next_page"],
            user_agent=user_agent,
            gate=gate,
            parse_cards=parse_cards,
            physical_attempts=physical_attempts,
            physical_limit=options["attempts"],
            maximum_lookback_seconds=options["recovery_cap"] * 3600,
            deadline=run_started + options["search_runtime"],
        )
        receipt = supabase_utils.commit_linkedin_discovery_page({
            **page,
            "cycle_id": cycle_id,
            "scope_key": scope["scope_key"],
            "ingestion_run_id": scope["ingestion_run_id"],
            "query_scope": scope["query_scope"],
            "provenance": {
                "archetype": scope["archetype"],
                "filter_profile": f"{scope['archetype']}_v1",
                "lane": scope["archetype"],
                "search_query": scope["query"],
                "search_query_id": scope["query_id"],
                "search_query_type": scope["query_type"],
                "search_query_language": scope["language"],
                "search_location_scope": scope["location_scope"],
                "geography_id": scope["geography_id"],
                "query_scope": scope["query_scope"],
            },
        })
        scope["next_page"] += 1
        scope["tail_yield"] = int(receipt["new_workflow_source_ids"])
        if page["kind"] == "no_results":
            scope["status"] = "exhausted"
        return receipt

    try:
        if cycle_status == "sealed":
            saved = _drain_tasks(
                cycle_id,
                options["detail"],
                user_agent=user_agent,
                detail_fetch=detail_fetch,
                save_details=save_details,
                physical_limit=options["detail_attempts"],
                deadline=(
                    run_started
                    + options["search_runtime"]
                    + options["detail_runtime"]
                ),
            )
            if not partial:
                supabase_utils.resolve_eligible_failed_linkedin_discovery_cycles(cycle_id)
            return DiscoveryResult(cycle_id, sequence, tuple(saved), not partial)
        for scope in manifests:
            if scope["status"] == "exhausted":
                supabase_utils.finish_linkedin_discovery_scope(
                    scope["ingestion_run_id"],
                    "exhausted",
                    coverage_reason="positive no-results response",
                )
                scope["status"] = "complete"
        search_deadline = run_started + options["search_runtime"]
        search_interrupted = False
        while (
            physical_attempts[0] < options["attempts"]
            and time.monotonic() < search_deadline
        ):
            eligible = [scope for scope in manifests if scope["status"] == "running"]
            if not eligible:
                break
            eligible.sort(key=lambda row: (row["next_page"], row["scope_key"]))
            made_progress = False
            for scope in eligible:
                if (
                    physical_attempts[0] >= options["attempts"]
                    or time.monotonic() >= search_deadline
                ):
                    break
                try:
                    fetch_commit(scope)
                except (RetryableDiscoveryInterruption, LinkedInCircuitOpen) as exc:
                    logging.warning(
                        "Discovery cycle %s remains resumable after source interruption: %s",
                        cycle_id,
                        exc,
                    )
                    search_interrupted = True
                    break
                made_progress = True
                if scope["status"] == "exhausted":
                    supabase_utils.finish_linkedin_discovery_scope(
                        scope["ingestion_run_id"],
                        "exhausted",
                        coverage_reason="positive no-results response",
                    )
                    scope["status"] = "complete"
            if search_interrupted or not made_progress:
                break
        coverage_complete = all(scope["status"] == "complete" for scope in manifests)
        if coverage_complete:
            sealed = supabase_utils.seal_linkedin_discovery_cycle(
                cycle_id, advance_watermark=not partial
            )
        else:
            sealed = {"watermark_advanced": False}
            logging.info(
                "Discovery cycle %s remains resumable: completed_scopes=%s/%s "
                "pages_this_run=%s",
                cycle_id,
                sum(scope["status"] == "complete" for scope in manifests),
                len(manifests),
                physical_attempts[0],
            )
        if coverage_complete and not partial:
            supabase_utils.resolve_eligible_failed_linkedin_discovery_cycles(cycle_id)
    except Exception as exc:
        supabase_utils.fail_linkedin_discovery_cycle(cycle_id, str(exc))
        raise

    saved = _drain_tasks(
        cycle_id,
        options["detail"],
        user_agent=user_agent,
        detail_fetch=detail_fetch,
        save_details=save_details,
        physical_limit=options["detail_attempts"],
        deadline=(
            run_started + options["search_runtime"] + options["detail_runtime"]
        ),
    )
    if coverage_complete and not partial:
        supabase_utils.resolve_eligible_failed_linkedin_discovery_cycles(cycle_id)
    return DiscoveryResult(
        cycle_id,
        sequence,
        tuple(saved),
        bool(sealed.get("watermark_advanced")) if coverage_complete else False,
    )
