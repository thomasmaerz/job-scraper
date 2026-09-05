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
)


CLASSIFIER_VERSION = "linkedin-guest-search-v2"
SCOPE_VERSION = "linkedin-scope-v1"
EXPECTED_PAGE_SIZE = 10
NO_RESULTS_SELECTORS = (
    ".jobs-search-no-results-banner",
    ".jobs-search-no-results-banner__image",
    "section.no-results",
)


class DiscoveryError(RuntimeError):
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
    if not 1 <= minimum <= soft <= hard <= 100:
        raise DiscoveryError("adaptive page limits must satisfy 1 <= minimum <= soft <= hard <= 100")
    if (extra < 0 or detail < 0 or detail > 10_000 or attempts < minimum
            or detail_attempts < 0 or (detail > 0 and detail_attempts == 0)):
        raise DiscoveryError("adaptive request budgets are invalid")
    if not 1 <= minimum_window <= maximum_window <= recovery_cap <= 8_760 or overlap < 0:
        raise DiscoveryError("adaptive lookback options are invalid")
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
) -> dict[str, Any]:
    page_start = (page_number - 1) * EXPECTED_PAGE_SIZE
    anchor = datetime.fromisoformat(scope["source_window_earliest_at"].replace("Z", "+00:00"))
    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        if physical_attempts[0] >= physical_limit:
            raise DiscoveryError("LinkedIn physical HTTP-attempt budget exhausted")
        grant = gate.acquire("search", f"{scope['scope_key']}:{page_number}:{attempt}")
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
            response = requests.get(url, headers={"User-Agent": user_agent}, timeout=config.REQUEST_TIMEOUT)
            kind, _soup, elements = classify_search_response(response)
            if kind == "challenge":
                gate.open_circuit(grant, "LinkedIn denied or challenged search access", response.status_code)
                raise LinkedInCircuitOpen("LinkedIn denied or challenged search access")
            if kind == "http_error":
                gate.finish(grant, "http_error", response.status_code)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = DiscoveryError(f"LinkedIn search HTTP {response.status_code}")
                    if attempt < config.MAX_RETRIES:
                        time.sleep(max(
                            _retry_after_seconds(response) or 0,
                            config.RETRY_DELAY_SECONDS,
                        ))
                        continue
                raise DiscoveryError(f"LinkedIn search HTTP {response.status_code}")
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
        except requests.RequestException as exc:
            gate.finish(grant, "transport_error", None)
            last_error = exc
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_DELAY_SECONDS)
                continue
            raise DiscoveryError(f"LinkedIn search transport failed: {exc}") from exc
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
    for key in keys:
        supabase_utils.expire_linkedin_coverage_debt(key, recovery_floor.isoformat())
    prior_states = supabase_utils.get_linkedin_scope_coverage_states(keys)
    pending_debt = supabase_utils.get_pending_linkedin_coverage_debt(keys)
    configured_hours = min(
        options["recovery_cap"],
        max(configuration.settings.lookback_days * 24, config.LINKEDIN_LOOKBACK_HOURS),
    )
    manual_recovery = os.getenv("LINKEDIN_RECOVERY_LOOKBACK_HOURS")
    if manual_recovery:
        try:
            configured_hours = min(options["recovery_cap"], max(1, int(manual_recovery)))
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
            window_hours = min(options["maximum_window"], desired_hours)
            desired_earliest = now - timedelta(hours=desired_hours)
            selected_earliest = now - timedelta(hours=window_hours)
            if desired_earliest < selected_earliest:
                truncated_earliest = max(desired_earliest, recovery_floor).isoformat()
                truncated_latest = selected_earliest.isoformat()
                if desired_earliest < recovery_floor:
                    expired_earliest = desired_earliest.isoformat()
                    expired_latest = recovery_floor.isoformat()
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


def _drain_tasks(
    cycle_id: int,
    limit: int,
    *,
    user_agent: str,
    detail_fetch: Callable[..., Any],
    save_details: Callable[[dict, str, dict], str],
    physical_limit: int | None = None,
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
    options = adaptive_options(configuration.settings)
    user_agent = os.getenv("LINKEDIN_USER_AGENT") or user_agents.USER_AGENTS[0]
    manifests = _scope_manifest(configuration, executions, options)
    if options["attempts"] < len(manifests) * options["minimum"]:
        raise DiscoveryError(
            "max_source_http_attempts_per_run cannot fund every required minimum page"
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
            )
            if not partial:
                supabase_utils.resolve_eligible_failed_linkedin_discovery_cycles(cycle_id)
            return DiscoveryResult(cycle_id, sequence, tuple(saved), not partial)
        while True:
            minimum_round = [
                scope for scope in manifests
                if scope["status"] == "running"
                and scope["next_page"] <= scope["minimum_pages"]
            ]
            if not minimum_round:
                break
            for scope in minimum_round:
                fetch_commit(scope)
        committed_extensions = sum(
            max(0, int(scope.get("committed_page_count") or 0) - scope["minimum_pages"])
            for scope in manifests
        )
        extras = max(0, options["extra"] - committed_extensions)
        while extras > 0:
            eligible = [
                scope for scope in manifests
                if scope["status"] == "running"
                and scope["next_page"] <= scope["target_pages"]
            ]
            if not eligible:
                break
            eligible.sort(key=lambda row: (
                row["coverage_debt_created_at"] is None,
                row["last_deep_sweep_at"] is not None,
                row["last_deep_sweep_at"] or "",
                row["coverage_debt_created_at"] or "",
                -row["tail_yield"],
                row["scope_key"],
            ))
            recovery_pages_needed = (
                eligible[0]["target_pages"] - eligible[0]["next_page"] + 1
            )
            if (eligible[0]["coverage_debt_created_at"] is not None
                    and extras >= recovery_pages_needed):
                scope = eligible[0]
                while (
                    extras > 0
                    and scope["status"] == "running"
                    and scope["next_page"] <= scope["target_pages"]
                ):
                    fetch_commit(scope)
                    extras -= 1
                continue
            for scope in eligible:
                if extras <= 0:
                    break
                fetch_commit(scope)
                extras -= 1
        for scope in manifests:
            if scope["status"] == "complete":
                continue
            coverage = scope["status"] if scope["status"] == "exhausted" else "right_censored"
            supabase_utils.finish_linkedin_discovery_scope(
                scope["ingestion_run_id"],
                coverage,
                coverage_reason=(
                    "positive no-results response"
                    if coverage == "exhausted"
                    else "adaptive target or global extension budget reached"
                ),
            )
        supabase_utils.seal_linkedin_discovery_cycle(
            cycle_id, advance_watermark=not partial
        )
        if not partial:
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
    )
    if not partial:
        supabase_utils.resolve_eligible_failed_linkedin_discovery_cycles(cycle_id)
    return DiscoveryResult(cycle_id, sequence, tuple(saved), not partial)
