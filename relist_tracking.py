"""Pure folds for replayable LinkedIn card observations."""

import hashlib
import re
import unicodedata
from datetime import date, datetime
from typing import Any


ALGORITHM_VERSION = "same-id-relist-v1"
CONTENT_HASH_VERSION = "description-sha256-v1"


def date_part(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", str(value).strip())
    return match.group(1) if match else None


def make_content_hash(description: str | None) -> str | None:
    if description is None:
        return None
    normalized = unicodedata.normalize("NFKC", description).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fold_observations(
    observations: list[dict],
    min_forward_days: int = 2,
    stable_observations: int = 2,
) -> dict:
    ordered = sorted(
        (dict(item) for item in observations if date_part(item.get("posted_at") or item.get("observed_posted_at"))),
        key=lambda item: (
            str(item.get("observed_at") or item.get("first_seen_at") or ""),
            str(item.get("ingestion_run_id") or item.get("scrape_run_id") or ""),
            date_part(item.get("posted_at") or item.get("observed_posted_at")) or "",
        ),
    )
    if not ordered:
        return {"latest_trusted_posted_date": None, "events": [], "corrections": []}

    trusted = date_part(ordered[0].get("posted_at") or ordered[0].get("observed_posted_at"))
    stable_count = 1
    events = []
    corrections = []
    for item in ordered[1:]:
        observed = date_part(item.get("posted_at") or item.get("observed_posted_at"))
        if observed == trusted:
            stable_count += 1
            continue
        delta = (date.fromisoformat(observed) - date.fromisoformat(trusted)).days
        if delta >= min_forward_days and stable_count >= stable_observations:
            events.append({
                "previous_posted_date": trusted,
                "relisted_on": observed,
                "observed_at": item.get("observed_at") or item.get("first_seen_at"),
                "ingestion_run_id": item.get("ingestion_run_id") or item.get("scrape_run_id"),
                "stable_observation_count": stable_count,
                "algorithm_version": ALGORITHM_VERSION,
            })
            trusted = observed
            stable_count = 1
            continue
        corrections.append({
            "trusted_posted_date": trusted,
            "observed_posted_date": observed,
            "observed_at": item.get("observed_at") or item.get("first_seen_at"),
            "reason": "backward_or_out_of_order" if delta < 0 else "unstable_or_below_threshold",
        })
    return {
        "latest_trusted_posted_date": trusted,
        "events": events,
        "corrections": corrections,
    }


def latest_pending_event(fold: dict, accepted_dates=None) -> dict | None:
    """Return the newest folded event not yet durably projected."""
    accepted = {date_part(value) for value in (accepted_dates or [])}
    pending = [
        event for event in fold.get("events", [])
        if date_part(event.get("relisted_on")) not in accepted
    ]
    return pending[-1] if pending else None
