"""Conservatively re-fetch unresolved LinkedIn listing locations."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone

import config
import scraper
import supabase_utils
from linkedin_source_policy import DurableLinkedInRequestGate


SELECT_FIELDS = "job_id,last_seen_at,listing_instances,posting_wave_count,repost_count"


def fetch_candidates(limit: int, page_size: int = 1000) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = (
            supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
            .select(SELECT_FIELDS)
            .eq("provider", "linkedin")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    grouped = defaultdict(list)
    row_by_id = {}
    for row in rows:
        canonical_id = str(row["job_id"])
        row_by_id[canonical_id] = row
        for instance in row.get("listing_instances") or []:
            source_id = instance.get("job_id")
            if source_id is None or str(instance.get("location") or "").strip():
                continue
            grouped[canonical_id].append(instance)

    canonical_ids = sorted(
        grouped,
        key=lambda canonical_id: str(row_by_id[canonical_id].get("last_seen_at") or ""),
        reverse=True,
    )
    for instances in grouped.values():
        instances.sort(
            key=lambda instance: str(instance.get("scraped_at") or ""),
            reverse=True,
        )

    candidates = []
    round_index = 0
    while len(candidates) < limit:
        added = False
        for canonical_id in canonical_ids:
            instances = grouped[canonical_id]
            if round_index >= len(instances):
                continue
            candidates.append({
                "row": row_by_id[canonical_id],
                "source_job_id": str(instances[round_index]["job_id"]),
            })
            added = True
            if len(candidates) == limit:
                break
        if not added:
            break
        round_index += 1
    return candidates


def build_location_payload(row: dict, source_job_id: str, details: dict, observed_at: str) -> dict | None:
    location = details.get("location")
    if not str(location or "").strip():
        return None

    instances = [dict(instance) for instance in (row.get("listing_instances") or [])]
    changed = False
    for instance in instances:
        if str(instance.get("job_id")) != source_job_id:
            continue
        if str(instance.get("location") or "").strip():
            return None
        instance["location"] = location
        instance["location_source"] = "linkedin_rescrape"
        instance["location_observed_at"] = observed_at
        instance["detail_metadata_checked_at"] = details.get("detail_metadata_checked_at") or observed_at
        changed = True
        break
    if not changed:
        return None

    instances, posting_wave_count, repost_count = supabase_utils.calculate_posting_waves(instances)
    return {
        "listing_instances": instances,
        "posting_wave_count": posting_wave_count,
        "repost_count": repost_count,
    }


def run(limit: int, apply: bool) -> dict:
    result = {
        "selected": 0,
        "available": 0,
        "locations_recovered": 0,
        "unavailable": 0,
        "updated": 0,
        "recovered_source_ids": [],
    }
    gate = DurableLinkedInRequestGate("location-backfill")
    user_agent = scraper.user_agents.USER_AGENTS[0]
    current_rows = {}
    for candidate in fetch_candidates(limit=limit):
        original_row = candidate["row"]
        canonical_id = str(original_row["job_id"])
        row = current_rows.get(canonical_id, original_row)
        source_job_id = candidate["source_job_id"]
        result["selected"] += 1
        detail_result = scraper._fetch_linkedin_job_details(
            source_job_id, durable_gate=gate, user_agent=user_agent
        )
        if not detail_result:
            result["unavailable"] += 1
            continue
        details, detail_metadata = detail_result
        details.update(detail_metadata)
        result["available"] += 1
        observed_at = datetime.now(timezone.utc).isoformat()
        payload = build_location_payload(row, source_job_id, details, observed_at)
        if payload is None:
            continue
        result["locations_recovered"] += 1
        result["recovered_source_ids"].append(source_job_id)
        if not apply:
            continue

        query = (
            supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
            .update(payload)
            .eq("job_id", row["job_id"])
            .eq("listing_instances", json.dumps(row.get("listing_instances") or []))
        )
        last_seen_at = row.get("last_seen_at")
        query = query.is_("last_seen_at", None) if last_seen_at is None else query.eq("last_seen_at", last_seen_at)
        response = query.execute()
        if len(response.data or []) != 1:
            raise RuntimeError(
                f"Concurrent update detected for job_id={row['job_id']}; rescrape stopped"
            )
        result["updated"] += 1
        current_rows[canonical_id] = {
            **row,
            **payload,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(run(limit=args.limit, apply=args.apply))
