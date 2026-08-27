"""Dry-run-first lower-bound repair from stored same-ID observations."""

import argparse
from collections import defaultdict

import config
import relist_tracking
import supabase_utils


JOBS_FIELDS = (
    "job_id,provider,location,description,last_seen_at,listing_instances,seen_count,posting_wave_count,"
    "repost_count,same_id_relist_count"
)
SELECTED_IDS_QUERY_CHUNK_SIZE = 100


def _selected_id_chunks(selected_ids: list[str] | None) -> list[list[str] | None]:
    if selected_ids is None:
        return [None]

    # Avoid duplicate queries/results while retaining the caller's ID order.
    unique_ids = list(dict.fromkeys(selected_ids))
    return [
        unique_ids[start:start + SELECTED_IDS_QUERY_CHUNK_SIZE]
        for start in range(0, len(unique_ids), SELECTED_IDS_QUERY_CHUNK_SIZE)
    ]


def fetch_all(
    table: str,
    fields: str,
    limit: int | None,
    page_size: int = 1000,
    selected_ids: list[str] | None = None,
    id_field: str = "canonical_job_id",
) -> list[dict]:
    rows = []
    for selected_id_chunk in _selected_id_chunks(selected_ids):
        offset = 0
        while limit is None or len(rows) < limit:
            request_size = page_size if limit is None else min(page_size, limit - len(rows))
            end = offset + request_size - 1
            query = supabase_utils.supabase.table(table).select(fields)
            if selected_id_chunk is not None:
                query = query.in_(id_field, selected_id_chunk)
            page = query.range(offset, end).execute().data or []
            rows.extend(page)
            if len(page) < request_size:
                break
            offset += len(page)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_payload(row: dict, observations: list[dict]) -> dict:
    instances = [dict(instance) for instance in (row.get("listing_instances") or [])]
    by_source = defaultdict(list)
    for observation in observations:
        by_source[str(observation["source_job_id"])].append(observation)

    accepted = 0
    existing_keys = {
        (str(instance.get("job_id")), relist_tracking.date_part(instance.get("posted_at")))
        for instance in instances
    }
    for source_id in sorted(by_source):
        fold = relist_tracking.fold_observations(
            by_source[source_id],
            min_forward_days=getattr(config, "LINKEDIN_RELIST_MIN_FORWARD_DAYS", 2),
            stable_observations=getattr(config, "LINKEDIN_RELIST_STABLE_OBSERVATIONS", 2),
        )
        source_instance = next(
            (instance for instance in instances if str(instance.get("job_id")) == source_id),
            None,
        )
        if source_instance is None:
            continue
        for event in fold["events"]:
            key = (source_id, event["relisted_on"])
            if key in existing_keys:
                continue
            instances.append({
                "job_id": source_id,
                "location": source_instance.get("location") or row.get("location"),
                "posted_at": event["relisted_on"],
                "scraped_at": event.get("observed_at"),
                "scrape_run_id": event.get("ingestion_run_id"),
                "same_id_relist": True,
                "relist_algorithm_version": event["algorithm_version"],
            })
            existing_keys.add(key)
            accepted += 1

    instances = sorted(
        instances,
        key=lambda item: (
            str(item.get("posted_at") or item.get("scraped_at") or ""),
            str(item.get("job_id") or ""),
        ),
    )
    instances, wave_count, repost_count = supabase_utils.calculate_posting_waves(instances)
    distinct_ids = {str(item["job_id"]) for item in instances if item.get("job_id") is not None}
    return {
        "job_id": str(row["job_id"]),
        "listing_instances": instances,
        "seen_count": len(distinct_ids),
        "posting_wave_count": wave_count,
        "repost_count": repost_count,
        "same_id_relist_count": max(
            int(row.get("same_id_relist_count") or 0),
            sum(bool(instance.get("same_id_relist")) for instance in instances),
        ),
    }


def build_content_version_seeds(row: dict, archived_descriptions: list[dict]) -> list[dict]:
    source_ids = sorted({
        str(instance["job_id"])
        for instance in (row.get("listing_instances") or [])
        if instance.get("job_id") is not None
    })
    fallback_source_id = source_ids[-1] if source_ids else str(row["job_id"])
    candidates = list(archived_descriptions)
    if row.get("description"):
        candidates.append({
            "source_job_id": fallback_source_id,
            "description": row["description"],
            "observed_at": row.get("last_seen_at"),
        })
    by_key = {}
    for candidate in candidates:
        description = candidate.get("description")
        source_id = str(candidate.get("source_job_id") or fallback_source_id)
        content_hash = supabase_utils.make_description_content_hash(description)
        if not content_hash:
            continue
        key = (source_id, content_hash)
        by_key.setdefault(key, {
            "provider": row.get("provider") or "linkedin",
            "source_job_id": source_id,
            "content_hash": content_hash,
            "canonical_job_id": str(row["job_id"]),
            "description": description,
            "description_fingerprint": supabase_utils.make_description_fingerprint(description),
            "first_observed_at": candidate.get("observed_at") or row.get("last_seen_at"),
            "last_observed_at": candidate.get("observed_at") or row.get("last_seen_at"),
        })
    return [by_key[key] for key in sorted(by_key)]


def fetch_archive(limit: int, page_size: int, selected_ids: list[str]) -> list[dict]:
    try:
        return fetch_all(
            "job_listing_archive",
            "canonical_job_id,source_job_id,observed_at,source_snapshot",
            limit,
            page_size,
            selected_ids=selected_ids,
        )
    except Exception as exc:
        if "job_listing_archive" not in str(exc):
            raise
        return []


def run(limit: int = 100, apply: bool = False, page_size: int = 1000) -> dict:
    rows = fetch_all(config.SUPABASE_TABLE_NAME, JOBS_FIELDS, limit, page_size)
    selected_ids = [str(row["job_id"]) for row in rows]
    observations = fetch_all(
        "listing_observations",
        "provider,source_job_id,canonical_job_id,ingestion_run_id,observed_at,posted_at,result",
        None,
        page_size,
        selected_ids=selected_ids,
    )
    archive = fetch_archive(max(limit * 100, 1000), page_size, selected_ids)
    by_canonical = defaultdict(list)
    for observation in observations:
        if observation.get("canonical_job_id") is not None and observation.get("result") == "seen":
            by_canonical[str(observation["canonical_job_id"])].append(observation)
    descriptions_by_canonical = defaultdict(list)
    for archived in archive:
        snapshot = archived.get("source_snapshot") or {}
        if snapshot.get("description"):
            descriptions_by_canonical[str(archived["canonical_job_id"])].append({
                "source_job_id": archived.get("source_job_id"),
                "description": snapshot["description"],
                "observed_at": archived.get("observed_at"),
            })

    result = {
        "scanned": len(rows),
        "changed": 0,
        "inferred_lower_bound_events": 0,
        "applied": 0,
        "conflicts": 0,
        "content_versions_seeded": 0,
        "dry_run": not apply,
    }
    for row in rows:
        content_seeds = build_content_version_seeds(
            row,
            descriptions_by_canonical.get(str(row["job_id"]), []),
        )
        result["content_versions_seeded"] += len(content_seeds)
        if apply and content_seeds:
            (
                supabase_utils.supabase.table("listing_content_versions")
                .upsert(
                    content_seeds,
                    on_conflict="provider,source_job_id,content_hash",
                    ignore_duplicates=True,
                )
                .execute()
            )
        payload = build_payload(row, by_canonical.get(str(row["job_id"]), []))
        changed = any(
            row.get(field) != payload[field]
            for field in payload
            if field != "job_id"
        )
        if not changed:
            continue
        result["changed"] += 1
        result["inferred_lower_bound_events"] += max(
            0,
            payload["same_id_relist_count"] - int(row.get("same_id_relist_count") or 0),
        )
        if not apply:
            continue
        response = supabase_utils.supabase.rpc("apply_same_id_relist_repair", {
            "p_canonical_job_id": payload["job_id"],
            "p_expected_listing_instances": row.get("listing_instances"),
            "p_expected_last_seen_at": row.get("last_seen_at"),
            "p_payload": {key: value for key, value in payload.items() if key != "job_id"},
        }).execute()
        if response.data is not True:
            result["conflicts"] += 1
            continue
        result["applied"] += 1
    return result


if __name__ == "__main__":
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=positive_int, default=100)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(run(limit=args.limit, apply=args.apply))
