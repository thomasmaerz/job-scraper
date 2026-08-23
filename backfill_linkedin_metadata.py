"""Re-fetch LinkedIn detail metadata without creating jobs or repost observations."""

import argparse
import logging
import time
from datetime import datetime, timezone

import config
import scraper
import supabase_utils


METADATA_FIELDS = (
    "applicant_count",
    "applicant_count_text",
    "applicant_count_type",
    "salary_text",
    "salary_min",
    "salary_max",
    "salary_currency",
    "recruiter_name",
    "recruiter_profile_url",
    "recruiter_identifier",
)


def fetch_candidates(limit: int) -> list[dict]:
    return (
        supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
        .select("job_id, latest_job_id, location, listing_instances")
        .eq("provider", "linkedin")
        .is_("detail_metadata_checked_at", None)
        .order("is_active", desc=True)
        .order("scraped_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )


def build_metadata_payload(row: dict, details: dict) -> dict:
    source_job_id = str(row.get("latest_job_id") or row["job_id"])
    payload = {
        field: details.get(field)
        for field in METADATA_FIELDS
        if details.get(field) is not None
    }
    payload["detail_metadata_checked_at"] = details.get("detail_metadata_checked_at") or datetime.now(timezone.utc).isoformat()

    instances = list(row.get("listing_instances") or [])
    for instance in instances:
        if str(instance.get("job_id")) != source_job_id:
            continue
        for field in METADATA_FIELDS:
            if details.get(field) is not None:
                instance[field] = details[field]
        if details.get("location") is not None:
            instance["location"] = details["location"]
        instance["detail_metadata_checked_at"] = payload["detail_metadata_checked_at"]
        instances, posting_wave_count, repost_count = supabase_utils.calculate_posting_waves(instances)
        payload["listing_instances"] = instances
        payload["posting_wave_count"] = posting_wave_count
        payload["repost_count"] = repost_count
        break
    return payload


def run(limit: int, apply: bool) -> dict:
    result = {"checked": 0, "available": 0, "updated": 0, "unavailable": 0}
    for row in fetch_candidates(limit):
        source_job_id = str(row.get("latest_job_id") or row["job_id"])
        details = scraper._fetch_linkedin_job_details(source_job_id)
        result["checked"] += 1
        if not details:
            result["unavailable"] += 1
            if apply:
                (
                    supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
                    .update({"detail_metadata_checked_at": datetime.now(timezone.utc).isoformat()})
                    .eq("job_id", row["job_id"])
                    .execute()
                )
            continue
        result["available"] += 1
        payload = build_metadata_payload(row, details)
        if apply:
            (
                supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
                .update(payload)
                .eq("job_id", row["job_id"])
                .execute()
            )
            result["updated"] += 1
        logging.info("job_id=%s source_job_id=%s fields=%s", row["job_id"], source_job_id, sorted(payload))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(limit=args.limit, apply=args.apply))
