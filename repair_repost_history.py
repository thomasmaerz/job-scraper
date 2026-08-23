"""Repair listing-instance locations and posting-wave counts without deleting history."""

import argparse
import json
from collections import defaultdict

import config
import supabase_utils


JOBS_FIELDS = (
    "job_id,provider,company,job_title,location,posted_at,scraped_at,canonical_key,original_job_id,"
    "latest_job_id,last_seen_at,listing_instances,seen_count,posting_wave_count,repost_count"
)


def fetch_all(table: str, fields: str, page_size: int = 1000) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = (
            supabase_utils.supabase.table(table)
            .select(fields)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def fetch_archive(page_size: int = 1000) -> list[dict]:
    try:
        return fetch_all(
            "job_listing_archive",
            "canonical_job_id,source_job_id,observed_at,source_snapshot",
            page_size=page_size,
        )
    except Exception as exc:
        if "job_listing_archive" not in str(exc):
            raise
        return []


def build_repair_payload(row: dict, archived_rows: list[dict]) -> dict:
    instances_by_id = {
        str(instance.get("job_id")): dict(instance)
        for instance in (row.get("listing_instances") or [])
        if instance.get("job_id") is not None
    }

    for archived in archived_rows:
        source_id = str(archived["source_job_id"])
        snapshot = archived.get("source_snapshot") or {}
        instance = instances_by_id.setdefault(source_id, {"job_id": source_id})
        for field in (
            "scraped_at",
            "posted_at",
            "posted_relative_text",
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
            "detail_metadata_checked_at",
            "scrape_run_id",
            "location",
        ):
            if instance.get(field) is None and snapshot.get(field) is not None:
                instance[field] = snapshot[field]
                if field == "location":
                    instance["location_source"] = "source_snapshot"
                    instance["location_observed_at"] = archived.get("observed_at")
        if instance.get("scraped_at") is None:
            instance["scraped_at"] = archived.get("observed_at")

    for identity_field in ("original_job_id", "latest_job_id"):
        source_id = row.get(identity_field)
        if source_id is not None:
            instances_by_id.setdefault(str(source_id), {"job_id": str(source_id)})

    canonical_id = str(row["job_id"])
    canonical_instance = instances_by_id.get(canonical_id)
    if canonical_instance is not None and canonical_instance.get("location") is None:
        canonical_instance["location"] = row.get("location")
        if row.get("location") is not None:
            canonical_instance["location_source"] = "canonical_anchor"
            canonical_instance["location_observed_at"] = row.get("scraped_at")

    if not instances_by_id:
        instance = supabase_utils.build_listing_instance(row)
        instances_by_id[canonical_id] = instance

    ordered_instances = sorted(
        instances_by_id.values(),
        key=lambda instance: (
            str(instance.get("scraped_at") or ""),
            str(instance.get("job_id") or ""),
        ),
    )
    instances, posting_wave_count, repost_count = supabase_utils.calculate_posting_waves(
        ordered_instances
    )
    listing_ids = {str(instance["job_id"]) for instance in instances if instance.get("job_id")}
    return {
        "job_id": canonical_id,
        "canonical_key": supabase_utils.build_canonical_key(
            row.get("provider"), row.get("company"), row.get("job_title"), row.get("location")
        ),
        "seen_count": len(listing_ids),
        "posting_wave_count": posting_wave_count,
        "repost_count": repost_count,
        "listing_instances": instances,
    }


def run(apply: bool, page_size: int = 1000) -> dict:
    jobs = fetch_all(config.SUPABASE_TABLE_NAME, JOBS_FIELDS, page_size=page_size)
    archive = fetch_archive(page_size=page_size)
    archive_by_canonical = defaultdict(list)
    for archived in archive:
        archive_by_canonical[str(archived["canonical_job_id"])].append(archived)

    payloads = [
        build_repair_payload(row, archive_by_canonical.get(str(row["job_id"]), []))
        for row in jobs
    ]
    changed = [
        payload
        for row, payload in zip(jobs, payloads)
        if any(row.get(field) != payload[field] for field in payload if field != "job_id")
    ]
    repaired_instances = [
        instance
        for payload in payloads
        for instance in payload["listing_instances"]
    ]
    if apply:
        for payload in changed:
            source_row = next(row for row in jobs if str(row["job_id"]) == payload["job_id"])
            query = (
                supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
                .update({key: value for key, value in payload.items() if key != "job_id"})
                .eq("job_id", payload["job_id"])
                .eq("listing_instances", json.dumps(source_row.get("listing_instances") or []))
            )
            last_seen_at = source_row.get("last_seen_at")
            query = query.is_("last_seen_at", None) if last_seen_at is None else query.eq("last_seen_at", last_seen_at)
            response = query.execute()
            if len(response.data or []) != 1:
                raise RuntimeError(
                    f"Concurrent update detected for job_id={payload['job_id']}; repair stopped"
                )
    return {
        "scanned": len(jobs),
        "changed": len(changed),
        "changed_job_ids_sample": [payload["job_id"] for payload in changed[:25]],
        "before_listing_ids": sum(int(row.get("seen_count") or 0) for row in jobs),
        "after_listing_ids": sum(payload["seen_count"] for payload in payloads),
        "before_reposts": sum(int(row.get("repost_count") or 0) for row in jobs),
        "after_reposts": sum(payload["repost_count"] for payload in payloads),
        "locations_recovered_from_archive": sum(
            instance.get("location_source") == "source_snapshot"
            for instance in repaired_instances
        ),
        "locations_recovered_from_anchor": sum(
            instance.get("location_source") == "canonical_anchor"
            for instance in repaired_instances
        ),
        "locations_still_missing": sum(
            not str(instance.get("location") or "").strip()
            for instance in repaired_instances
        ),
        "applied": len(changed) if apply else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    print(run(apply=args.apply, page_size=args.page_size))
