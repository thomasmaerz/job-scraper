"""Recover structured salary fields from stored listing-instance salary text."""

import argparse
import json

import config
import scraper
import supabase_utils


SELECT_FIELDS = "job_id,last_seen_at,listing_instances"


def fetch_jobs(page_size: int = 1000) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = (
            supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
            .select(SELECT_FIELDS)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def build_payload(row: dict) -> tuple[dict | None, int]:
    instances = [dict(instance) for instance in (row.get("listing_instances") or [])]
    recovered = 0
    for instance in instances:
        salary_text = instance.get("salary_text")
        if not str(salary_text or "").strip():
            continue
        parsed = scraper._parse_salary_fields(f"Salary: {salary_text}")
        changed = False
        for field in ("salary_min", "salary_max", "salary_currency"):
            if instance.get(field) is None and parsed.get(field) is not None:
                instance[field] = parsed[field]
                changed = True
        if changed:
            instance["salary_metadata_source"] = "salary_text_parser"
            recovered += 1
    return ({"listing_instances": instances} if recovered else None), recovered


def run(apply: bool) -> dict:
    result = {"scanned_jobs": 0, "recoverable_instances": 0, "updated_jobs": 0}
    for row in fetch_jobs():
        result["scanned_jobs"] += 1
        payload, recovered = build_payload(row)
        if payload is None:
            continue
        result["recoverable_instances"] += recovered
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
                f"Concurrent update detected for job_id={row['job_id']}; salary repair stopped"
            )
        result["updated_jobs"] += 1
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(run(apply=args.apply))
