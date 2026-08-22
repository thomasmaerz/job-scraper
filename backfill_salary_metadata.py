"""Recover structured salary ranges from stored job descriptions."""

import argparse

import config
import scraper
import supabase_utils


def fetch_missing_salary_jobs(page_size: int = 1000) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = (
            supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
            .select("job_id, description")
            .is_("salary_text", None)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def run(apply: bool) -> dict:
    result = {"scanned": 0, "recoverable": 0, "updated": 0}
    for row in fetch_missing_salary_jobs():
        result["scanned"] += 1
        salary = scraper._parse_salary_fields(row.get("description") or "")
        if not salary["salary_text"]:
            continue
        result["recoverable"] += 1
        if apply:
            (
                supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
                .update(salary)
                .eq("job_id", row["job_id"])
                .execute()
            )
            result["updated"] += 1
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(run(apply=args.apply))
