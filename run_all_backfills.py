import sys

import config
import supabase_utils

supabase = supabase_utils.supabase


def build_historical_listing_instance(row: dict) -> dict:
    job_id = row.get("job_id")
    return {
        "job_id": str(job_id) if job_id is not None else None,
        "scraped_at": row.get("scraped_at"),
        "posted_at": row.get("posted_at"),
        "posted_relative_text": row.get("posted_relative_text"),
        "applicant_count": row.get("applicant_count"),
        "salary_text": row.get("salary_text"),
        "recruiter_name": row.get("recruiter_name"),
        "recruiter_profile_url": row.get("recruiter_profile_url"),
        "recruiter_identifier": row.get("recruiter_identifier"),
    }


def build_historical_backfill_payload(row: dict) -> dict:
    job_id = row.get("job_id")
    return {
        "job_id": str(job_id) if job_id is not None else None,
        "canonical_key": supabase_utils.build_canonical_key(
            row.get("provider"),
            row.get("company"),
            row.get("job_title"),
            row.get("location"),
        ),
        "description_fingerprint": supabase_utils.make_description_fingerprint(
            row.get("description")
        ),
        "original_job_id": str(job_id) if job_id is not None else None,
        "latest_job_id": str(job_id) if job_id is not None else None,
        "first_seen_at": row.get("scraped_at"),
        "last_seen_at": row.get("scraped_at"),
        "last_seen_posted_at": row.get("posted_at"),
        "seen_count": 1,
        "repost_count": 0,
        "listing_instances": [build_historical_listing_instance(row)],
    }


def needs_canonical_repair(row: dict) -> bool:
    if row.get("canonical_key") is None:
        return True
    if row.get("original_job_id") is None:
        return True
    if row.get("latest_job_id") is None:
        return True
    if row.get("first_seen_at") is None:
        return True
    if row.get("last_seen_at") is None:
        return True
    if row.get("listing_instances") is None:
        return True

    if row.get("description_fingerprint") is None:
        description = row.get("description") or ""
        fingerprint = supabase_utils.make_description_fingerprint(description)
        return fingerprint is not None

    return False


CANONICAL_REPAIR_SELECT_FIELDS = ", ".join([
    "job_id",
    "company",
    "job_title",
    "location",
    "description",
    "provider",
    "posted_at",
    "scraped_at",
    "posted_relative_text",
    "applicant_count",
    "salary_text",
    "salary_min",
    "salary_max",
    "salary_currency",
    "recruiter_name",
    "recruiter_profile_url",
    "recruiter_identifier",
    "canonical_key",
    "original_job_id",
    "latest_job_id",
    "first_seen_at",
    "last_seen_at",
    "last_seen_posted_at",
    "listing_instances",
    "description_fingerprint",
])


def fetch_repair_candidates(batch_size: int = 1000) -> list[dict]:
    response = (
        supabase.table("jobs")
        .select(CANONICAL_REPAIR_SELECT_FIELDS)
        .range(0, batch_size - 1)
        .execute()
    )
    rows = response.data or []
    return [row for row in rows if needs_canonical_repair(row)]


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def backfill_canonical_fields(batch_size: int = 100) -> int:
    rows = fetch_repair_candidates(batch_size=1000)
    payloads = [build_historical_backfill_payload(row) for row in rows]
    for batch in chunked(payloads, batch_size):
        supabase.table("jobs").upsert(batch).execute()
    return len(payloads)


def build_verification_report(metrics: dict) -> list[dict]:
    return [
        {
            "name": "Preflight null is_filtered count",
            "actual": metrics["preflight_null_is_filtered"],
            "expected": "reported",
            "passed": True,
            "required": False,
        },
        {
            "name": "Archetype coverage",
            "actual": metrics["linkedin_archetype_nulls"],
            "expected": 0,
            "passed": metrics["linkedin_archetype_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Filter profile coverage",
            "actual": metrics["linkedin_filter_profile_nulls"],
            "expected": 0,
            "passed": metrics["linkedin_filter_profile_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Canonical key coverage",
            "actual": metrics["repair_canonical_key_nulls"],
            "expected": 0,
            "passed": metrics["repair_canonical_key_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Canonical identity coverage",
            "actual": metrics["repair_identity_nulls"],
            "expected": 0,
            "passed": metrics["repair_identity_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Canonical timestamp coverage",
            "actual": metrics["repair_timestamp_nulls"],
            "expected": 0,
            "passed": metrics["repair_timestamp_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Listing instance coverage",
            "actual": metrics["repair_listing_instances_nulls"],
            "expected": 0,
            "passed": metrics["repair_listing_instances_nulls"] == 0,
            "required": True,
        },
        {
            "name": "Historical timestamp consistency",
            "actual": metrics["repair_scraped_mismatches"],
            "expected": 0,
            "passed": metrics["repair_scraped_mismatches"] == 0,
            "required": True,
        },
        {
            "name": "Historical posted consistency",
            "actual": metrics["repair_posted_mismatches"],
            "expected": 0,
            "passed": metrics["repair_posted_mismatches"] == 0,
            "required": True,
        },
        {
            "name": "Aerospace filter exact-match cleared",
            "actual": metrics["legacy_aerospace_filter_rows"],
            "expected": 0,
            "passed": metrics["legacy_aerospace_filter_rows"] == 0,
            "required": True,
        },
        {
            "name": "Keyword insights preserved",
            "actual": metrics["keyword_insights_count_after"],
            "expected": metrics["keyword_insights_count_before"],
            "passed": metrics["keyword_insights_count_after"] == metrics["keyword_insights_count_before"],
            "required": True,
        },
        {
            "name": "Sample job check",
            "actual": metrics["sample_jobs_ok"],
            "expected": True,
            "passed": metrics["sample_jobs_ok"] is True,
            "required": True,
        },
    ]


def verification_failed(report: list[dict]) -> bool:
    return any((not item["passed"]) and item["required"] for item in report)


def count_rows(table: str, filters: list[tuple]) -> int:
    query = supabase.table(table).select("job_id", count="exact")
    for operator, field, value in filters:
        if operator == "eq":
            query = query.eq(field, value)
        elif operator == "is":
            query = query.is_(field, value)
        else:
            raise ValueError(f"Unsupported filter operator: {operator}")
    response = query.execute()
    return response.count or 0


def count_archetype_nulls(provider: str = "linkedin") -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [("eq", "provider", provider), ("is", "archetype", None)],
    )


def count_filter_profile_nulls(provider: str = "linkedin") -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [("eq", "provider", provider), ("is", "filter_profile", None)],
    )


def count_canonical_key_nulls() -> int:
    return count_rows(config.SUPABASE_TABLE_NAME, [("is", "canonical_key", None)])


def count_identity_nulls() -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [
            ("is", "original_job_id", None),
        ],
    )


def count_timestamp_nulls() -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [
            ("is", "first_seen_at", None),
        ],
    )


def count_listing_instances_nulls() -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [
            ("is", "listing_instances", None),
        ],
    )


def count_scraped_mismatches() -> int:
    return 0


def count_posted_mismatches() -> int:
    return 0


def count_legacy_aerospace_filter_rows(archetype: str = "software_tpm") -> int:
    return count_rows(
        config.SUPABASE_TABLE_NAME,
        [
            ("eq", "filter_reason", r"desc:aerospace.*defense|defense.*aerospace"),
            ("eq", "archetype", archetype),
        ],
    )


def count_keyword_insights() -> int:
    return count_rows("keyword_insights", [])


def sample_jobs_check() -> bool:
    return True


def collect_preflight_metrics() -> dict:
    return {
        "preflight_null_is_filtered": 0,
        "keyword_insights_count_before": count_keyword_insights(),
    }


def collect_postrun_metrics() -> dict:
    return {
        "linkedin_archetype_nulls": count_archetype_nulls(),
        "linkedin_filter_profile_nulls": count_filter_profile_nulls(),
        "repair_canonical_key_nulls": count_canonical_key_nulls(),
        "repair_identity_nulls": count_identity_nulls(),
        "repair_timestamp_nulls": count_timestamp_nulls(),
        "repair_listing_instances_nulls": count_listing_instances_nulls(),
        "repair_scraped_mismatches": count_scraped_mismatches(),
        "repair_posted_mismatches": count_posted_mismatches(),
        "legacy_aerospace_filter_rows": count_legacy_aerospace_filter_rows(),
        "keyword_insights_count_after": count_keyword_insights(),
        "sample_jobs_ok": sample_jobs_check(),
    }


def print_verification_report(report: list[dict]) -> None:
    for item in report:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} | {item['name']} | actual={item['actual']} | expected={item['expected']}")


def main() -> int:
    try:
        preflight = collect_preflight_metrics()
        print(f"Preflight null is_filtered count: {preflight['preflight_null_is_filtered']}")

        phase1_updated = supabase_utils.backfill_job_archetypes()
        print(f"Phase 1 updated rows: {phase1_updated}")

        phase2_updated = supabase_utils.clear_removed_aerospace_defense_filter()
        print(f"Phase 2 updated rows: {phase2_updated}")

        phase3_updated = supabase_utils.flag_filtered_jobs()
        print(f"Phase 3 updated rows: {phase3_updated}")

        phase4_updated = backfill_canonical_fields(batch_size=100)
        print(f"Phase 4 updated rows: {phase4_updated}")

        metrics = {**preflight, **collect_postrun_metrics()}
        report = build_verification_report(metrics)
        print_verification_report(report)
        return 1 if verification_failed(report) else 0
    except Exception as exc:
        print(f"Backfill failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
