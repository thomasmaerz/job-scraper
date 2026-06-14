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


def main() -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
