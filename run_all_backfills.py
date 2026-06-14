import supabase_utils


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


def main() -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
