import run_all_backfills


def test_build_historical_listing_instance_uses_stored_row_values():
    row = {
        "job_id": 4426608777,
        "scraped_at": "2026-06-14T10:15:00+00:00",
        "posted_at": "2026-06-12",
        "posted_relative_text": "2 days ago",
        "applicant_count": 17,
        "salary_text": "$120,000-$135,000 CAD",
        "recruiter_name": "Jane Recruiter",
        "recruiter_profile_url": "https://www.linkedin.com/in/jane-recruiter",
        "recruiter_identifier": "jane-recruiter",
    }

    listing = run_all_backfills.build_historical_listing_instance(row)

    assert listing == {
        "job_id": "4426608777",
        "scraped_at": "2026-06-14T10:15:00+00:00",
        "posted_at": "2026-06-12",
        "posted_relative_text": "2 days ago",
        "applicant_count": 17,
        "salary_text": "$120,000-$135,000 CAD",
        "recruiter_name": "Jane Recruiter",
        "recruiter_profile_url": "https://www.linkedin.com/in/jane-recruiter",
        "recruiter_identifier": "jane-recruiter",
    }


def test_build_historical_backfill_payload_uses_scraped_at_not_runtime_now():
    row = {
        "job_id": 4426608777,
        "provider": "linkedin",
        "company": "Foo-Bar",
        "job_title": "Sr. Project Manager",
        "location": "Toronto / Ontario - Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation. " * 12,
        "scraped_at": "2026-06-14T10:15:00+00:00",
        "posted_at": "2026-06-12",
        "posted_relative_text": "2 days ago",
        "applicant_count": 17,
        "salary_text": "$120,000-$135,000 CAD",
        "recruiter_name": "Jane Recruiter",
        "recruiter_profile_url": "https://www.linkedin.com/in/jane-recruiter",
        "recruiter_identifier": "jane-recruiter",
    }

    payload = run_all_backfills.build_historical_backfill_payload(row)

    assert payload["job_id"] == "4426608777"
    assert payload["canonical_key"] == "linkedin|foo bar|senior project manager|toronto ontario canada"
    assert payload["original_job_id"] == "4426608777"
    assert payload["latest_job_id"] == "4426608777"
    assert payload["first_seen_at"] == "2026-06-14T10:15:00+00:00"
    assert payload["last_seen_at"] == "2026-06-14T10:15:00+00:00"
    assert payload["last_seen_posted_at"] == "2026-06-12"
    assert payload["seen_count"] == 1
    assert payload["repost_count"] == 0
    assert payload["listing_instances"] == [
        run_all_backfills.build_historical_listing_instance(row)
    ]
    assert payload["description_fingerprint"] is not None


def test_needs_canonical_repair_detects_partial_state():
    assert run_all_backfills.needs_canonical_repair({"canonical_key": None}) is True
    assert run_all_backfills.needs_canonical_repair({"canonical_key": "x", "original_job_id": None}) is True
    assert run_all_backfills.needs_canonical_repair({"canonical_key": "x", "original_job_id": "1", "latest_job_id": None}) is True
    assert run_all_backfills.needs_canonical_repair({"canonical_key": "x", "original_job_id": "1", "latest_job_id": "1", "first_seen_at": None}) is True
    assert run_all_backfills.needs_canonical_repair({"canonical_key": "x", "original_job_id": "1", "latest_job_id": "1", "first_seen_at": "a", "last_seen_at": "a", "listing_instances": None}) is True
    assert run_all_backfills.needs_canonical_repair({
        "canonical_key": "x",
        "original_job_id": "1",
        "latest_job_id": "1",
        "first_seen_at": "a",
        "last_seen_at": "a",
        "listing_instances": [{}],
        "description_fingerprint": None,
        "description": "short desc",
    }) is False
