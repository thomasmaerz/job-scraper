import backfill_linkedin_metadata


def test_build_metadata_payload_updates_latest_listing_without_changing_counts():
    row = {
        "job_id": "canonical",
        "latest_job_id": "source-2",
        "location": "Toronto",
        "listing_instances": [
            {"job_id": "source-1", "applicant_count": 10},
            {"job_id": "source-2", "applicant_count": None},
        ],
    }
    details = {
        "applicant_count": 25,
        "applicant_count_text": "Be among the first 25 applicants",
        "applicant_count_type": "upper_bound",
        "detail_metadata_checked_at": "2026-08-22T00:00:00+00:00",
    }

    payload = backfill_linkedin_metadata.build_metadata_payload(row, details)

    assert payload["applicant_count"] == 25
    assert payload["listing_instances"][0]["applicant_count"] == 10
    assert payload["listing_instances"][1]["applicant_count"] == 25
    assert payload["listing_instances"][1].get("location") is None
    assert "seen_count" not in payload
    assert payload["posting_wave_count"] == 1
    assert payload["repost_count"] == 0
