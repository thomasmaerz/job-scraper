import repair_repost_history


def test_repair_recovers_archived_locations_without_inventing_missing_values():
    row = {
        "job_id": "canonical",
        "provider": "linkedin",
        "company": "Acme",
        "job_title": "TPM",
        "location": "Toronto",
        "listing_instances": [
            {"job_id": "canonical", "posted_at": "2026-08-01", "scraped_at": "2026-08-01T10:00:00Z"},
            {"job_id": "unknown-location", "posted_at": "2026-08-20", "scraped_at": "2026-08-20T10:00:00Z"},
        ],
    }
    archive = [
        {
            "source_job_id": "archived",
            "observed_at": "2026-08-01T10:00:00Z",
            "source_snapshot": {"location": "Calgary", "posted_at": "2026-08-01"},
        }
    ]

    payload = repair_repost_history.build_repair_payload(row, archive)
    instances = {instance["job_id"]: instance for instance in payload["listing_instances"]}

    assert instances["canonical"]["location"] == "Toronto"
    assert instances["canonical"]["location_source"] == "canonical_anchor"
    assert instances["archived"]["location"] == "Calgary"
    assert instances["archived"]["location_source"] == "source_snapshot"
    assert instances["unknown-location"].get("location") is None
    assert payload["seen_count"] == 3
    assert payload["repost_count"] == 0


def test_repair_is_deterministic_and_preserves_optional_metadata():
    row = {
        "job_id": "canonical",
        "provider": "linkedin",
        "company": "Acme",
        "job_title": "TPM",
        "location": "Toronto",
        "listing_instances": [
            {
                "job_id": "canonical",
                "location": "Toronto",
                "posted_at": "2026-08-01",
                "scraped_at": "2026-08-01T10:00:00Z",
                "salary_text": "$100k",
            },
            {
                "job_id": "later",
                "location": "Toronto",
                "posted_at": "2026-08-20",
                "scraped_at": "2026-08-20T10:00:00Z",
                "recruiter_identifier": "recruiter-a",
            },
        ],
    }

    first = repair_repost_history.build_repair_payload(row, [])
    repaired_row = {**row, **first}
    second = repair_repost_history.build_repair_payload(repaired_row, [])

    assert first == second
    assert first["posting_wave_count"] == 2
    assert first["repost_count"] == 1
    assert first["listing_instances"][0]["salary_text"] == "$100k"
    assert first["listing_instances"][1]["recruiter_identifier"] == "recruiter-a"
