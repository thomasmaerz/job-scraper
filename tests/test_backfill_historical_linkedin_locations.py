import backfill_historical_linkedin_locations as backfill


def test_build_location_payload_updates_only_matching_instance_and_recalculates_waves():
    row = {
        "job_id": "canonical",
        "listing_instances": [
            {"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"},
            {"job_id": "source-2", "location": None, "posted_at": "2026-08-01"},
        ],
    }

    payload = backfill.build_location_payload(
        row,
        "source-2",
        {"location": "Calgary", "detail_metadata_checked_at": "2026-08-22T12:00:00Z"},
        "2026-08-22T12:00:00Z",
    )

    assert payload["listing_instances"][0]["location"] == "Toronto"
    assert payload["listing_instances"][1]["location"] == "Calgary"
    assert payload["listing_instances"][1]["location_source"] == "linkedin_rescrape"
    assert payload["listing_instances"][1]["location_observed_at"] == "2026-08-22T12:00:00Z"
    assert payload["posting_wave_count"] == 1
    assert payload["repost_count"] == 0


def test_build_location_payload_rejects_missing_or_existing_location():
    row = {"listing_instances": [{"job_id": "source-1", "location": None}]}
    assert backfill.build_location_payload(row, "source-1", {"location": None}, "now") is None

    row["listing_instances"][0]["location"] = "Toronto"
    assert backfill.build_location_payload(row, "source-1", {"location": "Calgary"}, "now") is None


def test_fetch_candidates_stratifies_across_canonical_groups(monkeypatch):
    rows = [
        {
            "job_id": "a",
            "last_seen_at": "2026-08-20",
            "listing_instances": [
                {"job_id": "a-1", "location": None, "scraped_at": "2026-08-20"},
                {"job_id": "a-2", "location": None, "scraped_at": "2026-08-19"},
            ],
        },
        {
            "job_id": "b",
            "last_seen_at": "2026-08-21",
            "listing_instances": [{"job_id": "b-1", "location": None, "scraped_at": "2026-08-21"}],
        },
    ]

    class Query:
        def select(self, _fields): return self
        def eq(self, _field, _value): return self
        def range(self, _start, _end): return self
        def execute(self): return type("Response", (), {"data": rows})()

    monkeypatch.setattr(backfill.supabase_utils, "supabase", type("Client", (), {"table": lambda *_: Query()})())

    candidates = backfill.fetch_candidates(limit=3, page_size=1000)

    assert [candidate["source_job_id"] for candidate in candidates] == ["b-1", "a-1", "a-2"]


def test_run_is_dry_run_by_default_behavior(monkeypatch):
    row = {
        "job_id": "canonical",
        "listing_instances": [{"job_id": "source-1", "location": None}],
    }
    monkeypatch.setattr(backfill, "fetch_candidates", lambda limit: [{"row": row, "source_job_id": "source-1"}])
    monkeypatch.setattr(backfill.scraper, "_fetch_linkedin_job_details", lambda _job_id: ({"location": "Toronto"}, {}))

    result = backfill.run(limit=1, apply=False)

    assert result["selected"] == 1
    assert result["locations_recovered"] == 1
    assert result["updated"] == 0


def test_run_applies_multiple_instances_from_same_canonical_row(monkeypatch):
    row = {
        "job_id": "canonical",
        "last_seen_at": "2026-08-22T00:00:00Z",
        "listing_instances": [
            {"job_id": "source-1", "location": None},
            {"job_id": "source-2", "location": None},
        ],
    }
    monkeypatch.setattr(backfill, "fetch_candidates", lambda limit: [
        {"row": row, "source_job_id": "source-1"},
        {"row": row, "source_job_id": "source-2"},
    ])
    monkeypatch.setattr(
        backfill.scraper,
        "_fetch_linkedin_job_details",
        lambda job_id: ({"location": "Toronto" if job_id == "source-1" else "Calgary"}, {}),
    )

    updates = []

    class Query:
        def update(self, payload):
            updates.append(payload)
            return self
        def eq(self, _field, _value): return self
        def is_(self, _field, _value): return self
        def execute(self): return type("Response", (), {"data": [{"job_id": "canonical"}]})()

    monkeypatch.setattr(backfill.supabase_utils, "supabase", type("Client", (), {"table": lambda *_: Query()})())

    result = backfill.run(limit=2, apply=True)

    assert result["updated"] == 2
    assert updates[1]["listing_instances"][0]["location"] == "Toronto"
    assert updates[1]["listing_instances"][1]["location"] == "Calgary"
