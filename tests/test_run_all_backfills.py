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


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows=None, state=None):
        self.rows = rows or []
        self.state = state if state is not None else {}
        self.selected = None
        self.range_calls = []
        self.upsert_payloads = []

    def select(self, fields):
        self.selected = fields
        return self

    def range(self, start, end):
        self.range_calls.append((start, end))
        return self

    def execute(self):
        return _FakeResponse(self.rows)

    def upsert(self, payload):
        self.upsert_payloads.append(payload)
        return self


class _FakeSupabase:
    def __init__(self, select_query, upsert_query=None):
        self.select_query = select_query
        self.upsert_query = upsert_query or select_query

    def table(self, _name):
        if getattr(self, "_used_select", False):
            return self.upsert_query
        self._used_select = True
        return self.select_query


def test_needs_canonical_repair_returns_true_when_description_fingerprint_null_but_long_description():
    assert run_all_backfills.needs_canonical_repair({
        "canonical_key": "x",
        "original_job_id": "1",
        "latest_job_id": "1",
        "first_seen_at": "a",
        "last_seen_at": "a",
        "listing_instances": [{}],
        "description_fingerprint": None,
        "description": "long description " * 50,
    }) is True


def test_fetch_repair_candidates_returns_only_rows_needing_repair(monkeypatch):
    rows = [
        {
            "job_id": "1",
            "company": "Acme",
            "job_title": "TPM",
            "location": "Toronto",
            "description": "long description " * 50,
            "provider": "linkedin",
            "scraped_at": "2026-06-14T10:15:00+00:00",
            "posted_at": "2026-06-12",
            "canonical_key": None,
            "original_job_id": None,
            "latest_job_id": None,
            "first_seen_at": None,
            "last_seen_at": None,
            "listing_instances": None,
            "description_fingerprint": None,
        },
        {
            "job_id": "2",
            "company": "Acme",
            "job_title": "TPM",
            "location": "Toronto",
            "description": "short",
            "provider": "linkedin",
            "scraped_at": "2026-06-14T10:15:00+00:00",
            "posted_at": "2026-06-12",
            "canonical_key": "k",
            "original_job_id": "2",
            "latest_job_id": "2",
            "first_seen_at": "2026-06-14T10:15:00+00:00",
            "last_seen_at": "2026-06-14T10:15:00+00:00",
            "listing_instances": [{}],
            "description_fingerprint": None,
        },
    ]
    query = _FakeQuery(rows=rows)
    monkeypatch.setattr(run_all_backfills, "supabase", _FakeSupabase(query))

    result = run_all_backfills.fetch_repair_candidates(batch_size=1000)

    assert [row["job_id"] for row in result] == ["1"]
    assert "canonical_key" in query.selected
    assert "description_fingerprint" in query.selected


def test_backfill_canonical_fields_upserts_in_batches(monkeypatch):
    rows = [
        {
            "job_id": str(i),
            "company": "Acme",
            "job_title": "TPM",
            "location": "Toronto",
            "description": "long description " * 50,
            "provider": "linkedin",
            "scraped_at": "2026-06-14T10:15:00+00:00",
            "posted_at": "2026-06-12",
            "canonical_key": None,
            "original_job_id": None,
            "latest_job_id": None,
            "first_seen_at": None,
            "last_seen_at": None,
            "listing_instances": None,
            "description_fingerprint": None,
        }
        for i in range(205)
    ]
    select_query = _FakeQuery(rows=rows)
    upsert_query = _FakeQuery(rows=[])
    monkeypatch.setattr(run_all_backfills, "supabase", _FakeSupabase(select_query, upsert_query))

    repaired = run_all_backfills.backfill_canonical_fields(batch_size=100)

    assert repaired == 205
    assert len(upsert_query.upsert_payloads) == 3
    assert [len(batch) for batch in upsert_query.upsert_payloads] == [100, 100, 5]


def test_build_verification_report_marks_failed_checks():
    metrics = {
        "preflight_null_is_filtered": 3,
        "linkedin_archetype_nulls": 0,
        "linkedin_filter_profile_nulls": 0,
        "repair_canonical_key_nulls": 0,
        "repair_identity_nulls": 2,
        "repair_timestamp_nulls": 0,
        "repair_listing_instances_nulls": 0,
        "repair_scraped_mismatches": 0,
        "repair_posted_mismatches": 0,
        "legacy_aerospace_filter_rows": 0,
        "keyword_insights_count_before": 1739,
        "keyword_insights_count_after": 1739,
        "sample_jobs_ok": True,
    }

    report = run_all_backfills.build_verification_report(metrics)

    assert report[0]["name"] == "Preflight null is_filtered count"
    assert any(item["name"] == "Canonical identity coverage" and item["passed"] is False for item in report)
    assert run_all_backfills.verification_failed(report) is True


def test_build_verification_report_accepts_matching_keyword_counts_and_clean_metrics():
    metrics = {
        "preflight_null_is_filtered": 0,
        "linkedin_archetype_nulls": 0,
        "linkedin_filter_profile_nulls": 0,
        "repair_canonical_key_nulls": 0,
        "repair_identity_nulls": 0,
        "repair_timestamp_nulls": 0,
        "repair_listing_instances_nulls": 0,
        "repair_scraped_mismatches": 0,
        "repair_posted_mismatches": 0,
        "legacy_aerospace_filter_rows": 0,
        "keyword_insights_count_before": 1739,
        "keyword_insights_count_after": 1739,
        "sample_jobs_ok": True,
    }

    report = run_all_backfills.build_verification_report(metrics)

    assert all(item["passed"] for item in report if item["required"])
    assert run_all_backfills.verification_failed(report) is False
