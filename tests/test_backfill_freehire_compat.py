from types import SimpleNamespace

import pytest

import backfill_freehire_compat
import freehire_compat
import frontfill_freehire_compat


class Query:
    def __init__(self, db, rows=None):
        self.db = db
        self.rows = rows
        self.filters = []
        self.desc = False
        self.limit_value = None
        self.orders = []

    @property
    def not_(self):
        query = self

        class Negated:
            def is_(self, field, value):
                query.filters.append(("not_is", field, value))
                return query

        return Negated()

    def select(self, _fields):
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def gt(self, field, value):
        self.filters.append(("gt", field, value))
        return self

    def lte(self, field, value):
        self.filters.append(("lte", field, value))
        return self

    def or_(self, value):
        self.db.or_filters.append(value)
        return self

    def is_(self, field, value):
        self.filters.append(("is", field, value))
        return self

    def order(self, field, desc=False, **_kwargs):
        self.desc = desc
        self.orders.append((field, desc))
        self.db.orders.append((field, desc))
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def range(self, _start, _end):
        return self

    def update(self, payload):
        self.rows = payload
        return self

    def execute(self):
        if isinstance(self.rows, dict):
            matched = []
            for row in self.db.rows:
                if self._matches(row):
                    row.update(self.rows)
                    matched.append(dict(row))
            self.db.updates += len(matched)
            return SimpleNamespace(data=matched)
        rows = [dict(row) for row in self.db.rows if self._matches(row)]
        rows.sort(key=lambda row: row["job_id"], reverse=self.desc)
        if self.limit_value is not None:
            rows = rows[:self.limit_value]
        return SimpleNamespace(data=rows)

    def _matches(self, row):
        for op, field, value in self.filters:
            if op == "eq" and row.get(field) != value:
                return False
            if op == "is" and row.get(field) is not value:
                return False
            if op == "not_is" and row.get(field) is value:
                return False
            if op == "gt" and not row.get(field) > value:
                return False
            if op == "lte" and not row.get(field) <= value:
                return False
        return True


class Db:
    def __init__(self, rows):
        self.rows = rows
        self.updates = 0
        self.or_filters = []
        self.orders = []

    def table(self, name):
        assert name == "jobs"
        return Query(self)

    def rpc(self, name, params):
        db = self

        class Rpc:
            def execute(self):
                row = next((item for item in db.rows if item["job_id"] == params["p_job_id"]), None)
                if row is None:
                    return SimpleNamespace(data=False)
                if name == "claim_freehire_compat_job":
                    row.update({
                        "freehire_compat_status": "processing",
                        "freehire_compat_input_hash": params["p_expected_input_hash"],
                        "freehire_compat_claimed_by": params["p_worker_id"],
                    })
                elif name == "persist_freehire_compat_result":
                    if row.get("freehire_compat_claimed_by") != params["p_worker_id"]:
                        return SimpleNamespace(data=False)
                    row.update(params["p_payload"])
                elif name == "apply_freehire_compat_metadata":
                    snapshot = params["p_expected_source_snapshot"]
                    if any(row.get(key) != value for key, value in snapshot.items()):
                        return SimpleNamespace(data=False)
                    row.update(params["p_payload"])
                else:
                    raise AssertionError(name)
                db.updates += 1
                return SimpleNamespace(data=True)

        return Rpc()


class Client:
    model = "fake/model"
    model_chain = None
    last_model_used = "fake/model"

    def __init__(self):
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        return '{"jobs":[{"job_id":"1","category":"project_management","seniority":"senior","confidence":0.9}]}'


def test_mixed_success_and_failure_exits_successfully(monkeypatch):
    result = {"classified": 326, "failed": 174}
    monkeypatch.setattr(backfill_freehire_compat, "run", lambda **_kwargs: result)

    assert backfill_freehire_compat.result_status(result) == "partial_success"
    assert backfill_freehire_compat.main(["--apply"]) == 0


def test_all_failed_with_zero_progress_exits_nonzero(monkeypatch):
    result = {"classified": 0, "failed": 174}
    monkeypatch.setattr(backfill_freehire_compat, "run", lambda **_kwargs: result)

    assert backfill_freehire_compat.result_status(result) == "all_failed"
    assert backfill_freehire_compat.main(["--apply"]) == 1


def test_no_failure_exits_successfully(monkeypatch):
    result = {"classified": 500, "failed": 0}
    monkeypatch.setattr(backfill_freehire_compat, "run", lambda **_kwargs: result)

    assert backfill_freehire_compat.result_status(result) == "success"
    assert backfill_freehire_compat.main(["--apply"]) == 0


def test_apply_is_resumable_and_unchanged_rerun_uses_zero_llm_calls():
    db = Db([{
        "job_id": "1",
        "latest_job_id": "live",
        "provider": "linkedin",
        "job_title": "Senior TPM",
        "location": "Remote",
        "description": "Program delivery",
        "level": "Senior",
        "posted_at": "2026-01-01T00:00:00Z",
        "scraped_at": "2026-01-01T01:00:00Z",
        "first_seen_at": "2026-01-01T01:00:00Z",
        "last_seen_at": "2026-01-02T01:00:00Z",
        "last_seen_posted_at": "2026-01-01T00:00:00Z",
        "freehire_compat_status": "pending",
    }])
    first_client = Client()
    first = backfill_freehire_compat.run(apply=True, db=db, client=first_client)
    assert first["classified"] == 1
    assert first_client.calls == 1
    assert db.rows[0]["job_id"] == "1"
    assert db.rows[0]["latest_job_id"] == "live"

    second_client = Client()
    before_updates = db.updates
    second = backfill_freehire_compat.run(apply=True, db=db, client=second_client)
    assert second["unchanged"] == 1
    assert second_client.calls == 0
    assert db.updates == before_updates


def test_latest_id_only_drift_updates_import_hash_without_llm():
    row = {
        "job_id": "1",
        "latest_job_id": "old",
        "provider": "linkedin",
        "job_title": "TPM",
        "location": "Toronto",
        "description": "Delivery",
        "level": "Senior",
        "freehire_category": "project_management",
        "freehire_seniority": "senior",
        "is_remote": False,
        "freehire_remote_evidence": None,
        "freehire_compat_status": "current",
        "freehire_compat_model": "fake/model",
        "freehire_compat_prompt_version": "freehire-category-v1",
        "freehire_compat_schema_version": "freehire-compat-v1",
    }
    row["freehire_compat_input_hash"] = freehire_compat.compute_classification_hash(row)
    row["freehire_compat_import_hash"] = freehire_compat.compute_import_hash(row)
    row["latest_job_id"] = "new"
    db = Db([row])
    client = Client()
    result = backfill_freehire_compat.run(apply=True, db=db, client=client)
    assert result["metadata_updated"] == 1
    assert client.calls == 0
    assert row["freehire_compat_import_hash"] == freehire_compat.compute_import_hash(row)


def test_dry_run_never_calls_llm_or_updates():
    db = Db([{
        "job_id": "1",
        "provider": "linkedin",
        "job_title": "TPM",
        "description": "Delivery",
        "freehire_compat_status": "pending",
    }])
    client = Client()
    result = backfill_freehire_compat.run(apply=False, db=db, client=client)
    assert result["would_classify"] == 1
    assert result["llm_requests"] == 0
    assert client.calls == 0
    assert db.updates == 0


def test_frontfill_query_is_bounded_eligible_and_newest_first():
    db = Db([])

    backfill_freehire_compat.fetch_frontfill_candidates(db, 300)

    assert db.orders == [("last_seen_at", True), ("job_id", True)]
    assert len(db.or_filters) == 1
    eligibility = db.or_filters[0]
    assert "freehire_compat_status.eq.pending" in eligibility
    assert "freehire_compat_status.eq.failed" in eligibility
    assert "freehire_compat_next_retry_at.lte." in eligibility
    assert "freehire_compat_status.eq.processing" in eligibility
    assert "freehire_compat_claimed_at.lt." in eligibility
    assert "freehire_compat_status.eq.current" in eligibility


def test_non_drain_hard_caps_at_300_without_complete_keyset_scan(monkeypatch):
    rows = [
        {
            "job_id": str(index),
            "provider": "linkedin",
            "job_title": "TPM",
            "description": "Delivery",
            "freehire_compat_status": "pending",
        }
        for index in range(400)
    ]
    monkeypatch.setattr(
        backfill_freehire_compat,
        "fetch_frontfill_candidates",
        lambda _db, limit: rows[:limit],
    )
    monkeypatch.setattr(
        backfill_freehire_compat,
        "fetch_candidates",
        lambda *_args, **_kwargs: pytest.fail("non-drain must not run a complete keyset scan"),
    )

    result = backfill_freehire_compat.run(apply=False, limit=300, db=Db(rows))

    assert result["scanned"] == 300
    assert result["would_classify"] == 300


def test_hourly_frontfill_rejects_limit_above_300(monkeypatch):
    monkeypatch.setenv("FREEHIRE_CLASSIFY_LIMIT", "301")
    with pytest.raises(ValueError, match="hourly hard cap of 300"):
        frontfill_freehire_compat.classify_limit_from_env()


def test_capped_replacement_requires_cutoff_and_resumes_by_classified_timestamp():
    assert backfill_freehire_compat._replacement_cutoff("") is None
    with pytest.raises(ValueError, match="requires replacement_before"):
        backfill_freehire_compat.run(
            apply=False,
            limit=1000,
            replacement_backfill=True,
            db=Db([]),
        )

    row = {
        "job_id": "1",
        "provider": "linkedin",
        "job_title": "TPM",
        "description": "Delivery",
        "freehire_category": "project_management",
        "freehire_seniority": "senior",
        "freehire_compat_status": "current",
        "freehire_compat_model": "model",
        "freehire_compat_prompt_version": "freehire-category-v1",
        "freehire_compat_schema_version": "freehire-compat-v1",
        "freehire_compat_classified_at": "2025-01-01T00:00:00+00:00",
    }
    row["freehire_compat_input_hash"] = freehire_compat.compute_classification_hash(row)
    result = backfill_freehire_compat.run(
        apply=False,
        limit=1000,
        replacement_backfill=True,
        replacement_before="2025-02-01T00:00:00+00:00",
        db=Db([row]),
    )
    assert result["would_classify"] == 1

    row["freehire_compat_classified_at"] = "2025-02-01T00:00:00+00:00"
    result = backfill_freehire_compat.run(
        apply=False,
        limit=1000,
        replacement_backfill=True,
        replacement_before="2025-02-01T00:00:00+00:00",
        db=Db([row]),
    )
    assert result["would_classify"] == 0
