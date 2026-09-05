import asyncio

import job_manager


def test_mark_expired_jobs_uses_last_seen_at(monkeypatch):
    filters = []

    class FakeQuery:
        def select(self, _fields): return self
        def lt(self, field, value):
            filters.append((field, value))
            return self
        @property
        def not_(self): return self
        def in_(self, *_args): return self
        def eq(self, *_args): return self
        def execute(self): return type("Resp", (), {"data": []})()

    monkeypatch.setattr(job_manager, "supabase", type("Client", (), {"table": lambda *_: FakeQuery()})())

    asyncio.run(job_manager.mark_expired_jobs())

    assert any(field == "last_seen_at" for field, _value in filters)
    assert not any(field == "scraped_at" for field, _value in filters)


def test_check_linkedin_job_activity_uses_latest_job_id_for_fetch_and_canonical_job_id_for_updates(monkeypatch):
    selected = []
    checked_ids = []
    updated_batches = []

    class FakeQuery:
        def __init__(self, table_name):
            self.table_name = table_name
            self.payload = None

        def select(self, fields):
            selected.append(fields)
            return self

        def eq(self, *args, **kwargs):
            return self

        @property
        def not_(self):
            return self

        def in_(self, *args, **kwargs):
            return self

        def lt(self, *args, **kwargs):
            return self

        def order(self, *args, **kwargs):
            return self

        def limit(self, *args, **kwargs):
            return self

        def update(self, payload):
            self.payload = payload
            return self

        def execute(self):
            if self.payload is None:
                return type("Resp", (), {"data": [{
                    "job_id": "canonical-1",
                    "latest_job_id": "linkedin-live-99",
                    "last_checked": "2026-06-01T00:00:00+00:00",
                }]})()
            updated_batches.append(self.payload)
            return type("Resp", (), {"data": [{"job_id": "canonical-1"}]})()

    class FakeSupabase:
        def table(self, table_name):
            return FakeQuery(table_name)

    async def fake_check(job_id, client):
        checked_ids.append(job_id)
        return False

    monkeypatch.setattr(job_manager, "supabase", FakeSupabase())
    monkeypatch.setattr(job_manager, "_check_single_linkedin_job_active", fake_check)
    monkeypatch.setattr(job_manager.config, "JOB_CHECK_LIMIT", 10)

    asyncio.run(job_manager.check_linkedin_job_activity())

    assert any("latest_job_id" in fields for fields in selected)
    assert checked_ids == ["linkedin-live-99"]
    assert updated_batches, "Expected update call for canonical row"


def test_activity_check_consumes_durable_request_grant(monkeypatch):
    events = []

    class Gate:
        def acquire(self, kind, key):
            events.append(("acquire", kind, key))
            return "grant-1"

        def finish(self, grant, response_class, status):
            events.append(("finish", grant, response_class, status))

    class Client:
        async def get(self, *_args, **_kwargs):
            return type("Response", (), {
                "status_code": 200,
                "text": "active job",
                "url": "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/source-1",
            })()

    monkeypatch.setattr(job_manager, "_linkedin_request_gate", Gate())

    result = asyncio.run(
        job_manager._check_single_linkedin_job_active("source-1", Client())
    )

    assert result is False
    assert events == [
        ("acquire", "activity_check", "source-1:0"),
        ("finish", "grant-1", "complete", 200),
    ]
