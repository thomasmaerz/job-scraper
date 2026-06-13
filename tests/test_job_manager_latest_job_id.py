import asyncio

import job_manager


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
