from types import SimpleNamespace

import supabase_utils


class Builder:
    def __init__(self, calls):
        self.calls = calls

    def upsert(self, payload, **kwargs):
        self.calls.append(("upsert", payload, kwargs))
        return self

    def update(self, payload, **kwargs):
        self.calls.append(("update", payload, kwargs))
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        return SimpleNamespace(data=[])


class Db:
    def __init__(self):
        self.calls = []

    def table(self, _name):
        return Builder(self.calls)


def test_ingestion_run_mutations_request_minimal_responses(monkeypatch):
    db = Db()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.start_ingestion_run("run-1", "linkedin")
    supabase_utils.finish_ingestion_run("run-1", status="complete")

    assert db.calls[0][2]["returning"] is supabase_utils.ReturnMethod.minimal
    assert db.calls[1][2]["returning"] is supabase_utils.ReturnMethod.minimal


def test_batched_listing_writes_request_minimal_responses(monkeypatch):
    db = Db()
    monkeypatch.setattr(supabase_utils, "supabase", db)
    card = {"job_id": "1", "posted_at": "2026-09-03"}

    supabase_utils.save_listing_observations([card], "run-1")
    supabase_utils.save_listing_states([card], {})

    assert db.calls[0][2]["returning"] is supabase_utils.ReturnMethod.minimal
    assert db.calls[1][2]["returning"] is supabase_utils.ReturnMethod.minimal
