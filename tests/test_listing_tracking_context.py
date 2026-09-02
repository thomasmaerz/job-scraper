from types import SimpleNamespace

import supabase_utils


class Query:
    def __init__(self, table_name, calls):
        self.table_name = table_name
        self.calls = calls

    def select(self, columns):
        self.calls.append((self.table_name, "select", columns))
        return self

    def eq(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def is_(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, *_args):
        return self

    def execute(self):
        return SimpleNamespace(data=[])


class Supabase:
    def __init__(self):
        self.calls = []

    def table(self, table_name):
        return Query(table_name, self.calls)


def test_run_index_resolves_tracking_context_without_jobs_snapshot(monkeypatch):
    client = Supabase()
    monkeypatch.setattr(supabase_utils, "supabase", client)

    context = supabase_utils.get_listing_tracking_context(
        "linkedin",
        ["known-source", "new-source"],
        canonical_by_source={"known-source": "canonical-1"},
    )

    assert context["known-source"]["canonical_job_id"] == "canonical-1"
    assert "new-source" not in context
    assert all(table_name != "jobs" for table_name, _, _ in client.calls)
    assert {table_name for table_name, _, _ in client.calls} == {
        "listing_states",
        "listing_observations",
        "listing_relist_events",
    }
