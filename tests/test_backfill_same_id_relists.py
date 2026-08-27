import backfill_same_id_relists as backfill


class Response:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, rows_by_table, table, updates):
        self.rows_by_table = rows_by_table
        self.table = table
        self.updates = updates
        self.start = 0
        self.end = 999
        self.payload = None

    def select(self, _fields):
        return self

    def range(self, start, end):
        self.start, self.end = start, end
        return self

    def update(self, payload):
        self.payload = payload
        self.updates.append(payload)
        return self

    def upsert(self, payload, **_kwargs):
        self.payload = payload
        self.updates.append(payload)
        return self

    def eq(self, *_args):
        return self

    def is_(self, *_args):
        return self

    def in_(self, field, values):
        self.rows_by_table = {
            **self.rows_by_table,
            self.table: [
                row for row in self.rows_by_table.get(self.table, [])
                if str(row.get(field)) in values
            ],
        }
        return self

    def execute(self):
        if self.payload is not None:
            return Response([self.payload])
        return Response(self.rows_by_table.get(self.table, [])[self.start:self.end + 1])


class Client:
    def __init__(self, rows_by_table):
        self.rows_by_table = rows_by_table
        self.updates = []

    def table(self, table):
        return Query(self.rows_by_table, table, self.updates)


def test_run_is_dry_run_by_default_and_apply_uses_guard(monkeypatch):
    rows = {
        "jobs": [{
            "job_id": "canonical",
            "location": "Toronto",
            "last_seen_at": "2026-08-03T10:00:00Z",
            "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
            "seen_count": 1,
            "posting_wave_count": 1,
            "repost_count": 0,
            "same_id_relist_count": 0,
        }],
        "listing_observations": [
            {"provider": "linkedin", "source_job_id": "source-1", "canonical_job_id": "canonical", "posted_at": "2026-08-01", "observed_at": "2026-08-01T10:00:00Z", "ingestion_run_id": "a", "result": "seen"},
            {"provider": "linkedin", "source_job_id": "source-1", "canonical_job_id": "canonical", "posted_at": "2026-08-01", "observed_at": "2026-08-02T10:00:00Z", "ingestion_run_id": "b", "result": "seen"},
            {"provider": "linkedin", "source_job_id": "source-1", "canonical_job_id": "canonical", "posted_at": "2026-08-03", "observed_at": "2026-08-03T10:00:00Z", "ingestion_run_id": "c", "result": "seen"},
        ],
    }
    client = Client(rows)
    monkeypatch.setattr(backfill.supabase_utils, "supabase", client)

    dry_run = backfill.run(limit=10)
    applied = backfill.run(limit=10, apply=True)

    assert dry_run["dry_run"] is True
    assert dry_run["changed"] == 1
    assert client.updates
    assert applied["applied"] == 1
