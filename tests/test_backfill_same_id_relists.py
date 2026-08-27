import backfill_same_id_relists as backfill


class Response:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, rows_by_table, table, updates, in_queries):
        self.rows_by_table = rows_by_table
        self.table = table
        self.updates = updates
        self.in_queries = in_queries
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
        self.in_queries.append((self.table, field, list(values)))
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
    def __init__(self, rows_by_table, rpc_result=True):
        self.rows_by_table = rows_by_table
        self.updates = []
        self.in_queries = []
        self.rpc_calls = []
        self.rpc_result = rpc_result

    def table(self, table):
        return Query(self.rows_by_table, table, self.updates, self.in_queries)

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        result = self.rpc_result

        class Rpc:
            def execute(self):
                return Response(result)

        return Rpc()


def test_selected_id_chunks_deduplicate_without_reordering():
    assert backfill._selected_id_chunks(["canonical-b", "canonical-a", "canonical-b", "canonical-c"]) == [
        ["canonical-b", "canonical-a", "canonical-c"]
    ]


def test_fetch_all_chunks_1000_selected_ids_and_paginates_each_chunk(monkeypatch):
    selected_ids = [f"canonical-{index:04d}" for index in range(1000)]
    rows = {
        "listing_observations": [
            {"canonical_job_id": canonical_id, "sequence": index}
            for index, canonical_id in enumerate(selected_ids)
        ]
    }
    client = Client(rows)
    monkeypatch.setattr(backfill.supabase_utils, "supabase", client)

    result = backfill.fetch_all(
        "listing_observations",
        "canonical_job_id,sequence",
        None,
        page_size=73,
        selected_ids=selected_ids,
    )

    assert [row["sequence"] for row in result] == list(range(1000))
    assert len(client.in_queries) == 20
    assert all(
        len(query_ids) <= backfill.SELECTED_IDS_QUERY_CHUNK_SIZE
        for _table, _field, query_ids in client.in_queries
    )
    assert {
        query_id
        for _table, _field, query_ids in client.in_queries
        for query_id in query_ids
    } == set(selected_ids)


def test_fetch_all_applies_limit_across_selected_id_chunks(monkeypatch):
    selected_ids = [f"canonical-{index:04d}" for index in range(1000)]
    rows = {
        "listing_observations": [
            {"canonical_job_id": canonical_id, "sequence": index}
            for index, canonical_id in enumerate(selected_ids)
        ]
    }
    client = Client(rows)
    monkeypatch.setattr(backfill.supabase_utils, "supabase", client)

    result = backfill.fetch_all(
        "listing_observations",
        "canonical_job_id,sequence",
        157,
        page_size=73,
        selected_ids=selected_ids,
    )

    assert [row["sequence"] for row in result] == list(range(157))
    assert len(client.in_queries) == 3


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
    assert applied["applied"] == 1
    assert applied["conflicts"] == 0
    assert client.updates == []
    assert len(client.rpc_calls) == 1
    rpc_name, rpc_params = client.rpc_calls[0]
    assert rpc_name == "apply_same_id_relist_repair"
    assert rpc_params["p_canonical_job_id"] == "canonical"
    assert rpc_params["p_expected_listing_instances"] == [
        {"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}
    ]
    assert rpc_params["p_expected_last_seen_at"] == "2026-08-03T10:00:00Z"
    assert rpc_params["p_payload"]["same_id_relist_count"] == 1


def test_run_counts_rpc_cas_conflicts(monkeypatch):
    rows = {
        "jobs": [{
            "job_id": "canonical",
            "location": "Toronto",
            "last_seen_at": None,
            "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
            "seen_count": 1,
            "posting_wave_count": 1,
            "repost_count": 0,
            "same_id_relist_count": 0,
        }],
        "listing_observations": [
            {"provider": "linkedin", "source_job_id": "source-1", "canonical_job_id": "canonical", "posted_at": "2026-08-01", "observed_at": "2026-08-01T10:00:00Z", "ingestion_run_id": "a", "result": "seen"},
            {"provider": "linkedin", "source_job_id": "source-1", "canonical_job_id": "canonical", "posted_at": "2026-08-03", "observed_at": "2026-08-03T10:00:00Z", "ingestion_run_id": "b", "result": "seen"},
        ],
    }
    client = Client(rows, rpc_result=False)
    monkeypatch.setattr(backfill.supabase_utils, "supabase", client)

    result = backfill.run(limit=10, apply=True)

    assert result["applied"] == 0
    assert result["conflicts"] == 1
    assert client.rpc_calls[0][1]["p_expected_last_seen_at"] is None
