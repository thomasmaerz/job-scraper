import logging
from types import SimpleNamespace

import scraper
import supabase_utils
from scrape_configuration import parse_scrape_configuration
from test_scrape_configuration import rpc_payload


def test_get_last_successful_scrape_at_reads_existing_run_state(monkeypatch):
    calls = []

    class FakeQuery:
        def select(self, columns):
            calls.append(("select", columns))
            return self

        def eq(self, column, value):
            calls.append(("eq", column, value))
            return self

        def limit(self, value):
            calls.append(("limit", value))
            return self

        def execute(self):
            return SimpleNamespace(data=[{
                "last_successful_scrape_at": "2026-08-26T10:32:33+00:00"
            }])

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    assert supabase_utils.get_last_successful_scrape_at() == "2026-08-26T10:32:33+00:00"
    assert calls == [
        ("select", "last_successful_scrape_at"),
        ("eq", "id", 1),
        ("limit", 1),
    ]


def test_record_scrape_success_updates_existing_run_state(monkeypatch):
    calls = []
    state = {}

    class FakeQuery:
        operation = None

        def update(self, payload):
            self.operation = "update"
            self.payload = payload
            calls.append(("update", payload))
            return self

        def eq(self, column, value):
            calls.append(("eq", column, value))
            return self

        def select(self, columns):
            calls.append(("select", columns))
            if self.operation is None:
                self.operation = "select"
            return self

        def limit(self, value):
            calls.append(("limit", value))
            return self

        def execute(self):
            if self.operation == "update":
                state.update(self.payload)
                return SimpleNamespace(data=[])
            return SimpleNamespace(data=[state.copy()])

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    assert supabase_utils.record_scrape_success() is True
    assert calls[0][0] == "update"
    assert calls[0][1]["last_successful_scrape_at"].endswith("+00:00")
    assert calls[1:] == [
        ("eq", "id", 1),
        ("select", "last_successful_scrape_at"),
        ("eq", "id", 1),
        ("limit", 1),
    ]


def test_record_scrape_success_executes_update_builder_without_select_and_reads_back(monkeypatch):
    calls = []
    state = {"id": 1, "last_successful_scrape_at": "2026-08-26T10:32:33+00:00"}

    class MutationBuilder:
        def __init__(self, payload):
            self.payload = payload

        def eq(self, column, value):
            calls.append(("update_eq", column, value))
            return self

        def execute(self):
            calls.append(("update_execute",))
            state.update(self.payload)
            return SimpleNamespace(data=[])

    class ReadBuilder:
        def select(self, columns):
            calls.append(("select", columns))
            return self

        def eq(self, column, value):
            calls.append(("select_eq", column, value))
            return self

        def limit(self, value):
            calls.append(("limit", value))
            return self

        def execute(self):
            calls.append(("select_execute",))
            return SimpleNamespace(data=[state.copy()])

    class FakeTable:
        def update(self, payload):
            calls.append(("update", payload))
            return MutationBuilder(payload)

        def select(self, columns):
            return ReadBuilder().select(columns)

        def upsert(self, _payload, **_kwargs):
            raise AssertionError("upsert fallback should not run after verified update")

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeTable()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    assert supabase_utils.record_scrape_success() is True
    assert calls[0][0] == "update"
    assert calls[0][1]["last_successful_scrape_at"].endswith("+00:00")
    assert calls[1:] == [
        ("update_eq", "id", 1),
        ("update_execute",),
        ("select", "last_successful_scrape_at"),
        ("select_eq", "id", 1),
        ("limit", 1),
        ("select_execute",),
    ]


def test_record_scrape_success_upserts_missing_singleton_row(monkeypatch):
    writes = []
    state = {}

    class FakeQuery:
        operation = None

        def update(self, payload):
            self.operation = "update"
            writes.append(("update", payload, {}))
            return self

        def eq(self, _column, _value):
            return self

        def select(self, _columns):
            if self.operation is None:
                self.operation = "select"
            return self

        def limit(self, _value):
            return self

        def upsert(self, payload, **kwargs):
            self.operation = "upsert"
            self.payload = payload
            writes.append(("upsert", payload, kwargs))
            return self

        def execute(self):
            if self.operation == "update":
                return SimpleNamespace(data=[])
            if self.operation == "upsert":
                state.update(self.payload)
                return SimpleNamespace(data=[])
            return SimpleNamespace(data=[state.copy()] if state else [])

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    assert supabase_utils.record_scrape_success() is True

    operation, payload, options = writes[1]
    assert operation == "upsert"
    assert payload["id"] == 1
    assert payload["last_successful_scrape_at"].endswith("+00:00")
    assert options == {"on_conflict": "id"}


def test_record_scrape_success_returns_false_when_run_state_write_fails(monkeypatch, caplog):
    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.ERROR):
        assert supabase_utils.record_scrape_success() is False

    assert "Failed to persist scrape success watermark: database unavailable" in caplog.text


def test_record_scrape_success_returns_false_when_read_back_mismatches(monkeypatch, caplog):
    class FakeQuery:
        operation = None

        def update(self, _payload):
            self.operation = "update"
            return self

        def eq(self, _column, _value):
            return self

        def select(self, _columns):
            if self.operation is None:
                self.operation = "select"
            return self

        def limit(self, _value):
            return self

        def upsert(self, _payload, **_kwargs):
            self.operation = "upsert"
            return self

        def execute(self):
            if self.operation == "select":
                return SimpleNamespace(data=[{
                    "last_successful_scrape_at": "2026-08-26T10:32:33+00:00"
                }])
            return SimpleNamespace(data=[])

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.ERROR):
        assert supabase_utils.record_scrape_success() is False

    assert "Scrape success watermark did not match after update and upsert" in caplog.text


def test_record_scrape_success_returns_false_when_read_back_fails(monkeypatch, caplog):
    writes = []

    class FakeQuery:
        operation = None

        def update(self, _payload):
            self.operation = "update"
            writes.append("update")
            return self

        def eq(self, _column, _value):
            return self

        def select(self, _columns):
            if self.operation is None:
                self.operation = "select"
            return self

        def limit(self, _value):
            return self

        def upsert(self, _payload, **_kwargs):
            self.operation = "upsert"
            writes.append("upsert")
            return self

        def execute(self):
            if self.operation == "select":
                raise RuntimeError("read-back unavailable")
            return SimpleNamespace(data=[])

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.WARNING):
        assert supabase_utils.record_scrape_success() is False

    assert writes == ["update", "upsert"]
    assert caplog.text.count("Failed to verify scrape success watermark") == 2


def test_record_scrape_success_calls_rpc_without_run_id_and_verifies_read_back(monkeypatch):
    calls = []
    state = {}

    class RpcQuery:
        def __init__(self, params):
            self.params = params

        def execute(self):
            state["last_successful_scrape_at"] = self.params["p_finished_at"]
            return SimpleNamespace(data=self.params["p_finished_at"])

    class ReadQuery:
        def select(self, columns):
            calls.append(("select", columns))
            return self

        def eq(self, column, value):
            calls.append(("eq", column, value))
            return self

        def limit(self, value):
            calls.append(("limit", value))
            return self

        def execute(self):
            return SimpleNamespace(data=[state.copy()])

    class FakeSupabase:
        def rpc(self, name, params):
            calls.append(("rpc", name, params))
            return RpcQuery(params)

        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return ReadQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    assert supabase_utils.record_scrape_success() is True
    assert calls[0][0:2] == ("rpc", "record_scrape_success")
    assert set(calls[0][2]) == {"p_finished_at"}
    assert calls[0][2]["p_finished_at"].endswith("+00:00")
    assert calls[1:] == [
        ("select", "last_successful_scrape_at"),
        ("eq", "id", 1),
        ("limit", 1),
    ]


def test_record_scrape_success_rejects_unexpected_rpc_return(monkeypatch, caplog):
    class FakeSupabase:
        def rpc(self, _name, _params):
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=None))

        def table(self, _table_name):
            raise AssertionError("read-back must not hide an invalid RPC result")

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.ERROR):
        assert supabase_utils.record_scrape_success() is False

    assert "RPC returned an unexpected timestamp" in caplog.text


def test_record_scrape_success_rejects_mismatched_rpc_read_back(monkeypatch, caplog):
    class ReadQuery:
        def select(self, _columns):
            return self

        def eq(self, _column, _value):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            return SimpleNamespace(data=[{
                "last_successful_scrape_at": "2026-08-26T10:32:33+00:00"
            }])

    class FakeSupabase:
        def rpc(self, _name, params):
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(data=params["p_finished_at"])
            )

        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return ReadQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.ERROR):
        assert supabase_utils.record_scrape_success() is False

    assert "Scrape success watermark did not match after RPC write" in caplog.text


def test_record_scrape_success_does_not_fallback_for_rpc_execution_failure(monkeypatch, caplog):
    class FakeSupabase:
        def rpc(self, _name, _params):
            raise RuntimeError("database unavailable")

        def table(self, _table_name):
            raise AssertionError("direct writes are unsafe when the RPC exists but fails")

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.ERROR):
        assert supabase_utils.record_scrape_success() is False

    assert "Failed to persist scrape success watermark via RPC: database unavailable" in caplog.text


def test_record_scrape_success_falls_back_when_rpc_is_absent(monkeypatch, caplog):
    state = {}
    writes = []

    class MissingRpcError(Exception):
        code = "PGRST202"

    class FakeQuery:
        operation = None

        def update(self, payload):
            self.operation = "update"
            self.payload = payload
            writes.append("update")
            return self

        def upsert(self, _payload, **_kwargs):
            raise AssertionError("verified update must not upsert")

        def select(self, _columns):
            self.operation = "select"
            return self

        def eq(self, _column, _value):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            if self.operation == "update":
                state.update(self.payload)
                return SimpleNamespace(data=[])
            return SimpleNamespace(data=[state.copy()])

    class FakeSupabase:
        def rpc(self, _name, _params):
            raise MissingRpcError("function absent from schema cache")

        def table(self, table_name):
            assert table_name == "scrape_run_state"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    with caplog.at_level(logging.WARNING):
        assert supabase_utils.record_scrape_success() is True

    assert writes == ["update"]
    assert "RPC is absent; using legacy direct watermark write" in caplog.text


def test_main_consumes_canonical_saver_id_lists_for_all_sources(monkeypatch, caplog):
    query_run_id = "c76e1632-1a6a-42b5-8f2c-da3949988dc5"
    linkedin_jobs = [{
        "job_id": "linkedin-source-id",
        "provider": "linkedin",
        "scrape_run_id": query_run_id,
    }]
    careers_future_jobs = [
        {"job_id": "careers-future-source-id", "provider": "careers_future"}
    ]
    linkedin_save_calls = []
    linkedin_process_calls = []
    generic_save_calls = []
    success_calls = []

    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", {"linkedin", "careers_future"})
    monkeypatch.setattr(scraper.config, "LINKEDIN_LAST_SUCCESS_AT", None)
    monkeypatch.setattr(
        scraper.config,
        "ARCHETYPE_CONFIGS",
        {
            "software_tpm": {
                "provider": "linkedin",
                "location": "Canada",
                "filter_profile": "software_tpm_v1",
                "search_queries": ["Technical Program Manager"],
            }
        },
    )
    monkeypatch.setattr(
        scraper.config,
        "MAX_JOBS_PER_SEARCH",
        {"linkedin": 10, "careers_future": 10},
    )
    monkeypatch.setattr(
        scraper.config,
        "CAREERS_FUTURE_SEARCH_QUERIES",
        ["Program Manager"],
    )
    monkeypatch.setattr(
        scraper,
        "load_scrape_configuration",
        lambda db: parse_scrape_configuration(rpc_payload()),
    )
    monkeypatch.setattr(
        scraper,
        "process_linkedin_query",
        lambda **kwargs: (
            linkedin_process_calls.append(kwargs)
            or (linkedin_jobs if len(linkedin_process_calls) == 1 else [])
        ),
    )
    monkeypatch.setattr(
        scraper,
        "process_careers_future_query",
        lambda query, limit=None: careers_future_jobs,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_linkedin_jobs_canonicalized_with_mapping",
        lambda jobs, run_context=None: linkedin_save_calls.append(jobs) or scraper.supabase_utils.CanonicalSaveResult(
            canonical_ids=["linkedin-canonical-id"],
            canonical_by_source={"linkedin-id": "linkedin-canonical-id"},
            canonical_ids_by_input=["linkedin-canonical-id"],
        ),
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "persist_lane_filter_state",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_jobs_canonicalized",
        lambda jobs: generic_save_calls.append(jobs) or ["careers-future-canonical-id"],
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "get_last_successful_scrape_at",
        lambda: None,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "record_scrape_success",
        lambda: success_calls.append(True) or True,
    )

    with caplog.at_level(logging.INFO):
        saved_job_ids = scraper.main()

    assert saved_job_ids == [
        "linkedin-canonical-id",
        "careers-future-canonical-id",
    ]
    assert linkedin_save_calls == [linkedin_jobs]
    assert generic_save_calls == [careers_future_jobs]
    assert success_calls == [True]
    assert "Total new jobs saved across all queries: 2" in caplog.text


def test_main_uses_careers_future_saver_ids_without_renormalizing(monkeypatch):
    careers_future_jobs = [
        {"job_id": "careers-future-source-id", "provider": "careers_future"}
    ]
    canonical_ids = ["careers-future-canonical-id", "existing-canonical-id"]

    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", {"careers_future"})
    monkeypatch.setattr(
        scraper.config,
        "MAX_JOBS_PER_SEARCH",
        {"careers_future": 10},
    )
    monkeypatch.setattr(
        scraper.config,
        "CAREERS_FUTURE_SEARCH_QUERIES",
        ["Program Manager"],
    )
    monkeypatch.setattr(
        scraper,
        "process_careers_future_query",
        lambda query, limit=None: careers_future_jobs,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_jobs_canonicalized",
        lambda jobs: canonical_ids,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "record_scrape_success",
        lambda: True,
    )

    def fail_if_renormalized(value):
        raise AssertionError(f"main must not renormalize saver-returned IDs: {value!r}")

    monkeypatch.setattr(
        scraper.supabase_utils,
        "normalize_job_identifier",
        fail_if_renormalized,
    )

    assert scraper.main() == canonical_ids


def test_main_completes_after_successful_watermark_update(monkeypatch):
    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", set())
    monkeypatch.setattr(scraper.supabase_utils, "record_scrape_success", lambda: True)

    assert scraper.main() == []


def test_main_fails_when_required_watermark_cannot_be_persisted(monkeypatch):
    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", set())
    monkeypatch.setattr(scraper.supabase_utils, "record_scrape_success", lambda: False)

    try:
        scraper.main()
    except RuntimeError as error:
        assert str(error) == "Failed to persist scrape success watermark"
    else:
        raise AssertionError("main() should require watermark persistence")


def test_main_does_not_advance_global_watermark_for_single_lane_recovery(monkeypatch):
    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", {"linkedin"})
    monkeypatch.setattr(scraper, "load_scrape_configuration", lambda **_kwargs: object())
    monkeypatch.setattr(
        scraper,
        "_run_database_configured_linkedin",
        lambda _configuration, _saved_ids: False,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "record_scrape_success",
        lambda: (_ for _ in ()).throw(AssertionError("partial run advanced watermark")),
    )

    assert scraper.main() == []
