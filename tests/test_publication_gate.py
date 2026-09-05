from types import SimpleNamespace

import pytest

import publication_gate


class Query:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.filters = []
        self.is_count = False

    @property
    def not_(self):
        return self

    def select(self, fields, count=None):
        self.fields = fields
        self.is_count = count == "exact"
        return self

    def eq(self, field, value):
        self.filters.append((field, value))
        return self

    def is_(self, field, value):
        self.filters.append((f"not_is_{field}", value))
        return self

    def limit(self, value):
        assert value == 1
        return self

    def execute(self):
        self.db.calls.append((self.table, self.fields, tuple(self.filters), self.is_count))
        if self.table == "scrape_run_state":
            return SimpleNamespace(data=[{"last_successful_scrape_at": "2025-01-02T03:04:05+00:00"}])
        key = (self.table, tuple(self.filters))
        return SimpleNamespace(count=self.db.counts[key], data=[])


class Db:
    def __init__(self):
        self.calls = []
        self.counts = {
            ("jobs", (("provider", "linkedin"),)): 516,
            ("jobs", (("provider", "linkedin"), ("freehire_compat_status", "current"))): 12,
            ("jobs", (("provider", "linkedin"), ("freehire_compat_status", "pending"))): 500,
            ("jobs", (("provider", "linkedin"), ("freehire_compat_status", "failed"))): 4,
            ("jobs", (("provider", "linkedin"), ("freehire_compat_status", "processing"))): 0,
            ("jobs", (("provider", "linkedin"), ("not_is_freehire_compat_classified_at", None))): 12,
            ("freehire_jobs", ()): 10,
        }

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        self.rpc_calls = getattr(self, "rpc_calls", []) + [(name, params)]
        if name == "prune_freehire_publication_generations":
            assert params == {"p_keep_generations": 3, "p_max_generations": 3}
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=1))
        assert name == "finalize_freehire_publication"
        assert params == {"p_source_scrape_watermark": "2025-01-02T03:04:05+00:00"}
        return SimpleNamespace(
            execute=lambda: SimpleNamespace(data=[{
                "generation": 7,
                "published_at": "2025-01-02T03:10:00+00:00",
                "source_scrape_watermark": "2025-01-02T03:04:05+00:00",
                "row_count": 10,
                "schema_version": "freehire-publication-v1",
            }])
        )


def test_verify_pipeline_results_requires_every_dependency_success():
    publication_gate.verify_pipeline_results(
        {"scrape": "success", "freehire_compat": "success"}
    )

    with pytest.raises(RuntimeError, match="freehire_compat=failure"):
        publication_gate.verify_pipeline_results(
            {"scrape": "success", "freehire_compat": "failure"}
        )


def test_publication_pruning_is_bounded():
    db = Db()
    assert publication_gate.prune_publication_generations(db) == 1


def test_publication_pruning_failure_does_not_undo_publication(capsys):
    class FailedPruneDb:
        def rpc(self, _name, _params):
            return SimpleNamespace(
                execute=lambda: (_ for _ in ()).throw(RuntimeError("timeout"))
            )

    assert publication_gate.try_prune_publication_generations(FailedPruneDb()) == 0
    assert "Publication pruning deferred: timeout" in capsys.readouterr().out


def test_publication_state_reports_backlog_without_blocking_published_rows():
    state = publication_gate.query_publication_state(Db())

    assert state == {
        "total": 516,
        "current": 12,
        "pending": 500,
        "failed": 4,
        "processing": 0,
        "prior_publication": 12,
        "published": 10,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
        "stale": 2,
    }


def test_write_github_outputs_includes_optional_empty_watermark(tmp_path):
    output = tmp_path / "github-output"
    publication_gate.write_github_outputs(
        {
            "total": 7,
            "current": 1,
            "pending": 2,
            "failed": 3,
            "processing": 1,
            "prior_publication": 1,
            "published": 1,
            "scrape_watermark": None,
            "stale": 0,
        },
        str(output),
    )

    assert output.read_text().splitlines() == [
        "total=7",
        "current=1",
        "pending=2",
        "failed=3",
        "processing=1",
        "prior_publication=1",
        "published=1",
        "scrape_watermark=",
        "stale=0",
    ]


def test_validation_blocks_impossible_or_missing_publication_state():
    valid = {
        "current": 12,
        "published": 10,
        "prior_publication": 12,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
    }
    publication_gate.validate_publication_state(valid)

    with pytest.raises(RuntimeError, match="exceeds current"):
        publication_gate.validate_publication_state({**valid, "published": 13})
    with pytest.raises(RuntimeError, match="watermark is absent"):
        publication_gate.validate_publication_state({**valid, "scrape_watermark": None})
    with pytest.raises(RuntimeError, match="zero published"):
        publication_gate.validate_publication_state({**valid, "published": 0})


def test_validation_does_not_block_historical_pending_backlog():
    publication_gate.validate_publication_state({
        "current": 12,
        "pending": 5000,
        "published": 10,
        "prior_publication": 12,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
    })


def test_finalize_publication_verifies_and_returns_snapshot_metadata():
    db = Db()
    publication = publication_gate.finalize_publication(db, {
        "published": 10,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
    })

    assert publication == {
        "generation": 7,
        "published_at": "2025-01-02T03:10:00+00:00",
        "source_scrape_watermark": "2025-01-02T03:04:05+00:00",
        "row_count": 10,
        "schema_version": "freehire-publication-v1",
    }


def test_repeated_finalize_with_same_watermark_returns_same_generation():
    db = Db()
    state = {
        "published": 10,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
    }

    first = publication_gate.finalize_publication(db, state)
    repeated = publication_gate.finalize_publication(db, state)

    assert first == repeated
    assert first["generation"] == 7
    assert len(db.rpc_calls) == 2


def test_finalize_publication_uses_transactional_rpc_count():
    db = Db()
    publication = publication_gate.finalize_publication(db, {
        "published": 11,
        "scrape_watermark": "2025-01-02T03:04:05+00:00",
    })

    assert publication["row_count"] == 10


def test_cycle_publication_uses_queue_barrier_rpc():
    class CycleDb(Db):
        def rpc(self, name, params):
            self.rpc_calls = getattr(self, "rpc_calls", []) + [(name, params)]
            assert name == "finalize_freehire_publication_v2"
            assert params == {"p_cycle_id": 12}
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(data={
                    "outcome": "published",
                    "reason": None,
                    "requested_cycle_id": 12,
                    "eligible_cycle_id": 12,
                    "generation": 8,
                    "published_at": "2025-01-02T03:10:00+00:00",
                    "source_scrape_watermark": "2025-01-02T03:04:05+00:00",
                    "source_discovery_sequence": 10,
                    "row_count": 10,
                    "schema_version": "freehire-publication-v1",
                })
            )

    publication = publication_gate.finalize_publication(
        CycleDb(),
        {"scrape_watermark": "2025-01-02T03:04:05+00:00"},
        discovery_cycle_id=12,
    )

    assert publication["outcome"] == "published"
    assert publication["requested_cycle_id"] == 12
    assert publication["generation"] == 8


def test_cycle_publication_deferral_is_a_successful_typed_result():
    class DeferredDb:
        def rpc(self, name, params):
            assert name == "finalize_freehire_publication_v2"
            assert params == {"p_cycle_id": 12}
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(data={
                    "outcome": "deferred",
                    "reason": "unresolved discovery tasks",
                    "requested_cycle_id": 12,
                    "eligible_cycle_id": None,
                    "blocking_count": 3,
                })
            )

    publication = publication_gate.finalize_publication(
        DeferredDb(),
        {"scrape_watermark": "2025-01-02T03:04:05+00:00"},
        discovery_cycle_id=12,
    )

    assert publication["outcome"] == "deferred"
    assert publication["blocking_count"] == 3
    assert publication["generation"] is None


def test_cycle_publication_deferral_allows_initial_empty_state():
    class DeferredDb:
        def rpc(self, name, params):
            assert name == "finalize_freehire_publication_v2"
            assert params == {"p_cycle_id": 12}
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(data={
                    "outcome": "deferred",
                    "reason": "unresolved discovery tasks",
                    "requested_cycle_id": 12,
                    "eligible_cycle_id": None,
                    "blocking_count": 1,
                })
            )

    state = {"current": 0, "published": 0, "scrape_watermark": None}
    publication_gate.validate_publication_state(state, require_legacy_ready=False)
    publication = publication_gate.finalize_publication(
        DeferredDb(), state, discovery_cycle_id=12
    )

    assert publication["outcome"] == "deferred"
    assert publication["source_scrape_watermark"] is None
