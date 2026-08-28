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


def test_verify_pipeline_results_requires_every_dependency_success():
    publication_gate.verify_pipeline_results(
        {"scrape": "success", "freehire_compat": "success", "analyze_jobs": "success"}
    )

    with pytest.raises(RuntimeError, match="freehire_compat=failure"):
        publication_gate.verify_pipeline_results(
            {"scrape": "success", "freehire_compat": "failure", "analyze_jobs": "skipped"}
        )


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
