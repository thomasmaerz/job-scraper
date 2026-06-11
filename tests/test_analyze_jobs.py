import json
from types import SimpleNamespace

import analyze_jobs


def test_aggregate_keywords_normalizes_keyword_case_and_category():
    items = [
        analyze_jobs.KeywordItem(keyword="python", category="technology"),
        analyze_jobs.KeywordItem(keyword=" Python ", category="Technology"),
        analyze_jobs.KeywordItem(keyword="PMP", category="certification"),
    ]

    counts = analyze_jobs.aggregate_keywords(items)

    assert counts == {
        ("Python", "technology"): 2,
        ("Pmp", "certification"): 1,
    }


def test_aggregate_keywords_ignores_invalid_categories():
    items = [
        analyze_jobs.KeywordItem(keyword="Python", category="technology"),
        analyze_jobs.KeywordItem(keyword="nonsense", category="invalid"),
    ]

    counts = analyze_jobs.aggregate_keywords(items)

    assert counts == {("Python", "technology"): 1}


def test_parse_keyword_response_validates_structured_json():
    raw = json.dumps(
        {
            "keywords": [
                {"keyword": "Azure", "category": "technology"},
                {"keyword": "PMP", "category": "certification"},
            ]
        }
    )

    parsed = analyze_jobs.parse_keyword_response(raw)

    assert parsed == [
        analyze_jobs.KeywordItem(keyword="Azure", category="technology"),
        analyze_jobs.KeywordItem(keyword="PMP", category="certification"),
    ]


def test_extract_keywords_from_batch_uses_llm_client_response_format():
    calls = []

    class FakeClient:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return json.dumps(
                {
                    "keywords": [
                        {"keyword": "Python", "category": "technology"},
                        {"keyword": "Agile", "category": "skill"},
                    ]
                }
            )

    batch = [
        {
            "job_id": "1",
            "job_title": "Project Manager",
            "description": "Must know Agile and Python.",
        }
    ]

    result = analyze_jobs.extract_keywords_from_batch(batch, client=FakeClient())

    assert result == [
        analyze_jobs.KeywordItem(keyword="Python", category="technology"),
        analyze_jobs.KeywordItem(keyword="Agile", category="skill"),
    ]
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["response_format"] is analyze_jobs.KeywordList
    assert "Project Manager" in calls[0]["prompt"]
    assert "Must know Agile and Python." in calls[0]["prompt"]


def test_fetch_unanalyzed_jobs_queries_expected_filters():
    class FakeQuery:
        def __init__(self):
            self.calls = []

        def select(self, value):
            self.calls.append(("select", value))
            return self

        def eq(self, key, value):
            self.calls.append(("eq", key, value))
            return self

        def is_(self, key, value):
            self.calls.append(("is_", key, value))
            return self

        @property
        def not_(self):
            self.calls.append(("not_",))
            return self

        def limit(self, value):
            self.calls.append(("limit", value))
            return self

        def execute(self):
            self.calls.append(("execute",))
            return SimpleNamespace(data=[{"job_id": "1", "job_title": "A", "description": "B"}])

    class FakeDb:
        def __init__(self):
            self.query = FakeQuery()

        def table(self, name):
            assert name == "jobs"
            return self.query

    db = FakeDb()

    result = analyze_jobs.fetch_unanalyzed_jobs(db=db, limit=25)

    assert result == [{"job_id": "1", "job_title": "A", "description": "B"}]
    assert ("select", "job_id, job_title, description") in db.query.calls
    assert ("eq", "is_active", True) in db.query.calls
    assert ("eq", "job_state", "new") in db.query.calls
    assert ("is_", "insights_analyzed_at", None) in db.query.calls
    assert ("is_", "description", None) in db.query.calls
    assert ("limit", 25) in db.query.calls


def test_upsert_insights_merges_existing_counts_and_batches_rows():
    upserted = []

    class FakeSelectQuery:
        def __init__(self):
            self.offset = 0

        def select(self, value):
            assert value == "keyword, category, count"
            return self

        def range(self, start, end):
            self.offset = start
            return self

        def execute(self):
            if self.offset == 0:
                return SimpleNamespace(
                    data=[{"keyword": "Python", "category": "technology", "count": 4}]
                )
            return SimpleNamespace(data=[])

    class FakeUpsertQuery:
        def __init__(self, rows):
            self.rows = rows

        def execute(self):
            upserted.extend(self.rows)
            return SimpleNamespace(data=self.rows)

    class FakeTable:
        def select(self, value):
            return FakeSelectQuery().select(value)

        def upsert(self, rows, on_conflict):
            assert on_conflict == "keyword,category"
            return FakeUpsertQuery(rows)

    class FakeDb:
        def table(self, name):
            assert name == "keyword_insights"
            return FakeTable()

    analyze_jobs.upsert_insights(
        {("Python", "technology"): 3, ("Pmp", "certification"): 2},
        db=FakeDb(),
    )

    by_key = {(row["keyword"], row["category"]): row for row in upserted}
    assert by_key[("Python", "technology")]["count"] == 7
    assert by_key[("Pmp", "certification")]["count"] == 2
    assert all("last_updated" in row for row in upserted)


def test_mark_jobs_analyzed_updates_timestamp_for_ids():
    calls = []

    class FakeQuery:
        def update(self, payload):
            calls.append(("update", payload))
            return self

        def in_(self, key, values):
            calls.append(("in_", key, values))
            return self

        def execute(self):
            calls.append(("execute",))
            return SimpleNamespace(data=[])

    class FakeDb:
        def table(self, name):
            assert name == "jobs"
            return FakeQuery()

    analyze_jobs.mark_jobs_analyzed(["1", "2"], db=FakeDb())

    assert calls[0][0] == "update"
    assert "insights_analyzed_at" in calls[0][1]
    assert calls[1] == ("in_", "job_id", ["1", "2"])
    assert calls[2] == ("execute",)
