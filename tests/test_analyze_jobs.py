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
        ("PMP", "certification"): 1,
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
            "jobs": [
                {
                    "job_id": "job-1",
                    "keywords": [
                        {"keyword": "Azure", "category": "technology"},
                        {"keyword": "PMP", "category": "certification"},
                    ],
                }
            ]
        }
    )

    parsed = analyze_jobs.parse_keyword_response(raw)

    assert parsed == {
        "job-1": [
            analyze_jobs.KeywordItem(keyword="Azure", category="technology"),
            analyze_jobs.KeywordItem(keyword="PMP", category="certification"),
        ]
    }


def test_extract_keywords_from_batch_uses_llm_client_response_format():
    calls = []

    class FakeClient:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return json.dumps(
                {
                    "jobs": [
                        {
                            "job_id": "1",
                            "keywords": [
                                {"keyword": "Python", "category": "technology"},
                                {"keyword": "Agile", "category": "skill"},
                            ],
                        }
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

    assert result == {
        "1": [
            analyze_jobs.KeywordItem(keyword="Python", category="technology"),
            analyze_jobs.KeywordItem(keyword="Agile", category="skill"),
        ]
    }
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["response_format"] is analyze_jobs.JobKeywordResultList
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


def test_fetch_unanalyzed_jobs_for_backfill_omits_new_and_active_filters():
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
            return SimpleNamespace(data=[])

    class FakeDb:
        def __init__(self):
            self.query = FakeQuery()

        def table(self, name):
            assert name == "jobs"
            return self.query

    db = FakeDb()

    analyze_jobs.fetch_unanalyzed_jobs(db=db, limit=25, backfill_all=True)

    assert ("is_", "insights_analyzed_at", None) in db.query.calls
    assert ("is_", "description", None) in db.query.calls
    assert ("eq", "is_active", True) not in db.query.calls
    assert ("eq", "job_state", "new") not in db.query.calls


def test_extract_keywords_from_batch_raises_if_any_job_id_missing_from_response():
    class FakeClient:
        def generate_content(self, **kwargs):
            return json.dumps(
                {
                    "jobs": [
                        {
                            "job_id": "1",
                            "keywords": [
                                {"keyword": "Python", "category": "technology"},
                            ],
                        }
                    ]
                }
            )

    batch = [
        {"job_id": "1", "job_title": "A", "description": "Needs Python"},
        {"job_id": "2", "job_title": "B", "description": "Needs SQL"},
    ]

    try:
        analyze_jobs.extract_keywords_from_batch(batch, client=FakeClient(), max_retries=1)
    except ValueError as exc:
        assert "Missing keyword results for job_ids: 2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for omitted job_id")


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


def test_aggregate_keywords_preserves_acronyms_and_uppercase_keywords():
    items = [
        analyze_jobs.KeywordItem(keyword="AWS", category="technology"),
        analyze_jobs.KeywordItem(keyword="SQL", category="technology"),
        analyze_jobs.KeywordItem(keyword="PMP", category="certification"),
    ]

    counts = analyze_jobs.aggregate_keywords(items)

    assert counts == {
        ("AWS", "technology"): 1,
        ("SQL", "technology"): 1,
        ("PMP", "certification"): 1,
    }


def test_upsert_job_keyword_facts_ignores_existing_job_keyword_pairs():
    inserted_rows = []

    class FakeUpsertQuery:
        def __init__(self, rows):
            self.rows = rows

        def execute(self):
            inserted_rows.extend(self.rows[:1])
            return SimpleNamespace(data=self.rows[:1])

    class FakeTable:
        def upsert(self, rows, on_conflict, ignore_duplicates):
            assert on_conflict == "job_id,keyword,category"
            assert ignore_duplicates is True
            return FakeUpsertQuery(rows)

    class FakeDb:
        def table(self, name):
            assert name == "job_keyword_insights"
            return FakeTable()

    facts = [
        {"job_id": "1", "keyword": "AWS", "category": "technology"},
        {"job_id": "1", "keyword": "AWS", "category": "technology"},
    ]

    inserted = analyze_jobs.upsert_job_keyword_facts(facts, db=FakeDb())

    assert inserted == inserted_rows
    assert inserted == [{"job_id": "1", "keyword": "AWS", "category": "technology"}]


def test_update_keyword_insights_aggregates_existing_counts_plus_new_facts():
    upserted = []

    class FactsSelectQuery:
        def __init__(self):
            self.offset = 0

        def select(self, value):
            assert value == "job_id, keyword, category"
            return self

        def range(self, start, end):
            self.offset = start
            return self

        def execute(self):
            if self.offset == 0:
                return SimpleNamespace(
                    data=[
                        *[
                            {"job_id": str(i), "keyword": "AWS", "category": "technology"}
                            for i in range(1, 11)
                        ],
                        *[
                            {"job_id": f"p{i}", "keyword": "PMP", "category": "certification"}
                            for i in range(1, 5)
                        ],
                        {"job_id": "11", "keyword": "AWS", "category": "technology"},
                        {"job_id": "12", "keyword": "AWS", "category": "technology"},
                        {"job_id": "p5", "keyword": "PMP", "category": "certification"},
                    ]
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
            return FactsSelectQuery().select(value)

        def upsert(self, rows, on_conflict):
            assert on_conflict == "keyword,category"
            return FakeUpsertQuery(rows)

    class FakeDb:
        def table(self, name):
            if name == "job_keyword_insights":
                return FakeTable()
            if name == "keyword_insights":
                return FakeTable()
            raise AssertionError(name)

    source_facts = [
        {"job_id": "11", "keyword": "AWS", "category": "technology"},
        {"job_id": "12", "keyword": "AWS", "category": "technology"},
        {"job_id": "p5", "keyword": "PMP", "category": "certification"},
    ]

    analyze_jobs.update_keyword_insights_from_facts(source_facts, db=FakeDb())

    by_key = {(row["keyword"], row["category"]): row for row in upserted}
    assert by_key[("AWS", "technology")]["count"] == 12
    assert by_key[("PMP", "certification")]["count"] == 5
    assert all("last_updated" in row for row in upserted)


def test_update_keyword_insights_repairs_missing_aggregate_from_persisted_facts():
    upserted = []

    class KeywordInsightsSelectQuery:
        def __init__(self):
            self.offset = 0

        def select(self, value):
            assert value == "keyword, category, count"
            return self

        def range(self, start, end):
            self.offset = start
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class FactsSelectQuery:
        def __init__(self):
            self.offset = 0

        def select(self, value):
            assert value == "job_id, keyword, category"
            return self

        def range(self, start, end):
            self.offset = start
            return self

        def execute(self):
            if self.offset > 0:
                return SimpleNamespace(data=[])
            return SimpleNamespace(
                data=[
                    {"job_id": "1", "keyword": "AWS", "category": "technology"},
                    {"job_id": "2", "keyword": "AWS", "category": "technology"},
                    {"job_id": "2", "keyword": "PMP", "category": "certification"},
                ]
            )

    class KeywordInsightsTable:
        def select(self, value):
            return KeywordInsightsSelectQuery().select(value)

        def upsert(self, rows, on_conflict):
            assert on_conflict == "keyword,category"

            class FakeUpsertQuery:
                def execute(self_inner):
                    upserted.extend(rows)
                    return SimpleNamespace(data=rows)

            return FakeUpsertQuery()

    class JobKeywordInsightsTable:
        def select(self, value):
            return FactsSelectQuery().select(value)

    class FakeDb:
        def table(self, name):
            if name == "keyword_insights":
                return KeywordInsightsTable()
            if name == "job_keyword_insights":
                return JobKeywordInsightsTable()
            raise AssertionError(name)

    inserted_facts = [
        {"job_id": "1", "keyword": "AWS", "category": "technology"},
        {"job_id": "2", "keyword": "AWS", "category": "technology"},
        {"job_id": "2", "keyword": "PMP", "category": "certification"},
    ]

    analyze_jobs.update_keyword_insights_from_facts(inserted_facts, db=FakeDb())

    by_key = {(row["keyword"], row["category"]): row for row in upserted}
    assert by_key[("AWS", "technology")]["count"] == 2
    assert by_key[("PMP", "certification")]["count"] == 1


def test_build_job_keyword_facts_uses_job_keyed_results_not_cross_product():
    batch = [
        {"job_id": "1", "job_title": "A", "description": "Needs AWS"},
        {"job_id": "2", "job_title": "B", "description": "Needs PMP"},
    ]
    extracted = {
        "1": [analyze_jobs.KeywordItem(keyword="AWS", category="technology")],
        "2": [analyze_jobs.KeywordItem(keyword="PMP", category="certification")],
    }

    facts = analyze_jobs.build_job_keyword_facts(batch, extracted)

    assert facts == [
        {"job_id": "1", "keyword": "AWS", "category": "technology"},
        {"job_id": "2", "keyword": "PMP", "category": "certification"},
    ]


def test_run_backfill_loops_until_no_unanalyzed_jobs_remain(monkeypatch):
    batches = [
        [{"job_id": "1", "job_title": "A", "description": "Needs AWS"}],
        [{"job_id": "2", "job_title": "B", "description": "Needs SQL"}],
        [],
    ]
    extracted_batches = [
        {"1": [analyze_jobs.KeywordItem(keyword="AWS", category="technology")]},
        {"2": [analyze_jobs.KeywordItem(keyword="SQL", category="technology")]},
    ]
    fact_calls = []
    marked = []

    monkeypatch.setattr(analyze_jobs, "_get_db", lambda: object())
    monkeypatch.setattr(
        analyze_jobs,
        "fetch_unanalyzed_jobs",
        lambda db=None, limit=None, backfill_all=False: batches.pop(0),
    )
    monkeypatch.setattr(analyze_jobs, "extract_keywords_from_batch", lambda batch, client=None, max_retries=None: extracted_batches.pop(0))

    def fake_upsert_facts(facts, db=None):
        fact_calls.append(facts)
        return facts

    monkeypatch.setattr(analyze_jobs, "upsert_job_keyword_facts", fake_upsert_facts)
    monkeypatch.setattr(analyze_jobs, "update_keyword_insights_from_facts", lambda facts, db=None: None)
    monkeypatch.setattr(analyze_jobs, "mark_jobs_analyzed", lambda job_ids, db=None: marked.append(job_ids))

    processed = analyze_jobs.run(backfill_all=True)

    assert processed == 2
    assert len(fact_calls) == 2
    assert marked == [["1"], ["2"]]
