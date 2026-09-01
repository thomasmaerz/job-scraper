import json

import config
import freehire_compat


def test_pinned_vocabularies_are_exact():
    assert config.FREEHIRE_CATEGORIES == frozenset({
        "software_engineering", "backend", "frontend", "fullstack", "mobile",
        "devops", "sre", "network_engineering", "data_engineering", "data_science",
        "data_analytics", "ml_ai", "ai_engineering", "qa", "security", "hardware",
        "embedded", "blockchain", "architecture", "design", "engineering_design",
        "product", "project_management", "management", "marketing", "sales",
        "support", "business_analysis", "solutions_engineering", "developer_relations",
        "technical_writing", "recruiting", "hr", "finance", "legal", "operations",
        "customer_success", "other",
    })
    assert config.FREEHIRE_SENIORITY_LEVELS == frozenset({
        "", "intern", "junior", "middle", "senior", "lead", "staff", "principal", "c_level",
    })


def test_remote_is_standalone_visible_text_only():
    positives = [
        {"job_title": "Remote Engineer"},
        {"location": "REMOTE"},
        {"description": "Work (remote) or onsite"},
        {"description": "remote-first"},
        {"description": "[remote](https://example.com/office)"},
        {"description": "<p>Remote work</p>"},
    ]
    for job in positives:
        is_remote, evidence = freehire_compat.classify_remote(job)
        assert is_remote is True
        assert evidence["field"] in job
        assert evidence["text"] == "remote"

    negatives = [
        {"description": "Work remotely"},
        {"description": "Avoid remoteness"},
        {"description": "Hybrid or work from home"},
        {"location": "Canada"},
        {"description": "https://remote.example/jobs"},
        {"description": '<a href="https://example.com/remote">Apply</a>'},
        {"description": "[Apply](https://example.com/remote)"},
    ]
    for job in negatives:
        assert freehire_compat.classify_remote(job) == (False, None)


def test_visible_text_decodes_entities_and_removes_hidden_content():
    assert freehire_compat.normalize_visible_text("Remote &amp;amp; Canada") == "remote & canada"
    hidden = '<script>Remote</script><span hidden>Remote</span><p>Hybrid</p>'
    assert freehire_compat.normalize_visible_text(hidden) == "hybrid"
    assert freehire_compat.classify_remote({"description": hidden}) == (False, None)


def test_classification_and_import_hashes_are_exact_and_separate():
    job = {
        "job_id": "canonical",
        "latest_job_id": "live-1",
        "job_title": "Senior TPM",
        "location": "Toronto",
        "description": "<p>Remote delivery</p>",
        "level": "Senior",
        "first_seen_at": "2026-01-01T00:00:00Z",
        "last_seen_at": "2026-01-02T00:00:00Z",
        "posted_at": "2026-01-01T00:00:00Z",
        "freehire_category": "project_management",
        "freehire_seniority": "senior",
    }
    first = freehire_compat.compute_classification_hash(job)
    assert first == freehire_compat.compute_classification_hash(dict(job))
    assert first != freehire_compat.compute_classification_hash({**job, "description": "Onsite delivery"})
    assert first == freehire_compat.compute_classification_hash({**job, "latest_job_id": "live-2"})
    assert freehire_compat.compute_import_hash(job) != freehire_compat.compute_import_hash(
        {**job, "latest_job_id": "live-2"}
    )
    assert freehire_compat.compute_import_hash(job) != freehire_compat.compute_import_hash(
        {**job, "company": "Changed"}
    )


def test_source_seniority_maps_only_unambiguous_values():
    assert freehire_compat.source_seniority("Senior") == "senior"
    assert freehire_compat.source_seniority("Internship") == "intern"
    assert freehire_compat.source_seniority("Mid-Senior level") == ""
    assert freehire_compat.source_seniority("Entry level") == ""


def test_token_budget_packing_honors_budget_and_fifty_job_cap(monkeypatch):
    monkeypatch.setattr(freehire_compat, "_estimate_tokens", lambda text, model=None: len(text))
    monkeypatch.setattr(config, "FREEHIRE_OUTPUT_TOKENS_PER_JOB", 1)
    jobs = [{"job_id": str(index), "job_title": "x", "description": "y"} for index in range(51)]
    batches = freehire_compat.pack_batches(jobs, token_budget=1000000, max_jobs=100)
    assert [len(batch) for batch in batches] == [50, 1]


def test_classifier_retries_only_missing_or_invalid_ids_and_uses_schema(monkeypatch):
    calls = []

    class FakeClient:
        model = "fake/model"
        model_chain = None
        last_model_used = "fake/model"

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps({"jobs": [
                    {"job_id": "1", "category": "project_management", "seniority": "senior", "confidence": 0.9},
                    {"job_id": "2", "category": "placeholder", "seniority": "middle", "confidence": 0.8},
                ]})
            return json.dumps({"jobs": [
                {"job_id": "2", "category": "backend", "seniority": "middle", "confidence": 0.8},
            ]})

    monkeypatch.setattr(config, "FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS", 0)
    jobs = [
        {"job_id": "1", "job_title": "TPM", "description": "Programs"},
        {"job_id": "2", "job_title": "Engineer", "description": "APIs"},
    ]
    outcome = freehire_compat.classify_batch(jobs, client=FakeClient(), max_retries=2)
    assert set(outcome.results) == {"1", "2"}
    assert outcome.failures == {}
    assert calls[0]["response_format"] is freehire_compat.FreehireClassificationBatch
    assert "max_api_attempts" not in calls[0]
    assert "Job ID: 1" in calls[0]["prompt"] and "Job ID: 2" in calls[0]["prompt"]
    assert "Job ID: 1" not in calls[1]["prompt"] and "Job ID: 2" in calls[1]["prompt"]


def test_classifier_splits_and_isolates_poison_record(monkeypatch):
    calls = []

    class FakeClient:
        model = "fake/model"
        model_chain = None

        def generate_content(self, **kwargs):
            calls.append(kwargs["prompt"])
            if "Job ID: poison" in kwargs["prompt"]:
                raise ValueError("poison")
            return json.dumps({"jobs": [
                {"job_id": "good", "category": "other", "seniority": "", "confidence": 0.5},
            ]})

    monkeypatch.setattr(config, "FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS", 0)
    outcome = freehire_compat.classify_batch(
        [{"job_id": "good"}, {"job_id": "poison"}],
        client=FakeClient(),
        max_retries=1,
    )
    assert set(outcome.results) == {"good"}
    assert outcome.failures == {"poison": "poison"}
    assert outcome.splits == 1


def test_classifier_does_not_split_global_transport_failure(monkeypatch, caplog):
    class FakeClient:
        model = "fake/model"
        model_chain = None

        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            raise ConnectionError("transport unavailable")

    client = FakeClient()
    outcome = freehire_compat.classify_batch(
        [{"job_id": "1"}, {"job_id": "2"}],
        client=client,
        max_retries=3,
        max_requests=10,
    )
    assert client.calls == 1
    assert outcome.splits == 0
    assert set(outcome.failures) == {"1", "2"}
    assert outcome.global_error == "transport unavailable"
    assert "Freehire classification stopped after model-pool failure for 2 jobs" in caplog.text
    assert "ConnectionError: transport unavailable" in caplog.text


def test_classifier_hard_request_budget_bounds_split_tree(monkeypatch):
    class FakeClient:
        model = "fake/model"
        model_chain = None

        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return '{"jobs":[]}'

    monkeypatch.setattr(config, "FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS", 0)
    client = FakeClient()
    outcome = freehire_compat.classify_batch(
        [{"job_id": str(index)} for index in range(8)],
        client=client,
        max_retries=1,
        max_requests=3,
    )
    assert client.calls == outcome.requests == 3
    assert set(outcome.results) == set()
