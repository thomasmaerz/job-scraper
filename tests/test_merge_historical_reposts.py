import merge_historical_reposts


def test_build_merge_plan_uses_direct_non_transitive_clusters_and_best_survivor():
    common = " ".join(f"token{index}" for index in range(100))
    rows = [
        {
            "job_id": "1",
            "company": "Acme",
            "job_title": "Senior Project Manager - Toronto",
            "description": common + " first",
            "description_fingerprint": "one",
            "scraped_at": "2026-01-01",
        },
        {
            "job_id": "2",
            "company": "Acme",
            "job_title": "Senior Project Manager - Calgary",
            "description": common + " second",
            "description_fingerprint": "two",
            "scraped_at": "2026-02-01",
            "applicant_count": 25,
            "salary_text": "$100,000-$120,000",
        },
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)

    assert plan == [
        {
            "source_job_id": "1",
            "survivor_job_id": "2",
            "match_method": "cluster_member",
            "match_similarity": None,
        }
    ]
