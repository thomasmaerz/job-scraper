import backfill_salary_metadata


def test_salary_backfill_is_dry_run_by_default(monkeypatch):
    monkeypatch.setattr(
        backfill_salary_metadata,
        "fetch_missing_salary_jobs",
        lambda: [{"job_id": "1", "description": "Salary: $80,000-$95,000 CAD"}],
    )

    result = backfill_salary_metadata.run(apply=False)

    assert result == {"scanned": 1, "recoverable": 1, "updated": 0}
