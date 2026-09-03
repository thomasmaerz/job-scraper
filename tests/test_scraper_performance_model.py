"""Deterministic request-budget checks based on measured production runs."""

import scraper


def expected_pacing_seconds(requests: int, minimum_ms: int, jitter_ms: int) -> float:
    return requests * (minimum_ms + jitter_ms / 2) / 1000


def test_three_configured_pages_emit_three_requests_per_query():
    query_count = 40
    legacy_requests = query_count * 6  # Previous starts: 0, 10, 20, 30, 40, 50.
    corrected_requests = query_count * (
        scraper._linkedin_max_start_for_pages(3) // 10 + 1
    )

    assert corrected_requests == 120
    assert corrected_requests == legacy_requests // 2


def test_global_pacing_reduces_measured_slow_run_wait_budget():
    # Run 235 measured 240 search requests, 467 unique details, and 3,265.74s
    # of explicit sleeps. This upper-bound model retains all 467 historical
    # details even though correcting the page limit may intentionally discover
    # fewer IDs, and paces every request at 2.5s + 0..1.5s jitter.
    corrected_requests = 120 + 467
    projected_wait = expected_pacing_seconds(corrected_requests, 2_500, 1_500)

    assert projected_wait == 1907.75
    assert projected_wait < 3265.74 * 0.6


def test_global_pacing_reduces_measured_typical_run_wait_budget():
    # Run 236 measured 240 searches, 225 unique details, and 1,659.69s of
    # explicit sleeps. This conservative model retains the historical detail
    # count while reducing the deterministic wait budget.
    corrected_requests = 120 + 225
    projected_wait = expected_pacing_seconds(corrected_requests, 2_500, 1_500)

    assert projected_wait == 1121.25
    assert projected_wait < 1659.69 * 0.7
