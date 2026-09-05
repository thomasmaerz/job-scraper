import pytest

import linkedin_source_policy


def test_gate_waits_then_consumes_exact_grant(monkeypatch):
    grants = iter([
        {"outcome": "wait", "wait_ms": 250},
        {"outcome": "grant", "grant_id": "grant-1"},
    ])
    sleeps = []
    monkeypatch.setattr(
        linkedin_source_policy.supabase_utils,
        "acquire_linkedin_request_grant",
        lambda *_args, **_kwargs: next(grants),
    )
    monkeypatch.setattr(
        linkedin_source_policy.supabase_utils,
        "consume_linkedin_request_grant",
        lambda *_args, **_kwargs: {
            "consumed": True,
            "started_at": "2026-09-04T12:00:00Z",
        },
    )
    monkeypatch.setattr(linkedin_source_policy.time, "sleep", sleeps.append)

    grant = linkedin_source_policy.DurableLinkedInRequestGate("test").acquire(
        "detail", "source-1"
    )

    assert grant.grant_id == "grant-1"
    assert sleeps == [0.25]


def test_gate_propagates_open_circuit(monkeypatch):
    monkeypatch.setattr(
        linkedin_source_policy.supabase_utils,
        "acquire_linkedin_request_grant",
        lambda *_args, **_kwargs: {"outcome": "circuit_open", "reason": "challenge"},
    )

    with pytest.raises(linkedin_source_policy.LinkedInCircuitOpen, match="challenge"):
        linkedin_source_policy.DurableLinkedInRequestGate("test").acquire(
            "search", "scope-1"
        )


def test_gate_rejects_invalidated_finish(monkeypatch):
    monkeypatch.setattr(
        linkedin_source_policy.supabase_utils,
        "finish_linkedin_request_grant",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(
        linkedin_source_policy.LinkedInGrantRejected, match="invalidated"
    ):
        linkedin_source_policy.DurableLinkedInRequestGate("test").finish(
            linkedin_source_policy.ConsumedGrant(
                "grant-1",
                linkedin_source_policy.datetime.fromisoformat(
                    "2026-09-04T12:00:00+00:00"
                ),
            ),
            "complete",
            200,
        )
