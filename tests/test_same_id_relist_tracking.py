from copy import deepcopy

import backfill_same_id_relists
import relist_tracking
import supabase_utils


def observations(*dates):
    return [
        {
            "source_job_id": "source-1",
            "posted_at": value,
            "observed_at": f"2026-08-{index + 1:02d}T10:00:00Z",
            "ingestion_run_id": f"run-{index + 1}",
        }
        for index, value in enumerate(dates)
    ]


def test_stable_forward_two_day_transition_accepts_one_event():
    folded = relist_tracking.fold_observations(observations("2026-08-01", "2026-08-01", "2026-08-03"))
    assert [event["relisted_on"] for event in folded["events"]] == ["2026-08-03"]


def test_pending_event_excludes_already_projected_dates():
    folded = relist_tracking.fold_observations(
        observations("2026-08-01", "2026-08-01", "2026-08-03")
    )
    assert relist_tracking.latest_pending_event(folded)["relisted_on"] == "2026-08-03"
    assert relist_tracking.latest_pending_event(folded, {"2026-08-03"}) is None


def test_backward_and_unstable_forward_dates_are_corrections():
    folded = relist_tracking.fold_observations(observations("2026-08-03", "2026-08-01", "2026-08-05"))
    assert folded["events"] == []
    assert len(folded["corrections"]) == 2


def test_same_id_relist_is_idempotent_and_does_not_inflate_seen_count():
    existing = {
        "job_id": "canonical",
        "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
        "same_id_relist_count": 0,
    }
    incoming = {
        "job_id": "source-1",
        "location": "Toronto",
        "posted_at": "2026-08-03",
        "same_id_relist_candidate": True,
        "same_id_relist_date": "2026-08-03",
    }
    first = supabase_utils.prepare_repost_update_payload(existing, incoming)
    second = supabase_utils.prepare_repost_update_payload({**existing, **first}, incoming)
    assert first["seen_count"] == second["seen_count"] == 1
    assert first["same_id_relist_count"] == second["same_id_relist_count"] == 1
    assert len(first["listing_instances"]) == len(second["listing_instances"]) == 2


def test_same_id_relist_and_same_day_new_id_share_one_wave():
    existing = {
        "job_id": "canonical",
        "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
    }
    existing.update(supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-03", "same_id_relist_candidate": True,
    }))
    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-2", "location": "Toronto", "posted_at": "2026-08-03",
    })
    assert update["seen_count"] == 2
    assert update["posting_wave_count"] == 2
    assert update["repost_count"] == 1


def test_description_only_change_updates_canonical_without_relist():
    old = "old description\n" * 100
    new = "new description\n" * 100
    existing = {
        "job_id": "canonical",
        "description": old,
        "description_fingerprint": supabase_utils.make_description_fingerprint(old),
        "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
        "same_id_relist_count": 0,
    }
    update = supabase_utils.prepare_repost_update_payload(existing, {"job_id": "source-1", "description": new})
    assert update["description"] == new
    assert update["same_id_relist_count"] == 0
    assert len(update["listing_instances"]) == 1


def test_backfill_payload_is_deterministic_idempotent_and_pure():
    row = {
        "job_id": "canonical",
        "location": "Toronto",
        "listing_instances": [{"job_id": "source-1", "location": "Toronto", "posted_at": "2026-08-01"}],
        "same_id_relist_count": 0,
    }
    source = deepcopy(row)
    observed = observations("2026-08-01", "2026-08-01", "2026-08-03")
    first = backfill_same_id_relists.build_payload(row, list(reversed(observed)))
    second = backfill_same_id_relists.build_payload({**row, **first}, observed)
    assert row == source
    assert first == second
    assert first["same_id_relist_count"] == 1
