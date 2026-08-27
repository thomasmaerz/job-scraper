from unittest.mock import patch
from uuid import UUID

import scraper
import requests
import freehire_compat


def _detail(job_id, search_card=None):
    return {
        "job_id": job_id,
        "job_title": "Technical Program Manager",
        "company": "Acme",
        "location": "Toronto",
        "description": "Own software delivery.",
        "provider": "linkedin",
        "posted_at": (search_card or {}).get("posted_at"),
    }


def test_known_card_observation_happens_before_bounded_relist_fetch(monkeypatch):
    cards = [
        {"job_id": str(index), "posted_at": "2026-08-03", "posted_relative_text": "today"}
        for index in range(1, 5)
    ]
    context = {
        str(index): {
            "canonical_job_id": f"canonical-{index}",
            "observations": [
                {"posted_at": "2026-08-01", "observed_at": "2026-08-01T10:00:00Z", "ingestion_run_id": "a"},
                {"posted_at": "2026-08-01", "observed_at": "2026-08-02T10:00:00Z", "ingestion_run_id": "b"},
            ],
        }
        for index in range(1, 5)
    }
    fetched = []
    saved_observations = []
    monkeypatch.setattr(scraper.config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY", 2)
    monkeypatch.setattr(scraper.config, "LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN", 20)
    monkeypatch.setattr(scraper, "_relist_detail_fetches_used", 0)

    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=cards), \
         patch.object(scraper, "_fetch_linkedin_job_details", side_effect=lambda job_id, search_card=None: fetched.append(job_id) or (_detail(job_id, search_card), {"applicant_count": 26})), \
         patch.object(scraper.supabase_utils, "start_ingestion_run"), \
         patch.object(scraper.supabase_utils, "finish_ingestion_run"), \
         patch.object(scraper.supabase_utils, "get_listing_tracking_context", return_value=context), \
         patch.object(scraper.supabase_utils, "save_listing_states"), \
         patch.object(scraper.supabase_utils, "save_listing_observations", side_effect=lambda cards, **kwargs: saved_observations.extend(cards)), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=({"1", "2", "3", "4"}, set())), \
         patch.object(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", return_value=set()):
        jobs = scraper.process_linkedin_query("TPM", "Canada", archetype="software_tpm")

    assert [card["job_id"] for card in saved_observations] == ["1", "2", "3", "4"]
    assert fetched == ["1", "2"]
    assert all(job["applicant_count"] == 26 for job in jobs)
    assert all(job["same_id_relist_candidate"] is True for job in jobs)

    existing = {
        "job_id": "canonical-1",
        "provider": "linkedin",
        "job_title": "Technical Program Manager",
        "company": "Acme",
        "location": "Toronto",
        "description": "Old delivery description.",
        "listing_instances": [{"job_id": "1", "posted_at": "2026-08-01"}],
        "freehire_compat_status": "current",
    }
    existing["freehire_compat_input_hash"] = freehire_compat.compute_classification_hash(existing)

    update = scraper.supabase_utils.prepare_repost_update_payload(existing, jobs[0])

    assert update["applicant_count"] == 26
    assert update["freehire_compat_status"] == "pending"
    assert update["freehire_compat_input_hash"] != existing["freehire_compat_input_hash"]


def test_same_run_without_new_transition_does_not_refetch_known_id():
    card = {"job_id": "1", "posted_at": "2026-08-03"}
    context = {
        "1": {
            "canonical_job_id": "canonical",
            "observations": [
                {"posted_at": "2026-08-01", "observed_at": "2026-08-01T10:00:00Z", "ingestion_run_id": "a"},
                {"posted_at": "2026-08-01", "observed_at": "2026-08-02T10:00:00Z", "ingestion_run_id": "b"},
                {"posted_at": "2026-08-03", "observed_at": "2026-08-03T10:00:00Z", "ingestion_run_id": scraper.SCRAPE_RUN_ID},
            ],
        }
    }
    with patch.object(scraper.uuid, "uuid4", return_value=UUID(scraper.SCRAPE_RUN_ID)), \
         patch.object(scraper, "_fetch_linkedin_job_ids", return_value=[card]), \
         patch.object(scraper, "_fetch_linkedin_job_details") as fetch_details, \
         patch.object(scraper.supabase_utils, "start_ingestion_run"), \
         patch.object(scraper.supabase_utils, "finish_ingestion_run"), \
         patch.object(scraper.supabase_utils, "get_listing_tracking_context", return_value=context), \
         patch.object(scraper.supabase_utils, "save_listing_states"), \
         patch.object(scraper.supabase_utils, "save_listing_observations"), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=({"1"}, set())), \
         patch.object(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", return_value=set()):
        jobs = scraper.process_linkedin_query("TPM", "Canada", archetype="software_tpm")

    assert jobs == []
    fetch_details.assert_not_called()


def test_relist_refresh_uses_existing_detail_retry_path(monkeypatch):
    class Response:
        def __init__(self, status_code, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(response=self)

    responses = [Response(429), Response(200, "<html></html>")]
    monkeypatch.setattr(scraper.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(scraper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(scraper.random, "uniform", lambda *_args: 0)
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    result = scraper._fetch_linkedin_job_details("known", {"posted_at": "2026-08-03"})

    assert result is not None
    details, metadata = result
    assert details["job_id"] == "known"
    assert metadata["detail_metadata_checked_at"]
    assert responses == []
