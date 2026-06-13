from unittest.mock import patch

import scraper


def test_process_linkedin_query_stamps_search_query_archetype_and_filter_profile():
    fake_details = {
        "job_id": "123",
        "job_title": "Technical Program Manager",
        "description": "Own software delivery and cross-functional execution.",
        "company": "Example Corp",
        "provider": "linkedin",
    }

    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["123"]), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=(set(), set())), \
         patch.object(scraper, "_fetch_linkedin_job_details", return_value=fake_details):
        jobs = scraper.process_linkedin_query(
            search_query="Technical Program Manager",
            location="Canada",
            limit=None,
            archetype="software_tpm",
            filter_profile="software_tpm_v1",
        )

    assert jobs == [{
        "job_id": "123",
        "job_title": "Technical Program Manager",
        "description": "Own software delivery and cross-functional execution.",
        "company": "Example Corp",
        "provider": "linkedin",
        "search_query": "Technical Program Manager",
        "archetype": "software_tpm",
        "filter_profile": "software_tpm_v1",
    }]
