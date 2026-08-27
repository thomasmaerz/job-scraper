import supabase_utils
import freehire_compat


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _RecordingQuery:
    def __init__(self, response_data=None):
        self.response_data = response_data or []
        self.selected = None
        self.filters = []
        self.upsert_payloads = []
        self.update_payloads = []

    def select(self, fields):
        self.selected = fields
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def gte(self, field, value):
        self.filters.append(("gte", field, value))
        return self

    def range(self, start, end):
        self.filters.append(("range", start, end))
        return self

    def upsert(self, payload):
        self.upsert_payloads.append(payload)
        return self

    def update(self, payload):
        self.update_payloads.append(payload)
        return self

    def is_(self, field, value):
        self.filters.append(("is", field, value))
        return self

    def execute(self):
        return _FakeResponse(self.response_data)


class _FakeSupabase:
    def __init__(self, query):
        self.query = query

    def table(self, _name):
        return self.query


def test_normalize_title_handles_clear_cut_abbreviations():
    assert supabase_utils.normalize_title("Sr. Project Manager") == "senior project manager"
    assert supabase_utils.normalize_title("Technical   Project-Manager") == "technical project manager"


def test_normalize_title_does_not_corrupt_embedded_tokens():
    assert supabase_utils.normalize_title("SRE Manager") == "sre manager"


def test_normalize_location_collapses_formatting_noise():
    assert supabase_utils.normalize_location(" Toronto , Ontario  , Canada ") == "toronto ontario canada"


def test_build_canonical_key_uses_normalized_parts():
    key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Chandos Construction",
        title="Sr. Project Manager",
        location="Chalk River, Ontario, Canada",
    )
    assert key == "linkedin|chandos construction|senior project manager|chalk river ontario canada"


def test_normalize_company_treats_separator_variants_consistently():
    assert supabase_utils.normalize_company("Foo-Bar") == "foo bar"
    assert supabase_utils.normalize_company("Foo/Bar") == "foo bar"
    assert supabase_utils.normalize_company("Foo Bar") == "foo bar"


def test_build_canonical_key_uses_normalized_company_separator_variants():
    dash_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo-Bar",
        title="Sr. Project Manager",
        location="Toronto / Ontario - Canada",
    )
    slash_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo/Bar",
        title="Senior Project Manager",
        location="Toronto, Ontario, Canada",
    )
    space_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo Bar",
        title="Sr Project Manager",
        location="Toronto Ontario Canada",
    )

    assert dash_key == slash_key == space_key == "linkedin|foo bar|senior project manager|toronto ontario canada"


def test_description_fingerprint_ignores_minor_formatting_changes():
    sentence = "We are Chandos. Inclusion, collaboration, innovation, and continuous improvement drive every project we deliver. "
    a = sentence * 6
    b = (sentence.replace(". ", "\n\n").replace(", ", ",  ").replace(" improvement", " improvement!") * 6)

    fingerprint_a = supabase_utils.make_description_fingerprint(a)
    fingerprint_b = supabase_utils.make_description_fingerprint(b)

    assert fingerprint_a is not None
    assert fingerprint_a == fingerprint_b


def test_description_fingerprint_normalizes_unicode_punctuation_for_long_equivalents():
    plain = (
        "We're building client-focused teams that solve complex problems with care, speed, and accountability. "
        "Our people partner across design, delivery, and operations to keep commitments clear and work moving forward. "
    ) * 4
    formatted = (
        "We’re building client—focused teams that solve complex problems with care, speed, and accountability.\n"
        "• Our people partner across design, delivery, and operations to keep commitments clear and work moving forward.\n\n"
    ) * 4

    fingerprint_plain = supabase_utils.make_description_fingerprint(plain)
    fingerprint_formatted = supabase_utils.make_description_fingerprint(formatted)

    assert fingerprint_plain is not None
    assert fingerprint_formatted is not None
    assert fingerprint_plain == fingerprint_formatted


def test_prepare_canonical_insert_payload():
    job = {
        "job_id": "4426608777",
        "provider": "linkedin",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation.",
        "posted_at": "2026-06-12",
        "posted_relative_text": "18 hours ago",
        "applicant_count": 26,
    }

    payload = supabase_utils.prepare_canonical_insert_payload(job)

    assert payload["original_job_id"] == "4426608777"
    assert payload["latest_job_id"] == "4426608777"
    assert payload["seen_count"] == 1
    assert payload["posting_wave_count"] == 1
    assert payload["repost_count"] == 0
    assert payload["listing_instances"][0]["job_id"] == "4426608777"
    assert payload["listing_instances"][0]["location"] == "Chalk River, Ontario, Canada"
    assert payload["listing_instances"][0]["variant_type"] == "original"
    assert payload["freehire_compat_status"] == "pending"
    assert payload["freehire_compat_input_hash"]
    assert payload["is_remote"] is False


def test_prepare_repost_update_payload():
    existing = {
        "job_id": "4394716706",
        "listing_instances": [{"job_id": "4394716706", "location": "Toronto", "posted_at": "2026-05-29", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
        "provider": "linkedin",
        "job_title": "Project Manager",
        "location": "Toronto",
        "description": "Delivery",
        "freehire_compat_input_hash": "old",
    }
    new_job = {
        "job_id": "4426608777",
        "location": "Toronto",
        "posted_at": "2026-06-12",
        "posted_relative_text": "18 hours ago",
        "applicant_count": 26,
        "salary_text": "$120,000-$135,000 CAD",
    }

    update = supabase_utils.prepare_repost_update_payload(existing, new_job)

    assert update["latest_job_id"] == "4426608777"
    assert update["seen_count"] == 2
    assert update["posting_wave_count"] == 2
    assert update["repost_count"] == 1
    assert len(update["listing_instances"]) == 2
    assert update["freehire_compat_status"] == "pending"
    assert update["freehire_compat_input_hash"] != "old"


def test_prepare_canonical_insert_payload_preserves_missing_job_id_as_none():
    job = {
        "job_id": None,
        "provider": "linkedin",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
    }

    payload = supabase_utils.prepare_canonical_insert_payload(job)

    assert payload["original_job_id"] is None
    assert payload["latest_job_id"] is None
    assert payload["listing_instances"][0]["job_id"] is None


def test_prepare_repost_update_payload_preserves_existing_canonical_fields_on_partial_scrape():
    existing = {
        "job_id": "4394716706",
        "latest_job_id": "4394716706",
        "last_seen_at": "2026-05-29T12:00:00Z",
        "last_seen_posted_at": "2026-05-29",
        "posted_relative_text": "2 weeks ago",
        "applicant_count": 13,
        "salary_text": "$120,000-$135,000 CAD",
        "salary_min": 120000,
        "salary_max": 135000,
        "salary_currency": "CAD",
        "recruiter_name": "Jane Recruiter",
        "recruiter_profile_url": "https://www.linkedin.com/in/jane-recruiter",
        "recruiter_identifier": "jane-recruiter",
        "listing_instances": [{"job_id": "4394716706", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
    }
    new_job = {
        "job_id": "4426608777",
        "posted_at": None,
        "posted_relative_text": None,
        "applicant_count": None,
        "salary_text": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "recruiter_name": None,
        "recruiter_profile_url": None,
        "recruiter_identifier": None,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, new_job)

    assert update["latest_job_id"] == "4426608777"
    assert update["last_seen_posted_at"] == "2026-05-29"
    assert update["posted_relative_text"] == "2 weeks ago"
    assert update["applicant_count"] == 13
    assert update["salary_text"] == "$120,000-$135,000 CAD"
    assert update["salary_min"] == 120000
    assert update["salary_max"] == 135000
    assert update["salary_currency"] == "CAD"
    assert update["recruiter_name"] == "Jane Recruiter"
    assert update["recruiter_profile_url"] == "https://www.linkedin.com/in/jane-recruiter"
    assert update["recruiter_identifier"] == "jane-recruiter"


def test_partial_repost_none_values_do_not_invalidate_current_classification():
    existing = {
        "job_id": "canonical",
        "provider": "linkedin",
        "job_title": "Senior TPM",
        "location": "Toronto",
        "description": "Remote delivery",
        "level": "Senior",
        "listing_instances": [{"job_id": "source-1"}],
    }
    existing["freehire_compat_input_hash"] = freehire_compat.compute_classification_hash(existing)
    existing["freehire_compat_status"] = "current"

    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-1",
        "job_title": None,
        "location": None,
        "description": None,
        "level": None,
    })

    assert "freehire_compat_status" not in update
    assert "freehire_compat_input_hash" not in update
    assert update["is_remote"] is True


def test_prepare_repost_update_payload_preserves_existing_latest_job_id_when_new_job_id_missing():
    existing = {
        "job_id": "4394716706",
        "latest_job_id": "4426608777",
        "listing_instances": [{"job_id": "4394716706", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
    }
    new_job = {
        "job_id": None,
        "posted_at": None,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, new_job)

    assert update["latest_job_id"] == "4426608777"


def test_prepare_repost_update_payload_does_not_count_same_listing_twice():
    existing = {
        "job_id": "canonical",
        "latest_job_id": "source-2",
        "listing_instances": [{"job_id": "source-1"}, {"job_id": "source-2"}],
        "seen_count": 2,
        "repost_count": 1,
    }

    update = supabase_utils.prepare_repost_update_payload(
        existing,
        {"job_id": "source-2", "applicant_count": 42},
    )

    assert update["seen_count"] == 2
    assert update["repost_count"] == 0
    assert len(update["listing_instances"]) == 2
    assert update["listing_instances"][1]["applicant_count"] == 42


def test_prepare_repost_update_payload_does_not_mutate_cas_source_snapshot():
    existing = {
        "job_id": "canonical",
        "listing_instances": [{
            "job_id": "source-1",
            "location": "Toronto",
            "scraped_at": "2026-08-01T00:00:00Z",
            "applicant_count": 10,
        }],
        "seen_count": 1,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-1",
        "location": "Toronto",
        "applicant_count": 25,
    })

    assert existing["listing_instances"][0]["applicant_count"] == 10
    assert "last_seen_at" not in existing["listing_instances"][0]
    assert update["listing_instances"][0]["applicant_count"] == 25
    assert update["listing_instances"][0]["last_seen_at"]


def test_eight_simultaneous_ids_are_one_wave_with_zero_reposts():
    existing = {
        "job_id": "source-1",
        "listing_instances": [{
            "job_id": "source-1",
            "location": "Toronto, Ontario, Canada",
            "posted_at": "2026-08-20",
            "scrape_run_id": "affirm-run",
        }],
        "seen_count": 1,
        "repost_count": 0,
    }

    for index in range(1, 8):
        existing.update(supabase_utils.prepare_repost_update_payload(existing, {
            "job_id": f"source-{index + 1}",
            "location": "Toronto, Ontario, Canada",
            "posted_at": "2026-08-20",
            "scrape_run_id": "affirm-run",
        }))

    assert existing["seen_count"] == 8
    assert existing["posting_wave_count"] == 1
    assert existing["repost_count"] == 0
    assert len(existing["listing_instances"]) == 8
    assert {instance["variant_type"] for instance in existing["listing_instances"]} == {
        "original", "simultaneous_variant"
    }


def test_recruiter_variant_in_same_wave_does_not_increment_reposts():
    existing = {
        "job_id": "source-1",
        "listing_instances": [{
            "job_id": "source-1",
            "location": "Toronto",
            "posted_at": "2026-08-20",
            "scrape_run_id": "run-1",
            "recruiter_identifier": "recruiter-a",
        }],
        "seen_count": 1,
        "repost_count": 0,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-2",
        "location": "Toronto",
        "posted_at": "2026-08-20",
        "scrape_run_id": "run-1",
        "recruiter_identifier": "recruiter-b",
    })

    assert update["seen_count"] == 2
    assert update["posting_wave_count"] == 1
    assert update["repost_count"] == 0
    assert update["listing_instances"][1]["recruiter_identifier"] == "recruiter-b"


def test_multiple_ids_in_two_posting_waves_count_one_repost():
    instances = [
        {"job_id": "1", "location": "Toronto", "posted_at": "2026-08-01", "scrape_run_id": "run-a"},
        {"job_id": "2", "location": "Toronto", "posted_at": "2026-08-01", "scrape_run_id": "run-a"},
        {"job_id": "3", "location": "Toronto", "posted_at": "2026-08-20", "scrape_run_id": "run-b"},
        {"job_id": "4", "location": "Toronto", "posted_at": "2026-08-20", "scrape_run_id": "run-b"},
    ]

    annotated, wave_count, repost_count = supabase_utils.calculate_posting_waves(instances)

    assert len(annotated) == 4
    assert wave_count == 2
    assert repost_count == 1
    assert [instance["posting_wave_index"] for instance in annotated] == [1, 1, 2, 2]


def test_missing_posted_at_same_scrape_run_is_one_wave():
    _, wave_count, repost_count = supabase_utils.calculate_posting_waves([
        {"job_id": "1", "location": "Toronto", "posted_at": None, "scrape_run_id": "run-a"},
        {"job_id": "2", "location": "Toronto", "posted_at": None, "scrape_run_id": "run-a"},
    ])

    assert wave_count == 1
    assert repost_count == 0


def test_missing_all_wave_timestamps_does_not_confirm_reposts():
    _, wave_count, repost_count = supabase_utils.calculate_posting_waves([
        {"job_id": "1", "location": "Toronto"},
        {"job_id": "2", "location": "Toronto"},
    ])

    assert wave_count == 1
    assert repost_count == 0


def test_unknown_locations_never_confirm_chronological_reposts():
    _, wave_count, repost_count = supabase_utils.calculate_posting_waves([
        {"job_id": "1", "location": None, "posted_at": "2026-08-01"},
        {"job_id": "2", "location": None, "posted_at": "2026-08-20"},
    ])

    assert wave_count == 1
    assert repost_count == 0


def test_repeat_source_id_preserves_original_scrape_run_wave():
    existing = {
        "job_id": "source-1",
        "listing_instances": [{
            "job_id": "source-1",
            "location": "Toronto",
            "scrape_run_id": "original-run",
        }],
        "seen_count": 99,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-1",
        "location": "Toronto",
        "scrape_run_id": "later-run",
    })

    assert update["seen_count"] == 1
    assert update["repost_count"] == 0
    assert update["listing_instances"][0]["scrape_run_id"] == "original-run"


def test_repeat_source_id_without_run_keeps_original_scrape_date_fallback():
    existing = {
        "job_id": "source-1",
        "listing_instances": [{
            "job_id": "source-1",
            "location": "Toronto",
            "scraped_at": "2026-08-01T10:00:00Z",
        }],
        "seen_count": 1,
    }

    update = supabase_utils.prepare_repost_update_payload(existing, {
        "job_id": "source-1",
        "location": "Toronto",
        "scrape_run_id": "later-run",
    })

    assert update["listing_instances"][0].get("scrape_run_id") is None
    assert update["listing_instances"][0]["posting_wave_key"] == "toronto|scrape_date:2026-08-01"


def test_cross_location_variants_do_not_inflate_repost_count():
    annotated, wave_count, repost_count = supabase_utils.calculate_posting_waves([
        {"job_id": "toronto-1", "location": "Toronto", "posted_at": "2026-08-01"},
        {"job_id": "calgary-1", "location": "Calgary", "posted_at": "2026-08-01"},
        {"job_id": "calgary-2", "location": "Calgary", "posted_at": "2026-08-01"},
    ])

    assert wave_count == 1
    assert repost_count == 0
    assert annotated[1]["variant_type"] == "location_variant"
    assert annotated[2]["variant_type"] == "simultaneous_variant"


def test_same_normalized_location_later_wave_is_one_repost():
    _, wave_count, repost_count = supabase_utils.calculate_posting_waves([
        {"job_id": "1", "location": "Toronto, Ontario", "posted_at": "2026-08-01"},
        {"job_id": "2", "location": "Toronto / Ontario", "posted_at": "2026-08-20"},
    ])

    assert wave_count == 2
    assert repost_count == 1


def test_posting_wave_indexes_follow_observation_chronology_not_run_id():
    annotated, _, _ = supabase_utils.calculate_posting_waves([
        {"job_id": "later", "location": "Toronto", "scraped_at": "2026-02-01T00:00:00Z", "scrape_run_id": "a"},
        {"job_id": "earlier", "location": "Toronto", "scraped_at": "2026-01-01T00:00:00Z", "scrape_run_id": "z"},
    ])

    by_id = {instance["job_id"]: instance for instance in annotated}
    assert by_id["earlier"]["posting_wave_index"] == 1
    assert by_id["later"]["posting_wave_index"] == 2


def test_prepare_repost_update_payload_reactivates_expired_canonical_role():
    existing = {
        "job_id": "canonical",
        "latest_job_id": "source-1",
        "listing_instances": [{"job_id": "source-1"}],
        "seen_count": 1,
        "repost_count": 0,
        "is_active": False,
        "job_state": "expired",
    }

    update = supabase_utils.prepare_repost_update_payload(existing, {"job_id": "source-2"})

    assert update["is_active"] is True
    assert update["job_state"] == "new"


def test_find_canonical_match_prefers_existing_role():
    existing_rows = [{
        "job_id": "4394716706",
        "canonical_key": "linkedin|chandos construction|industrial construction senior project manager|chalk river ontario canada",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation.",
        "description_fingerprint": supabase_utils.make_description_fingerprint("We are Chandos. Inclusion, collaboration, innovation."),
        "listing_instances": [],
        "seen_count": 1,
        "repost_count": 0,
    }]
    job = {
        "provider": "linkedin",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation!",
    }

    match = supabase_utils.find_canonical_match(job, existing_rows)

    assert match["job_id"] == "4394716706"


def test_get_canonical_candidates_selects_fields_needed_for_partial_repost_updates(monkeypatch):
    query = _RecordingQuery(response_data=[])
    monkeypatch.setattr(supabase_utils, "supabase", _FakeSupabase(query))

    supabase_utils.get_canonical_candidates(provider="linkedin")

    assert query.selected == (
        "job_id, canonical_key, company, job_title, location, description, description_fingerprint, "
        "listing_instances, seen_count, posting_wave_count, repost_count, latest_job_id, last_seen_at, last_seen_posted_at, "
        "posted_relative_text, applicant_count, applicant_count_text, applicant_count_type, "
        "salary_text, salary_min, salary_max, salary_currency, recruiter_name, "
        "recruiter_profile_url, recruiter_identifier, detail_metadata_checked_at, "
        "is_active, job_state, same_id_relist_count, provider, level, "
        "freehire_category, freehire_seniority, is_remote, freehire_remote_evidence, "
        "freehire_compat_status, freehire_compat_input_hash, freehire_compat_import_hash"
    )


def test_find_canonical_match_does_not_merge_different_locations():
    shared = " ".join(f"delivery token{index}" for index in range(100))
    existing = {
        "job_id": "old",
        "company": "Example Inc.",
        "job_title": "Senior Project Manager - Toronto",
        "location": "Toronto",
        "description": shared + " legacy",
        "description_fingerprint": "different",
    }
    job = {
        "company": "Example Inc",
        "job_title": "Senior Project Manager - Calgary",
        "location": "Calgary",
        "description": shared + " updated",
    }

    assert supabase_utils.find_canonical_match(job, [existing]) is None


def test_find_canonical_match_requires_known_location_for_content_match():
    description = " ".join(f"delivery token{index}" for index in range(100))
    existing = {
        "job_id": "old",
        "company": "Example Inc",
        "job_title": "Senior Project Manager",
        "location": None,
        "description": description,
        "description_fingerprint": "same",
    }
    job = {
        "job_id": "new",
        "company": "Example Inc",
        "job_title": "Senior Project Manager",
        "location": None,
        "description": description,
    }

    assert supabase_utils.find_canonical_match(job, [existing]) is None


def test_find_canonical_match_rejects_low_similarity_same_title():
    existing = {
        "job_id": "old",
        "company": "Example Inc.",
        "job_title": "Senior Project Manager",
        "description": "alpha bravo charlie delta echo foxtrot",
        "description_fingerprint": "different",
    }
    job = {
        "company": "Example Inc",
        "job_title": "Senior Project Manager",
        "description": "systems delivery program stakeholder schedule budget",
    }

    assert supabase_utils.find_canonical_match(job, [existing]) is None


def test_save_jobs_to_supabase_preserves_canonical_and_task2_metadata_fields(monkeypatch):
    query = _RecordingQuery(response_data=[])
    monkeypatch.setattr(supabase_utils, "supabase", _FakeSupabase(query))

    job = {
        "job_id": "4426608777",
        "provider": "linkedin",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation.",
        "posted_at": "2026-06-12",
        "search_query": "Technical Program Manager",
        "archetype": "software_tpm",
        "filter_profile": "software_tpm_v1",
    }

    payload = supabase_utils.prepare_canonical_insert_payload(job)

    supabase_utils.save_jobs_to_supabase([payload])

    saved = query.upsert_payloads[0][0]

    assert saved["search_query"] == "Technical Program Manager"
    assert saved["archetype"] == "software_tpm"
    assert saved["filter_profile"] == "software_tpm_v1"
    assert saved["canonical_key"] == payload["canonical_key"]
    assert saved["original_job_id"] == "4426608777"
    assert saved["latest_job_id"] == "4426608777"
    assert saved["first_seen_at"] == payload["first_seen_at"]
    assert saved["last_seen_at"] == payload["last_seen_at"]
    assert saved["last_seen_posted_at"] == "2026-06-12"
    assert saved["seen_count"] == 1
    assert saved["repost_count"] == 0
    assert saved["listing_instances"] == payload["listing_instances"]
    assert saved["description_fingerprint"] == payload["description_fingerprint"]
    assert saved["freehire_compat_status"] == "pending"
    assert saved["freehire_compat_input_hash"] == payload["freehire_compat_input_hash"]
    assert saved["is_remote"] is False


def test_save_linkedin_jobs_canonicalized_matches_repost_across_normalized_company_variants(monkeypatch):
    existing = {
        "job_id": "4394716706",
        "canonical_key": "linkedin|foo bar|senior project manager|toronto ontario canada",
        "company": "Foo Bar",
        "job_title": "Senior Project Manager",
        "location": "Toronto, Ontario, Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation. " * 4,
        "description_fingerprint": supabase_utils.make_description_fingerprint(
            "We are Chandos. Inclusion, collaboration, innovation. " * 4
        ),
        "listing_instances": [{"job_id": "4394716706", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
        "latest_job_id": "4394716706",
        "last_seen_at": "2026-05-29T12:00:00Z",
        "last_seen_posted_at": "2026-05-29",
        "posted_relative_text": "2 weeks ago",
        "applicant_count": 13,
        "salary_text": "$120,000-$135,000 CAD",
        "salary_min": 120000,
        "salary_max": 135000,
        "salary_currency": "CAD",
        "recruiter_name": "Jane Recruiter",
        "recruiter_profile_url": "https://www.linkedin.com/in/jane-recruiter",
        "recruiter_identifier": "jane-recruiter",
    }
    query = _RecordingQuery(response_data=[existing])
    monkeypatch.setattr(supabase_utils, "supabase", _FakeSupabase(query))

    inserted_payloads = []

    def fake_save_jobs_to_supabase(payloads):
        inserted_payloads.extend(payloads)

    monkeypatch.setattr(supabase_utils, "save_jobs_to_supabase", fake_save_jobs_to_supabase)
    monkeypatch.setattr(supabase_utils, "save_listing_content_version", lambda *args, **kwargs: None)

    supabase_utils.save_linkedin_jobs_canonicalized([{
        "job_id": "4426608777",
        "provider": "linkedin",
        "company": "Foo-Bar",
        "job_title": "Sr. Project Manager",
        "location": "Toronto / Ontario - Canada",
        "description": "We are Chandos. Inclusion, collaboration, innovation! " * 4,
        "posted_at": None,
        "posted_relative_text": None,
        "applicant_count": None,
        "salary_text": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "recruiter_name": None,
        "recruiter_profile_url": None,
        "recruiter_identifier": None,
    }])

    assert ("eq", "company", "Foo-Bar") not in query.filters
    assert inserted_payloads == []
    assert query.update_payloads
    assert query.update_payloads[0]["latest_job_id"] == "4426608777"
    assert query.update_payloads[0]["salary_text"] == "$120,000-$135,000 CAD"
    assert ("eq", "job_id", "4394716706") in query.filters
    assert any(filter_[0:2] == ("eq", "listing_instances") for filter_ in query.filters)
    assert ("eq", "last_seen_at", "2026-05-29T12:00:00Z") in query.filters


def test_save_linkedin_jobs_canonicalized_caches_candidates_by_provider(monkeypatch):
    calls = []

    def fake_get_canonical_candidates(provider):
        calls.append(provider)
        return []

    inserted_payloads = []

    def fake_save_jobs_to_supabase(payloads):
        inserted_payloads.extend(payloads)

    monkeypatch.setattr(supabase_utils, "get_canonical_candidates", fake_get_canonical_candidates)
    monkeypatch.setattr(supabase_utils, "save_jobs_to_supabase", fake_save_jobs_to_supabase)
    monkeypatch.setattr(supabase_utils, "save_listing_content_version", lambda *args, **kwargs: None)

    jobs = [
        {
            "job_id": "1",
            "provider": "linkedin",
            "company": "Acme",
            "job_title": "TPM",
            "location": "Toronto",
            "description": "desc 1",
        },
        {
            "job_id": "2",
            "provider": "linkedin",
            "company": "Acme",
            "job_title": "TPM 2",
            "location": "Toronto",
            "description": "desc 2",
        },
    ]

    supabase_utils.save_linkedin_jobs_canonicalized(jobs)

    assert calls == ["linkedin"]
    assert len(inserted_payloads) == 2


def test_provider_agnostic_canonical_save_builds_listing_history(monkeypatch):
    saved = []
    monkeypatch.setattr(supabase_utils, "get_canonical_candidates", lambda provider: [])
    monkeypatch.setattr(supabase_utils, "save_jobs_to_supabase", lambda payloads: saved.extend(payloads))

    supabase_utils.save_jobs_canonicalized([{
        "job_id": "career-1",
        "provider": "careers_future",
        "company": "Acme",
        "job_title": "Program Manager",
        "location": "Singapore",
        "posted_at": "2026-08-22",
    }])

    assert saved[0]["seen_count"] == 1
    assert saved[0]["posting_wave_count"] == 1
    assert saved[0]["repost_count"] == 0
    assert saved[0]["listing_instances"][0]["location"] == "Singapore"


def test_existing_id_lookup_includes_historical_listing_instance_ids(monkeypatch):
    query = _RecordingQuery(response_data=[{
        "job_id": "canonical",
        "latest_job_id": "latest",
        "company": "Acme",
        "job_title": "TPM",
        "listing_instances": [{"job_id": "historical"}],
    }])
    monkeypatch.setattr(supabase_utils, "supabase", _FakeSupabase(query))

    ids, _ = supabase_utils.get_existing_jobs_from_supabase()

    assert ids == {"canonical", "latest", "historical"}
    assert "listing_instances" in query.selected
