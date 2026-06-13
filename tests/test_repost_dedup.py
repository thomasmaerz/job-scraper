import supabase_utils


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _RecordingQuery:
    def __init__(self, response_data=None):
        self.response_data = response_data or []
        self.selected = None
        self.filters = []
        self.upsert_payloads = []

    def select(self, fields):
        self.selected = fields
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def gte(self, field, value):
        self.filters.append(("gte", field, value))
        return self

    def upsert(self, payload):
        self.upsert_payloads.append(payload)
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
    assert payload["repost_count"] == 0
    assert payload["listing_instances"][0]["job_id"] == "4426608777"


def test_prepare_repost_update_payload():
    existing = {
        "job_id": "4394716706",
        "listing_instances": [{"job_id": "4394716706", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
    }
    new_job = {
        "job_id": "4426608777",
        "posted_at": "2026-06-12",
        "posted_relative_text": "18 hours ago",
        "applicant_count": 26,
        "salary_text": "$120,000-$135,000 CAD",
    }

    update = supabase_utils.prepare_repost_update_payload(existing, new_job)

    assert update["latest_job_id"] == "4426608777"
    assert update["seen_count"] == 2
    assert update["repost_count"] == 1
    assert len(update["listing_instances"]) == 2


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


def test_find_canonical_match_prefers_existing_role():
    existing_rows = [{
        "job_id": "4394716706",
        "canonical_key": "linkedin|chandos construction|industrial construction senior project manager|chalk river ontario canada",
        "company": "Chandos Construction",
        "job_title": "Industrial Construction - Senior Project Manager",
        "location": "Chalk River, Ontario, Canada",
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


def test_get_recent_canonical_candidates_selects_fields_needed_for_partial_repost_updates(monkeypatch):
    query = _RecordingQuery(response_data=[])
    monkeypatch.setattr(supabase_utils, "supabase", _FakeSupabase(query))

    supabase_utils.get_recent_canonical_candidates(
        provider="linkedin",
        company="Chandos Construction",
        days=30,
    )

    assert query.selected == (
        "job_id, canonical_key, company, job_title, location, description_fingerprint, "
        "listing_instances, seen_count, repost_count, latest_job_id, last_seen_posted_at, "
        "posted_relative_text, applicant_count, salary_text, salary_min, salary_max, "
        "salary_currency, recruiter_name, recruiter_profile_url, recruiter_identifier"
    )


def test_save_linkedin_jobs_canonicalized_matches_repost_across_normalized_company_variants(monkeypatch):
    existing = {
        "job_id": "4394716706",
        "canonical_key": "linkedin|foo bar|senior project manager|toronto ontario canada",
        "company": "Foo Bar",
        "job_title": "Senior Project Manager",
        "location": "Toronto, Ontario, Canada",
        "description_fingerprint": supabase_utils.make_description_fingerprint(
            "We are Chandos. Inclusion, collaboration, innovation. " * 4
        ),
        "listing_instances": [{"job_id": "4394716706", "scraped_at": "2026-05-29T12:00:00Z"}],
        "seen_count": 1,
        "repost_count": 0,
        "latest_job_id": "4394716706",
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
    assert query.upsert_payloads
    assert query.upsert_payloads[0][0]["job_id"] == "4394716706"
    assert query.upsert_payloads[0][0]["latest_job_id"] == "4426608777"
    assert query.upsert_payloads[0][0]["salary_text"] == "$120,000-$135,000 CAD"
