from supabase import create_client, Client
import config # Import configuration
from typing import Optional, Any, Dict
from models import Resume
import datetime # Import datetime module
import hashlib
import logging # Import logging
import re # Import re for filter pattern matching
import string
import unicodedata
import html
import json
from datetime import date, datetime, timezone, timedelta

# --- Initialize Supabase Client ---
# Ensure URL and Key are provided
if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Supabase URL and Key must be set in environment variables or config.")

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def _collapse_spaces(value: str) -> str:
    return " ".join((value or "").split())


def normalize_title(title: str) -> str:
    value = (title or "").lower()
    value = value.replace("-", " ").replace("/", " ")
    for raw, replacement in getattr(config, "TITLE_NORMALIZATION_REPLACEMENTS", {}).items():
        pattern = rf"\b{re.escape(raw)}\b"
        value = re.sub(pattern, replacement, value)
    value = value.translate(str.maketrans("", "", string.punctuation.replace("&", "")))
    value = value.replace("&", " and ")
    return _collapse_spaces(value)


def normalize_location(location: str) -> str:
    value = (location or "").lower()
    value = value.replace("-", " ").replace("/", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _collapse_spaces(value)


def normalize_company(company: str) -> str:
    value = (company or "").lower()
    value = value.replace("-", " ").replace("/", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _collapse_spaces(value)


def build_canonical_key(provider: str, company: str, title: str, location: str) -> str:
    return "|".join([
        (provider or "").lower().strip(),
        normalize_company(company),
        normalize_title(title),
        normalize_location(location),
    ])


def _normalize_description_for_fingerprint(description: str) -> str:
    value = unicodedata.normalize("NFKD", description or "").lower()
    value = value.replace("’", "").replace("‘", "")
    value = value.replace("-", " ").replace("–", " ").replace("—", " ").replace("•", " ")
    value = value.replace("*", " ").replace("_", " ").replace("`", " ")
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _collapse_spaces(value)


def make_description_fingerprint(description: str) -> str | None:
    normalized = _normalize_description_for_fingerprint(description)
    min_len = getattr(config, "DESCRIPTION_FINGERPRINT_MIN_LENGTH", 500)
    if len(normalized) < min_len:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_role_title(title: str) -> str:
    value = re.sub(r"[()]", " ", title or "")
    trailing_clause = re.compile(r"(?:\s*,\s*|\s+[|@]\s*|\s+[-–—]\s*)[^,|@\-–—]*$")
    match = trailing_clause.search(value)
    if match and len(value[:match.start()].split()) >= 2:
        value = value[:match.start()]
    value = re.sub(r"<[^>]*>", " ", value)
    return normalize_title(html.unescape(value))


def description_token_signature(description: str) -> set[str]:
    value = re.sub(r"<[^>]*>", " ", description or "")
    value = _collapse_spaces(html.unescape(value).lower())
    return {token for token in re.findall(r"[a-z0-9]+", value) if len(token) > 3}


def description_similarity(left: str, right: str) -> float:
    left_tokens = description_token_signature(left)
    right_tokens = description_token_signature(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _date_part(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", str(value).strip())
    return match.group(1) if match else None


def calculate_posting_waves(listing_instances: list[dict]) -> tuple[list[dict], int, int]:
    instances = [dict(instance) for instance in listing_instances]
    if not instances:
        return instances, 0, 0

    locations: dict[str, list[int]] = {}
    for index, instance in enumerate(instances):
        normalized_location = normalize_location(instance.get("location"))
        if normalized_location:
            instance["normalized_location"] = normalized_location
        else:
            instance.pop("normalized_location", None)
        locations.setdefault(normalized_location, []).append(index)

    location_order = sorted(
        locations,
        key=lambda location: min(locations[location]),
    )
    wave_counts = []
    for location_index, location in enumerate(location_order):
        member_indexes = locations[location]
        parent = {index: index for index in member_indexes}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        shared_keys: dict[str, int] = {}
        instance_keys: dict[int, list[str]] = {}
        for member_index in member_indexes:
            instance = instances[member_index]
            posted_date = _date_part(instance.get("posted_at"))
            scrape_run_id = instance.get("scrape_run_id")
            keys = []
            if not location:
                keys.append("unknown_location")
            else:
                if posted_date:
                    keys.append(f"posted:{posted_date}")
                if scrape_run_id:
                    keys.append(f"scrape_run:{scrape_run_id}")
            if location and not keys:
                scraped_date = _date_part(instance.get("scraped_at"))
                keys.append(
                    f"scrape_date:{scraped_date}"
                    if scraped_date
                    else "unknown"
                )
            instance_keys[member_index] = keys
            for key in keys:
                if key in shared_keys:
                    union(shared_keys[key], member_index)
                else:
                    shared_keys[key] = member_index

        components: dict[int, list[int]] = {}
        for member_index in member_indexes:
            components.setdefault(find(member_index), []).append(member_index)
        waves = sorted(
            components.values(),
            key=lambda wave: min(
                str(
                    _date_part(instances[index].get("posted_at"))
                    or instances[index].get("scraped_at")
                    or instances[index].get("scrape_run_id")
                    or "9999-12-31"
                )
                for index in wave
            ),
        )
        wave_counts.append(len(waves))
        for wave_index, wave_members in enumerate(waves, start=1):
            wave_key = min(
                key for index in wave_members for key in instance_keys[index]
            )
            for member_position, member_index in enumerate(wave_members):
                instance = instances[member_index]
                instance["posting_wave_key"] = f"{location}|{wave_key}"
                instance["posting_wave_index"] = wave_index
                if member_position > 0:
                    instance["variant_type"] = "simultaneous_variant"
                elif wave_index > 1:
                    instance["variant_type"] = "repost"
                elif location_index > 0:
                    instance["variant_type"] = "location_variant"
                else:
                    instance["variant_type"] = "original"

    posting_wave_count = max(wave_counts, default=0)
    repost_count = max(posting_wave_count - 1, 0)
    return instances, posting_wave_count, repost_count


def build_listing_instance(job: dict) -> dict:
    job_id = job.get("job_id")
    scraped_at = job.get("scraped_at") or datetime.now(timezone.utc).isoformat()
    return {
        "job_id": str(job_id) if job_id is not None else None,
        "scraped_at": scraped_at,
        "last_seen_at": scraped_at,
        "scrape_run_id": job.get("scrape_run_id"),
        "location": job.get("location"),
        "posted_at": job.get("posted_at"),
        "posted_relative_text": job.get("posted_relative_text"),
        "applicant_count": job.get("applicant_count"),
        "applicant_count_text": job.get("applicant_count_text"),
        "applicant_count_type": job.get("applicant_count_type"),
        "salary_text": job.get("salary_text"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "salary_currency": job.get("salary_currency"),
        "recruiter_name": job.get("recruiter_name"),
        "recruiter_profile_url": job.get("recruiter_profile_url"),
        "recruiter_identifier": job.get("recruiter_identifier"),
        "detail_metadata_checked_at": job.get("detail_metadata_checked_at"),
    }


def prepare_canonical_insert_payload(job: dict) -> dict:
    canonical_key = build_canonical_key(
        job.get("provider"),
        job.get("company"),
        job.get("job_title"),
        job.get("location"),
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    job_id = job.get("job_id")
    payload = dict(job)
    payload["canonical_key"] = canonical_key
    payload["original_job_id"] = str(job_id) if job_id is not None else None
    payload["latest_job_id"] = str(job_id) if job_id is not None else None
    payload["first_seen_at"] = now_iso
    payload["last_seen_at"] = now_iso
    payload["last_seen_posted_at"] = job.get("posted_at")
    listing_instances, posting_wave_count, repost_count = calculate_posting_waves(
        [build_listing_instance(job)]
    )
    payload["seen_count"] = 1
    payload["posting_wave_count"] = posting_wave_count
    payload["repost_count"] = repost_count
    payload["listing_instances"] = listing_instances
    payload["description_fingerprint"] = make_description_fingerprint(job.get("description"))
    return payload


def prepare_repost_update_payload(existing: dict, new_job: dict) -> dict:
    listing_instances = list(existing.get("listing_instances") or [])
    new_job_id = new_job.get("job_id")
    new_job_id = str(new_job_id) if new_job_id is not None else None
    known_listing_ids = {
        str(instance.get("job_id"))
        for instance in listing_instances
        if instance.get("job_id") is not None
    }
    is_new_listing = new_job_id is not None and new_job_id not in known_listing_ids
    if is_new_listing:
        listing_instances.append(build_listing_instance(new_job))
    elif new_job_id is not None:
        for instance in listing_instances:
            if str(instance.get("job_id")) != new_job_id:
                continue
            for field in (
                "location",
                "posted_relative_text",
                "applicant_count",
                "applicant_count_text",
                "applicant_count_type",
                "salary_text",
                "salary_min",
                "salary_max",
                "salary_currency",
                "recruiter_name",
                "recruiter_profile_url",
                "recruiter_identifier",
            ):
                if new_job.get(field) is not None:
                    instance[field] = new_job[field]
            if instance.get("posted_at") is None and new_job.get("posted_at") is not None:
                instance["posted_at"] = new_job["posted_at"]
            instance["last_seen_at"] = datetime.now(timezone.utc).isoformat()
            instance["detail_metadata_checked_at"] = new_job.get("detail_metadata_checked_at")
            break
    distinct_listing_ids = {
        str(instance.get("job_id"))
        for instance in listing_instances
        if instance.get("job_id") is not None
    }
    listing_instances, posting_wave_count, repost_count = calculate_posting_waves(listing_instances)
    payload = {
        "job_id": existing["job_id"],
        "latest_job_id": new_job_id if new_job_id is not None else existing.get("latest_job_id"),
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
        "last_seen_posted_at": new_job.get("posted_at") if new_job.get("posted_at") is not None else existing.get("last_seen_posted_at"),
        "posted_relative_text": new_job.get("posted_relative_text") if new_job.get("posted_relative_text") is not None else existing.get("posted_relative_text"),
        "applicant_count": new_job.get("applicant_count") if new_job.get("applicant_count") is not None else existing.get("applicant_count"),
        "applicant_count_text": new_job.get("applicant_count_text") if new_job.get("applicant_count_text") is not None else existing.get("applicant_count_text"),
        "applicant_count_type": new_job.get("applicant_count_type") if new_job.get("applicant_count_type") is not None else existing.get("applicant_count_type"),
        "salary_text": new_job.get("salary_text") if new_job.get("salary_text") is not None else existing.get("salary_text"),
        "salary_min": new_job.get("salary_min") if new_job.get("salary_min") is not None else existing.get("salary_min"),
        "salary_max": new_job.get("salary_max") if new_job.get("salary_max") is not None else existing.get("salary_max"),
        "salary_currency": new_job.get("salary_currency") if new_job.get("salary_currency") is not None else existing.get("salary_currency"),
        "recruiter_name": new_job.get("recruiter_name") if new_job.get("recruiter_name") is not None else existing.get("recruiter_name"),
        "recruiter_profile_url": new_job.get("recruiter_profile_url") if new_job.get("recruiter_profile_url") is not None else existing.get("recruiter_profile_url"),
        "recruiter_identifier": new_job.get("recruiter_identifier") if new_job.get("recruiter_identifier") is not None else existing.get("recruiter_identifier"),
        "seen_count": len(distinct_listing_ids),
        "posting_wave_count": posting_wave_count,
        "repost_count": repost_count,
        "listing_instances": listing_instances,
        "detail_metadata_checked_at": new_job.get("detail_metadata_checked_at") or existing.get("detail_metadata_checked_at"),
    }
    if is_new_listing and existing.get("job_state") in {"expired", "removed"}:
        payload["is_active"] = True
        payload["job_state"] = "new"
    return payload


def find_canonical_match(job: dict, existing_rows: list[dict]) -> dict | None:
    target_company = normalize_company(job.get("company"))
    target_title = normalize_role_title(job.get("job_title"))
    target_location = normalize_location(job.get("location"))
    target_fp = make_description_fingerprint(job.get("description"))
    target_job_id = str(job.get("job_id")) if job.get("job_id") is not None else None

    ordered_rows = sorted(existing_rows, key=lambda row: str(row.get("job_id") or ""))
    for row in ordered_rows:
        known_ids = {str(row.get("job_id")), str(row.get("latest_job_id"))}
        known_ids.update(
            str(instance.get("job_id"))
            for instance in (row.get("listing_instances") or [])
            if instance.get("job_id") is not None
        )
        if target_job_id and target_job_id in known_ids:
            return row

    matching_bucket = [
        row for row in ordered_rows
        if target_location
        and normalize_company(row.get("company")) == target_company
        and normalize_role_title(row.get("job_title")) == target_title
        and bool(normalize_location(row.get("location")))
        and normalize_location(row.get("location")) == target_location
    ]
    if len(matching_bucket) > 200:
        return None

    for row in matching_bucket:
        same_company = normalize_company(row.get("company")) == target_company
        same_title = normalize_role_title(row.get("job_title")) == target_title
        same_fp = target_fp and row.get("description_fingerprint") == target_fp
        if same_company and same_title and same_fp:
            return row

    for row in matching_bucket:
        same_company = normalize_company(row.get("company")) == target_company
        same_title = normalize_role_title(row.get("job_title")) == target_title
        similarity = description_similarity(job.get("description"), row.get("description"))
        threshold = getattr(config, "REPOST_DESCRIPTION_SIMILARITY_THRESHOLD", 0.90)
        if same_company and same_title and similarity >= threshold:
            return row

    return None


def get_canonical_candidates(provider: str, page_size: int = 1000) -> list[dict]:
    candidates = []
    offset = 0
    while True:
        response = (
            supabase.table(config.SUPABASE_TABLE_NAME)
            .select(
                "job_id, canonical_key, company, job_title, location, description, description_fingerprint, "
                "listing_instances, seen_count, posting_wave_count, repost_count, latest_job_id, last_seen_posted_at, "
                "posted_relative_text, applicant_count, applicant_count_text, applicant_count_type, "
                "salary_text, salary_min, salary_max, salary_currency, recruiter_name, "
                "recruiter_profile_url, recruiter_identifier, detail_metadata_checked_at, "
                "is_active, job_state"
            )
            .eq("provider", provider)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data or []
        candidates.extend(page)
        if len(page) < page_size:
            return candidates
        offset += page_size


def save_jobs_canonicalized(jobs_data: list):
    candidates_cache = {}
    scrape_run_id = datetime.now(timezone.utc).isoformat()
    for raw_job in jobs_data:
        job = dict(raw_job)
        job.setdefault("scrape_run_id", scrape_run_id)
        if not getattr(config, "ENABLE_REPOST_DEDUP", True):
            save_jobs_to_supabase([prepare_canonical_insert_payload(job)])
            continue

        cache_key = job.get("provider")
        candidates = candidates_cache.get(cache_key)
        if candidates is None:
            candidates = get_canonical_candidates(provider=cache_key)
            candidates_cache[cache_key] = candidates
        match = find_canonical_match(job, candidates)

        if match:
            payload = prepare_repost_update_payload(match, job)
            query = (
                supabase.table(config.SUPABASE_TABLE_NAME)
                .update({key: value for key, value in payload.items() if key != "job_id"})
                .eq("job_id", match["job_id"])
                .eq("listing_instances", json.dumps(match.get("listing_instances") or []))
            )
            last_seen_at = match.get("last_seen_at")
            query = query.is_("last_seen_at", None) if last_seen_at is None else query.eq("last_seen_at", last_seen_at)
            response = query.execute()
            if len(response.data or []) != 1:
                raise RuntimeError(
                    f"Concurrent canonical update detected for job_id={match['job_id']}"
                )
            match.update(payload)
        else:
            payload = prepare_canonical_insert_payload(job)
            save_jobs_to_supabase([payload])
            candidates.append(payload)


def save_linkedin_jobs_canonicalized(jobs_data: list):
    save_jobs_canonicalized(jobs_data)

# --- Supabase Functions ---
def get_existing_jobs_from_supabase(batch_size: int = 1000) -> tuple[set, set]:
    """
    Fetches all existing job IDs and company-title pairs from the Supabase 'jobs' table.
    Returns:
        - A set of job_ids
        - A set of 'company|job_title' keys (both lowercased for consistency)
    """
    existing_ids = set()
    existing_company_title_keys = set()
    offset = 0

    try:
        while True:
            response = (
                supabase.table(config.SUPABASE_TABLE_NAME)
                .select("job_id, latest_job_id, company, job_title")
                .range(offset, offset + batch_size - 1)
                .execute()
            )

            data = response.data

            if not data:
                break  # No more data to fetch

            for item in data:
                job_id = item.get("job_id")
                company = item.get("company")
                job_title = item.get("job_title")

                if job_id:
                    existing_ids.add(str(job_id))

                latest_job_id = item.get("latest_job_id")
                if latest_job_id:
                    existing_ids.add(str(latest_job_id))

                if company and job_title:
                    normalized_company = company.strip().lower()
                    normalized_title = job_title.strip().lower()
                    existing_company_title_keys.add((normalized_company, normalized_title))

            offset += batch_size

        print(f"Fetched {len(existing_ids)} job IDs and {len(existing_company_title_keys)} company-title pairs.")

    except Exception as e:
        print(f"Error fetching existing jobs from Supabase: {e}")

    return existing_ids, existing_company_title_keys


def get_incomplete_linkedin_metadata_ids(job_ids: list[str]) -> set[str]:
    if not job_ids:
        return set()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    response = (
        supabase.table(config.SUPABASE_TABLE_NAME)
        .select("job_id, latest_job_id, detail_metadata_checked_at")
        .eq("provider", "linkedin")
        .in_("latest_job_id", [str(job_id) for job_id in job_ids])
        .execute()
    )
    return {
        str(row.get("latest_job_id") or row.get("job_id"))
        for row in (response.data or [])
        if not row.get("detail_metadata_checked_at") or row["detail_metadata_checked_at"] < cutoff
    }


def get_filter_profile(archetype: str | None) -> dict:
    resolved = archetype or config.DEFAULT_ARCHETYPE
    profile = config.ARCHETYPE_CONFIGS.get(resolved)
    if profile is None:
        return config.ARCHETYPE_CONFIGS[config.DEFAULT_ARCHETYPE]
    return profile


def match_filter_reason(job: dict) -> tuple[str | None, bool]:
    title = job.get("job_title") or ""
    company = job.get("company") or ""
    desc = job.get("description") or ""
    profile = get_filter_profile(job.get("archetype"))

    for raw_pattern in profile["company_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(company):
            return f"company:{raw_pattern}", False

    for raw_pattern in profile["title_entry_level_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(title):
            return f"title_entry_level:{raw_pattern}", True

    for raw_pattern in profile["title_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(title):
            return f"title:{raw_pattern}", False

    for raw_pattern in profile["desc_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(desc):
            return f"desc:{raw_pattern}", False

    return None, False

def save_jobs_to_supabase(jobs_data: list):
    """
    Saves or updates a list of job data dictionaries to the Supabase table using upsert.
    This avoids duplicate key errors by updating existing records based on job_id.
    """
    if not jobs_data:
        print("No job data provided to save/update.")
        return

    # Ensure job_id is present and potentially convert to the correct type if needed
    # (Assuming job_id in jobs_data is already the correct string type for your 'text' column)
    allowed_fields = {
        "job_id",
        "company",
        "job_title",
        "level",
        "location",
        "description",
        "status",
        "is_active",
        "application_date",
        "resume_score",
        "notes",
        "scraped_at",
        "last_checked",
        "job_state",
        "resume_score_stage",
        "is_interested",
        "customized_resume_id",
        "provider",
        "posted_at",
        "search_query",
        "archetype",
        "filter_profile",
        "canonical_key",
        "original_job_id",
        "latest_job_id",
        "first_seen_at",
        "last_seen_at",
        "last_seen_posted_at",
        "seen_count",
        "posting_wave_count",
        "repost_count",
        "listing_instances",
        "description_fingerprint",
        "posted_relative_text",
        "applicant_count",
        "applicant_count_text",
        "applicant_count_type",
        "salary_text",
        "salary_min",
        "salary_max",
        "salary_currency",
        "recruiter_name",
        "recruiter_profile_url",
        "recruiter_identifier",
        "detail_metadata_checked_at",
    }
    processed_jobs_data = []
    for job in jobs_data:
        if 'job_id' in job and job['job_id'] is not None:
             # If your Supabase job_id column was numeric, you'd convert here:
             # try:
             #     job['job_id'] = int(job['job_id'])
             #     processed_jobs_data.append(job)
             # except (ValueError, TypeError):
             #     print(f"Warning: Invalid job_id format found: {job.get('job_id')}. Skipping.")
             # Since it's text, just ensure it's a string (it likely already is)
             filtered_job = {key: value for key, value in job.items() if key in allowed_fields}
             filtered_job['job_id'] = str(job['job_id'])
             processed_jobs_data.append(filtered_job)
        else:
            print(f"Warning: Job data missing job_id. Skipping: {job}")


    if not processed_jobs_data:
        print("No valid job data remaining after processing.")
        return

    print(f"Attempting to upsert {len(processed_jobs_data)} jobs to Supabase...")

    try:
        # Use table name from config
        # Use upsert instead of insert. It will insert new rows
        # or update existing rows if a job_id conflict occurs based on the primary key.
        # Ensure 'job_id' is the primary key or has a unique constraint in your Supabase table.
        # By default, supabase-py's upsert updates the row on conflict.
        data, count = supabase.table(config.SUPABASE_TABLE_NAME).upsert(processed_jobs_data).execute()

        # Check the actual response structure from your Supabase client version for upsert
        # It might differ slightly from insert's response structure
        if data and isinstance(data, tuple) and len(data) > 1:
             # The actual data returned might be in data[1] for upsert
             actual_data = data[1]
             print(f"Successfully upserted/updated {len(processed_jobs_data)} jobs. Supabase response count: {count}")
             # You might want to log the actual response data for debugging:
             # print(f"Supabase response data: {actual_data}")
        else:
             # Log raw response if structure is unexpected or for debugging
             print(f"Attempted to upsert {len(processed_jobs_data)} jobs. Supabase response: {data}")

    except Exception as e:
        print(f"Error upserting data to Supabase: {e}")
        # Consider logging the data that failed to upsert for debugging
        # print(f"Failed data: {processed_jobs_data}")


def flag_filtered_jobs() -> int:
    """
    Pre-pass filter that scans ALL jobs where is_filtered=False and flags irrelevant ones.
    Runs against the whole DB (backfill-safe) — jobs already flagged are skipped.

    Filter order (short-circuits on first match per job):
      1. Company blocklist  → is_filtered=True, filter_reason="company:<pattern>"
      2. Entry-level title  → is_filtered=True, is_entry_level_filtered=True, filter_reason="title_entry_level:<pattern>"
      3. Title blocklist    → is_filtered=True, filter_reason="title:<pattern>"
      4. Desc blocklist     → is_filtered=True, filter_reason="desc:<pattern>"

    Returns count of newly flagged jobs.
    """
    batch_size = 1000
    last_job_id = None
    newly_flagged = 0
    select_fields = "job_id, job_title, company, description, archetype"
    fallback_fields = "job_id, job_title, company, description"
    use_fallback_select = False

    try:
        while True:
            fields = fallback_fields if use_fallback_select else select_fields
            try:
                query = (
                    supabase.table(config.SUPABASE_TABLE_NAME)
                    .select(fields)
                    .eq("is_filtered", False)
                    .order("job_id", desc=False)
                )
                if last_job_id is not None:
                    query = query.gt("job_id", last_job_id)
                response = query.range(0, batch_size - 1).execute()
            except Exception as e:
                if (not use_fallback_select) and 'column "archetype" does not exist' in str(e):
                    use_fallback_select = True
                    query = (
                        supabase.table(config.SUPABASE_TABLE_NAME)
                        .select(fallback_fields)
                        .eq("is_filtered", False)
                        .order("job_id", desc=False)
                    )
                    if last_job_id is not None:
                        query = query.gt("job_id", last_job_id)
                    response = query.range(0, batch_size - 1).execute()
                else:
                    raise
            batch = response.data
            if not batch:
                break

            last_job_id = batch[-1].get("job_id")

            for job in batch:
                job_id  = job.get("job_id", "")
                title   = job.get("job_title") or ""
                company = job.get("company") or ""
                reason, entry_level = match_filter_reason(job)

                if reason is not None:
                    update_payload = {
                        "is_filtered": True,
                        "filter_reason": reason,
                        "is_entry_level_filtered": entry_level,
                    }
                    try:
                        supabase.table(config.SUPABASE_TABLE_NAME)\
                                .update(update_payload)\
                                .eq("job_id", job_id)\
                                .execute()
                        newly_flagged += 1
                        logging.info(f"Filtered job {job_id} [{company}] '{title}' — reason: {reason}")
                    except Exception as e:
                        logging.error(f"Failed to update filter flag for job_id {job_id}: {e}")
    except Exception as e:
        logging.error(f"Error during flag_filtered_jobs scan: {e}")

    logging.info(f"flag_filtered_jobs complete. Newly flagged: {newly_flagged}")
    return newly_flagged


def backfill_job_archetypes(
    archetype: str = "software_tpm",
    provider: str = "linkedin",
    filter_profile: str = "software_tpm_v1",
) -> int:
    response = (
        supabase.table(config.SUPABASE_TABLE_NAME)
        .update({
            "archetype": archetype,
            "filter_profile": filter_profile,
        })
        .eq("provider", provider)
        .is_("archetype", None)
        .execute()
    )
    return len(response.data or [])


def clear_removed_aerospace_defense_filter(archetype: str = "software_tpm") -> int:
    response = (
        supabase.table(config.SUPABASE_TABLE_NAME)
        .update({
            "is_filtered": False,
            "filter_reason": None,
            "is_entry_level_filtered": False,
        })
        .eq("filter_reason", r"desc:aerospace.*defense|defense.*aerospace")
        .eq("archetype", archetype)
        .execute()
    )
    return len(response.data or [])


def get_jobs_to_score(limit: int) -> list:
    """
    Fetches jobs from the Supabase 'jobs' table that need scoring.
    Filters by is_active = true, resume_score = null, and is_filtered = false.
    Selects only necessary fields (job_id, job_title, description).
    Orders by scraped_at ascending to process older jobs first.
    """
    if limit <= 0:
        logging.warning("Limit for jobs to score must be positive.")
        return []

    try:
        logging.info(f"Fetching up to {limit} jobs needing scoring...")
        # Select fields needed for scoring
        response = supabase.table(config.SUPABASE_TABLE_NAME)\
                           .select("job_id, job_title, company, description, level")\
                           .eq("is_active", True)\
                           .eq("is_filtered", False)\
                           .is_("resume_score", None)\
                           .order("scraped_at", desc=False)\
                           .limit(limit)\
                           .execute()

        if response.data:
            logging.info(f"Successfully fetched {len(response.data)} jobs to score.")
            return response.data
        else:
            logging.info("No jobs found needing scoring at this time.")
            return []

    except Exception as e:
        logging.error(f"Error fetching jobs to score from Supabase: {e}")
        return []

def get_top_scored_jobs_to_apply(limit: int) -> list:
    """
    Fetches the top-scored jobs from Supabase that are ready for application.
    Filters by is_active = true, resume_score is not null, and status is null.
    Orders by resume_score descending.
    Selects fields needed for the application process.
    """
    if limit <= 0:
        logging.warning("Limit for jobs to apply must be positive.")
        return []

    try:
        logging.info(f"Fetching up to {limit} top-scored jobs to apply for...")
        response = supabase.table(config.SUPABASE_TABLE_NAME)\
                           .select("job_id, job_title, company, resume_score")\
                           .eq("is_active", True)\
                           .eq("status", "new")\
                           .eq("is_filtered", False)\
                           .not_.is_("resume_score", None)\
                           .order("resume_score", desc=True)\
                           .limit(limit)\
                           .execute()

        if response.data:
            logging.info(f"Successfully fetched {len(response.data)} top-scored jobs to apply for.")
            return response.data
        else:
            logging.info("No top-scored jobs found ready for application at this time.")
            return []

    except Exception as e:
        logging.error(f"Error fetching top-scored jobs to apply for from Supabase: {e}")
        return []

def get_top_scored_jobs_for_resume_generation(limit: int) -> list:
    """
    Fetches the top-scored jobs from Supabase using the RPC 'get_top_scored_jobs_custom_sort'.
    p_page_number is set to 1 and p_page_size is set to the limit.
    Selects fields needed for the application process.
    """
    if limit <= 0:
        logging.warning("Limit for jobs to apply must be positive.")
        return []

    try:
        logging.info(f"Fetching up to {limit} top-scored jobs to apply for using RPC 'get_top_scored_jobs_custom_sort'...")
        response = supabase.rpc(
                "get_jobs_for_resume_generation_custom_sort",
                {"p_page_number": 1, "p_page_size": limit}
            ).execute()

        if response.data:
            logging.info(f"Successfully fetched {len(response.data)} top-scored jobs to apply for via RPC.")
            return response.data
        else:
            # Check for RPC specific errors if any, or just log general empty data
            if hasattr(response, 'error') and response.error:
                logging.error(f"Error calling RPC 'get_top_scored_jobs_custom_sort': {response.error.message}")
            else:
                logging.info("No top-scored jobs found ready for application at this time via RPC.")
            return []

    except Exception as e:
        logging.error(f"Error fetching top-scored jobs to apply for from Supabase RPC: {e}")
        return []

def get_jobs_to_rescore(limit: int) -> list:
    """
    Fetches jobs from Supabase that are ready for re-scoring with a custom resume.
    Filters by is_active = true, resume_link is not null, and resume_score_stage = 'initial'.
    Orders by resume_score descending.
    Selects fields needed for the re-scoring process.
    """
    if limit <= 0:
        logging.warning("Limit for jobs to rescore must be positive.")
        return []

    try:
        logging.info(f"Fetching up to {limit} jobs for re-scoring via RPC...")
        # Note: We updated the RPC to also return customized_resume_id
        response = supabase.rpc(
            "get_jobs_for_rescore", 
            {"p_limit_val": limit}   
        ).execute()

        if hasattr(response, 'data') and response.data is not None:
            if response.data: # Check if list is not empty
                logging.info(f"Successfully fetched {len(response.data)} jobs for re-scoring via RPC.")
                return response.data
            else:
                logging.info("No jobs found meeting re-scoring criteria via RPC at this time (empty list returned).")
                return []
        elif hasattr(response, 'error') and response.error: # Handle explicit error attribute
             logging.error(f"Error calling RPC get_jobs_for_rescore: {response.error}")
             return []
        else: # Fallback for unexpected response structure
            logging.warning(f"Unexpected response structure from RPC call: {response}")
            return []


    except Exception as e:
        logging.error(f"Exception calling RPC get_jobs_for_rescore: {e}", exc_info=True)
        return []

def update_job_score(job_id: str, score: int, resume_score_stage: str = "initial") -> bool:
    """
    Updates the 'resume_score' and 'resume_score_stage' for a specific job_id in the Supabase 'jobs' table.
    Returns True on success, False on failure.
    """
    if not job_id or score is None:
        logging.error(f"Invalid input for updating job score: job_id={job_id}, score={score}")
        return False

    if resume_score_stage not in ["initial", "custom"]:
        logging.error(f"Invalid resume_score_stage: {resume_score_stage}. Must be 'initial' or 'custom'.")
        return False

    try:
        logging.info(f"Updating score for job_id {job_id} to {score} and stage to {resume_score_stage}...")
        update_payload = {
            "resume_score": score,
            "resume_score_stage": resume_score_stage
        }
        response = supabase.table(config.SUPABASE_TABLE_NAME)\
                           .update(update_payload)\
                           .eq("job_id", job_id)\
                           .execute()

        # Check if the update was successful (response structure might vary)
        # A common pattern is checking if data is returned or count is non-zero
        if hasattr(response, 'data') and response.data:
             logging.info(f"Successfully updated score for job_id {job_id}.")
             return True
        elif hasattr(response, 'count') and response.count is not None and response.count > 0:
             logging.info(f"Successfully updated score for job_id {job_id} (count={response.count}).")
             return True
        elif not hasattr(response, 'data') and not hasattr(response, 'count'):
             # Handle cases where the response might not have data/count but didn't error
             logging.warning(f"Update score for job_id {job_id} executed, but response structure unclear: {response}")
             return True # Assume success if no exception occurred
        else:
             logging.warning(f"Update score for job_id {job_id} might have failed or job not found. Response: {response}")
             return False


    except Exception as e:
        logging.error(f"Error updating score for job_id {job_id} in Supabase: {e}")
        return False

def get_job_by_id(job_id: str) -> dict | None:
    """
    Fetches a single job record from the Supabase 'jobs' table based on job_id.
    """
    if not job_id:
        logging.error("No job_id provided to fetch job details.")
        return None
    if not hasattr(config, 'SUPABASE_TABLE_NAME') or not config.SUPABASE_TABLE_NAME:
        logging.error("SUPABASE_TABLE_NAME is not defined in config.py")
        return None

    try:
        logging.info(f"Fetching job details for job_id: {job_id} from table '{config.SUPABASE_TABLE_NAME}'")
        response = supabase.table(config.SUPABASE_TABLE_NAME)\
                           .select("company, job_title, level, description")\
                           .eq("job_id", job_id) \
                           .limit(1)\
                           .execute() # Assuming 'job_id' is the column name

        if response.data:
            logging.info(f"Successfully fetched job data for job_id: {job_id}.")
            return response.data[0] # Return the first matching job
        else:
            logging.warning(f"No job found for job_id: {job_id}")
            return None

    except Exception as e:
        logging.error(f"Error fetching job data from Supabase for job_id {job_id}: {e}")
        return None

def upload_customized_resume_to_storage(file_content: bytes, destination_path: str) -> Optional[str]:
    """
    Uploads the generated resume PDF (as bytes) to Supabase Storage.

    Args:
        file_content: The resume content in bytes.
        destination_path: The desired path and filename within the bucket
                          (e.g., "personalized_resumes/resume_job_12345.pdf").
                          Ensure this path is unique per job/resume.

    Returns:
        The destination path of the uploaded file, or None if upload fails.
    """
    if not file_content:
        logging.error("Cannot upload empty file content.")
        return None
    if not config.SUPABASE_STORAGE_BUCKET:
        logging.error("Supabase storage bucket name not configured.")
        return None

    try:
        logging.info(f"Uploading resume to Supabase Storage at path: {destination_path}")

        # Use upsert=True if you want to overwrite if a file with the same name exists,
        # otherwise False (or omit) to potentially get an error if it exists.
        # Ensure your destination_path includes job_id or similar for uniqueness.
        upload_response = supabase.storage.from_(config.SUPABASE_STORAGE_BUCKET)\
            .upload(
                path=destination_path,
                file=file_content,
                file_options={"content-type": "application/pdf", "upsert": "true"} # Set upsert based on desired behavior
            )

        logging.info(f"Successfully uploaded resume to path: {destination_path}")
        return destination_path

    except Exception as e:
        # Supabase client might raise specific exceptions, catch broadly for now
        logging.error(f"Error uploading file to Supabase Storage: {e}")
        # Attempt to remove partially uploaded file if possible/needed (more complex error handling)
        # try:
        #     supabase.storage.from_(config.SUPABASE_STORAGE_BUCKET).remove([destination_path])
        # except:
        #     logging.warning(f"Could not clean up potentially failed upload at {destination_path}")
        return None

def update_job_with_resume_link(job_id: str, customized_resume_id: str,  new_status: Optional[str] = "resume_generated") -> bool:
    """
    Updates the job record in the Supabase table with the resume link and optionally a new status.

    Args:
        job_id: The unique ID of the job to update.
        customized_resume_id: The id the generated resume in Supabase customized_resumes table.
        new_status: The status to set for the job after processing (e.g., 'resume_generated').
                    Set to None to only update the link without changing status.

    Returns:
        True if the update was successful, False otherwise.
    """
    if not job_id or not customized_resume_id:
        logging.error("Job ID and Customized Resume id are required for updating the job.")
        return False

    try:
        update_data = {"customized_resume_id": customized_resume_id}
        # if new_status:
        #     update_data["job_state"] = new_status # Assuming 'status' is your column name

        logging.info(f"Updating job {job_id} with resume link, resume id and status '{new_status or 'unchanged'}'...")

        response = supabase.table(config.SUPABASE_TABLE_NAME)\
                           .update(update_data)\
                           .eq("job_id", job_id)\
                           .execute()

        # Check if the update affected any rows (response.data might contain updated rows)
        if response.data:
            logging.info(f"Successfully updated job {job_id}.")
            return True
        else:
            # This might happen if the job_id didn't exist or matched 0 rows
            logging.warning(f"Update query executed for job {job_id}, but no rows seemed to be affected.")
            # Depending on strictness, you might return False here
            return False # Treat as failure if no row was confirmed updated

    except Exception as e:
        logging.error(f"Error updating job {job_id} in Supabase: {e}")
        return False

def save_customized_resume(resume_data: 'Resume', resume_path: str) -> Optional[Any]: # Return type changed
    """
    Saves a customized resume to the Supabase 'customized_resumes' table.

    Args:
        resume_data: A Resume object (Pydantic model) containing the resume details.
        resume_path: The path of the uploaded resume in storage.

    Returns:
        The ID (typically string UUID or integer) of the inserted resume if successful, None otherwise.
    """

    if not resume_path:
        logging.error("Resume Path is required for saving the resume.")
        return False

    if not resume_data:
        logging.error("No resume data provided to save.")
        return None

    if not hasattr(config, 'SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME') or \
       not config.SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME:
        logging.error("SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME is not defined in config.py")
        return None

    try:
        # Convert Pydantic model to dict for Supabase
        if hasattr(resume_data, 'model_dump'):
            data_to_insert = resume_data.model_dump(exclude_none=True)
        else:
            data_to_insert = resume_data.dict(exclude_none=True)

        data_to_insert['resume_link'] = resume_path

        logging.info(
            f"Saving customized resume for email: {getattr(resume_data, 'email', 'N/A')} "
            f"with path '{resume_path}' to table '{config.SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME}'"
        )

        response = supabase.table(config.SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME)\
                           .insert(data_to_insert)\
                           .execute()

        if response.data and len(response.data) > 0:
            inserted_record = response.data[0]
            if 'id' in inserted_record:
                resume_id = inserted_record['id']
                logging.info(
                    f"Successfully saved customized resume for {getattr(resume_data, 'email', 'N/A')} "
                    f"with ID: {resume_id}."
                )
                return resume_id
            else:
                logging.warning(
                    f"Customized resume for {getattr(resume_data, 'email', 'N/A')} saved, "
                    f"but 'id' key not found in the response data. Full record: {inserted_record}"
                )
                return None
        else:
            error_message = "Unknown error"
            if hasattr(response, 'error') and response.error:
                error_message = response.error
                logging.error(
                    f"Failed to save customized resume for {getattr(resume_data, 'email', 'N/A')}. "
                    f"Supabase Error: {error_message}"
                )
            elif hasattr(response, 'message') and response.message:
                error_message = response.message
                logging.error(
                    f"Failed to save customized resume for {getattr(resume_data, 'email', 'N/A')}. "
                    f"Supabase API Error: {error_message}"
                )
            else:
                logging.warning(
                    f"Customized resume for {getattr(resume_data, 'email', 'N/A')} might not have been saved "
                    f"or ID not returned. Response data is empty or missing. Response: {response}"
                )
            return None

    except Exception as e:
        logging.error(
            f"Error saving customized resume for {getattr(resume_data, 'email', 'N/A')} to Supabase: {e}",
            exc_info=True
        )
        return None

def get_customized_resume(resume_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a customized resume record from Supabase by ID.
    """
    if not resume_id:
        return None
    
    try:
        logging.info(f"Fetching customized resume data from database for ID: {resume_id}")
        response = supabase.table(config.SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME)\
            .select("*")\
            .eq("id", resume_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logging.error(f"Error fetching customized resume {resume_id}: {e}")
        return None


# --- Base Resume Functions ---
# These functions handle storing and retrieving the user's base resume
# securely via Supabase, instead of committing sensitive files to the repo.

def download_resume_from_storage(file_name: str = "resume.pdf") -> Optional[bytes]:
    """
    Downloads the user's resume PDF from the 'resumes' Supabase Storage bucket.

    Args:
        file_name: The name of the resume file in the storage bucket.

    Returns:
        The file content as bytes, or None if download fails.
    """
    bucket_name = config.SUPABASE_RESUME_STORAGE_BUCKET
    if not bucket_name:
        logging.error("Resume storage bucket name not configured (SUPABASE_RESUME_STORAGE_BUCKET).")
        return None

    try:
        logging.info(f"Downloading '{file_name}' from Supabase Storage bucket '{bucket_name}'...")
        file_bytes = supabase.storage.from_(bucket_name).download(file_name)

        if file_bytes:
            logging.info(f"Successfully downloaded '{file_name}' ({len(file_bytes)} bytes).")
            return file_bytes
        else:
            logging.warning(f"Downloaded empty content for '{file_name}' from bucket '{bucket_name}'.")
            return None

    except Exception as e:
        logging.error(f"Error downloading '{file_name}' from Supabase Storage: {e}")
        return None


def save_base_resume(resume_data: dict) -> bool:
    """
    Saves (upserts) the parsed base resume JSON to the 'base_resume' table.
    Deletes any existing rows first to ensure only one base resume exists.

    Args:
        resume_data: The parsed resume data as a dictionary.

    Returns:
        True if saved successfully, False otherwise.
    """
    if not resume_data:
        logging.error("No resume data provided to save.")
        return False

    table_name = config.SUPABASE_BASE_RESUME_TABLE_NAME
    try:
        # Delete any existing base resume rows (there should only be one)
        logging.info(f"Clearing existing base resume data from '{table_name}'...")
        supabase.table(table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

        # Insert the new base resume
        logging.info(f"Saving parsed base resume to '{table_name}'...")
        response = supabase.table(table_name).insert({
            "resume_data": resume_data
        }).execute()

        if response.data and len(response.data) > 0:
            logging.info(f"Successfully saved base resume to '{table_name}'.")
            return True
        else:
            logging.warning(f"Base resume insert returned no data. Response: {response}")
            return False

    except Exception as e:
        logging.error(f"Error saving base resume to Supabase: {e}", exc_info=True)
        return False


def get_base_resume() -> Optional[dict]:
    """
    Fetches the base resume JSON data from the 'base_resume' table.

    Returns:
        The resume data as a dictionary, or None if not found or on error.
    """
    table_name = config.SUPABASE_BASE_RESUME_TABLE_NAME
    try:
        logging.info(f"Fetching base resume from '{table_name}'...")
        response = supabase.table(table_name)\
            .select("resume_data")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        if response.data and len(response.data) > 0:
            resume_data = response.data[0].get("resume_data")
            if resume_data:
                logging.info("Successfully fetched base resume data from Supabase.")
                return resume_data
            else:
                logging.warning("Base resume row found but 'resume_data' is empty.")
                return None
        else:
            logging.warning("No base resume found in Supabase. Please run the 'Parse Resume' workflow first.")
            return None

    except Exception as e:
        logging.error(f"Error fetching base resume from Supabase: {e}", exc_info=True)
        return None
