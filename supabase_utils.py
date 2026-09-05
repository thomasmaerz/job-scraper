from supabase import create_client, Client
from postgrest.types import ReturnMethod
import httpx
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
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import date, datetime, timezone, timedelta
import relist_tracking
import freehire_compat
from lane_catalog import CANONICAL_LANE_SLUGS, LANE_ALIASES, canonical_lane_slug


# jobs.location_scope is generated from jobs.location. Configured search
# geography belongs only in transient/query provenance.
TRANSIENT_JOB_FIELDS = frozenset({
    "query_scope",
    "search_location_scope",
    "geography_id",
    "lane",
    "search_query_id",
    "search_query_type",
    "search_query_language",
})
GENERATED_JOB_FIELDS = frozenset({
    "location_province_code",
    "location_scope",
    "location_metro",
    "listing_location_province_codes",
    "listing_location_scopes",
})
JOB_WRITE_FIELDS = frozenset({
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
    "same_id_relist_count",
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
    "freehire_category",
    "freehire_seniority",
    "is_remote",
    "freehire_remote_evidence",
    "freehire_compat_status",
    "freehire_compat_input_hash",
    "freehire_compat_import_hash",
    "freehire_compat_model",
    "freehire_compat_prompt_version",
    "freehire_compat_schema_version",
    "freehire_compat_confidence",
    "freehire_compat_classified_at",
    "freehire_compat_error",
    "freehire_compat_attempts",
    "freehire_compat_claimed_at",
    "freehire_compat_claimed_by",
    "freehire_compat_next_retry_at",
    "freehire_compat_provenance",
})


def _canonical_job_write_payload(job: Mapping[str, Any]) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in TRANSIENT_JOB_FIELDS and key not in GENERATED_JOB_FIELDS
    }

# --- Initialize Supabase Client ---
# Ensure URL and Key are provided
if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Supabase URL and Key must be set in environment variables or config.")

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def _collect_job_identifiers(value: Any, seen: set[int] | None = None) -> list[str]:
    """Extract scalar job IDs from job records and Supabase response wrappers."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (str, int)):
        identifier = str(value).strip()
        return [identifier] if identifier else []

    seen = seen or set()
    value_identity = id(value)
    if value_identity in seen:
        return []
    seen.add(value_identity)

    if hasattr(value, "data") and not isinstance(value, Mapping):
        return _collect_job_identifiers(value.data, seen)

    if isinstance(value, Mapping):
        # Row identifiers take precedence over response-envelope fields. Some
        # legacy saver results wrapped a complete row in their ``id`` field.
        for key in ("job_id", "id"):
            if key in value:
                identifiers = _collect_job_identifiers(value[key], seen)
                if identifiers:
                    return identifiers
        for key in ("data", "result"):
            if key in value:
                identifiers = _collect_job_identifiers(value[key], seen)
                if identifiers:
                    return identifiers
        return []

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Older postgrest clients exposed ``(data, count)``. A numeric count is
        # metadata, not another job identifier.
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (int, type(None))):
            return _collect_job_identifiers(value[0], seen)
        identifiers = []
        for item in value:
            identifiers.extend(_collect_job_identifiers(item, seen))
        return identifiers
    return []


def extract_job_identifiers(value: Any) -> list[str]:
    """Return unique, ordered scalar job IDs from a saver/Supabase result."""
    return list(dict.fromkeys(_collect_job_identifiers(value)))


def normalize_job_identifier(value: Any) -> str | None:
    """Normalize a singular job ID/result to a stable string, or reject it."""
    identifiers = extract_job_identifiers(value)
    return identifiers[0] if len(identifiers) == 1 else None


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


def combine_listing_locations(listing_instances: list[dict], fallback: str | None = None) -> str | None:
    """Return stable display/search text containing every observed location."""
    by_normalized = {}
    fallback_values = re.split(r"\s*;\s*", fallback or "")
    for value in [*fallback_values, *(instance.get("location") for instance in listing_instances)]:
        normalized = normalize_location(value)
        if normalized and normalized not in by_normalized:
            by_normalized[normalized] = _collapse_spaces(value)
    if not by_normalized:
        return None
    return "; ".join(by_normalized[key] for key in sorted(by_normalized))


def normalized_listing_locations(row: dict) -> set[str]:
    values = {
        normalize_location(instance.get("location"))
        for instance in (row.get("listing_instances") or [])
        if isinstance(instance, dict)
    }
    values.update(
        normalize_location(value)
        for value in re.split(r"\s*;\s*", row.get("location") or "")
    )
    values.discard("")
    return values


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


def make_description_content_hash(description: str | None) -> str | None:
    return relist_tracking.make_content_hash(description)


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


def has_matching_body_hash(left: dict, right: dict) -> bool:
    """Require the same sufficiently long normalized description body hash."""
    left_fp = left.get("description_fingerprint") or make_description_fingerprint(left.get("description"))
    right_fp = right.get("description_fingerprint") or make_description_fingerprint(right.get("description"))
    return bool(left_fp and left_fp == right_fp)


def extract_explicit_description_title(description: str) -> str | None:
    match = re.search(
        r"(?:^|\n)\s*(?:\*\*)?job\s+title\s*:\s*([^\n*]+)",
        description or "",
        flags=re.IGNORECASE,
    )
    return _collapse_spaces(match.group(1)) if match else None


TITLE_FUZZY_NOISE_TOKENS = {
    "canada", "remote", "hybrid", "onsite", "on", "site", "en", "fr",
}


def normalize_title_for_similarity(title: str) -> str:
    value = normalize_title(title)
    value = re.sub(r"\bpm\b", "project manager", value)
    tokens = [token for token in value.split() if token not in TITLE_FUZZY_NOISE_TOKENS]
    return " ".join(tokens)


def title_similarity(left: str, right: str) -> float:
    left_value = normalize_title_for_similarity(left)
    right_value = normalize_title_for_similarity(right)
    if not left_value or not right_value:
        return 0.0
    left_tokens = set(left_value.split())
    right_tokens = set(right_value.split())
    token_overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    sequence = SequenceMatcher(None, left_value, right_value).ratio()
    return max(token_overlap, sequence)


def is_high_confidence_repost_match(left: dict, right: dict) -> tuple[bool, str | None, float]:
    """Allow fuzzy matching only within one role/location; otherwise require identity."""
    if normalize_company(left.get("company")) != normalize_company(right.get("company")):
        return False, None, 0.0

    same_role = normalize_role_title(left.get("job_title")) == normalize_role_title(right.get("job_title"))
    same_title = normalize_title(left.get("job_title")) == normalize_title(right.get("job_title"))
    left_locations = normalized_listing_locations(left)
    right_locations = normalized_listing_locations(right)
    if not left_locations or not right_locations:
        return False, None, 0.0
    same_location = bool(left_locations & right_locations)
    similarity = description_similarity(left.get("description"), right.get("description"))
    if has_matching_body_hash(left, right):
        fuzzy_title = title_similarity(left.get("job_title"), right.get("job_title"))
        title_threshold = getattr(config, "REPOST_TITLE_SIMILARITY_THRESHOLD", 0.70)
        if not same_title and fuzzy_title < title_threshold:
            return False, None, similarity
        method = "exact_fingerprint" if same_title and same_location else "body_hash_fuzzy_title"
        return True, method, similarity

    threshold = getattr(config, "REPOST_DESCRIPTION_SIMILARITY_THRESHOLD", 0.90)
    if same_role and same_location and similarity >= threshold:
        return True, "fuzzy_description", similarity
    return False, None, similarity


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
    job_id = normalize_job_identifier(job.get("job_id"))
    payload = _canonical_job_write_payload(job)
    payload["job_id"] = job_id
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
    if job.get("provider") == "linkedin":
        is_remote, evidence = freehire_compat.classify_remote(payload)
        payload["is_remote"] = is_remote
        payload["freehire_remote_evidence"] = evidence
        payload["freehire_compat_input_hash"] = freehire_compat.compute_classification_hash(payload)
        payload["freehire_compat_import_hash"] = freehire_compat.compute_import_hash(payload, is_remote=is_remote)
        payload["freehire_compat_status"] = "pending"
        payload["freehire_compat_schema_version"] = config.FREEHIRE_COMPAT_SCHEMA_VERSION
    return payload


def prepare_repost_update_payload(existing: dict, new_job: dict) -> dict:
    listing_instances = [
        dict(instance) for instance in (existing.get("listing_instances") or [])
    ]
    new_job_id = new_job.get("job_id")
    new_job_id = str(new_job_id) if new_job_id is not None else None
    known_listing_ids = {
        str(instance.get("job_id"))
        for instance in listing_instances
        if instance.get("job_id") is not None
    }
    is_new_listing = new_job_id is not None and new_job_id not in known_listing_ids
    relist_date = _date_part(new_job.get("same_id_relist_date") or new_job.get("posted_at"))
    accepted_same_id_relist = bool(
        new_job.get("same_id_relist_candidate")
        and new_job_id in known_listing_ids
        and relist_date
        and not any(
            str(instance.get("job_id")) == new_job_id
            and _date_part(instance.get("posted_at")) == relist_date
            for instance in listing_instances
        )
    )
    if is_new_listing or accepted_same_id_relist:
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
        "search_query": new_job.get("search_query") or existing.get("search_query"),
        "archetype": new_job.get("archetype") or existing.get("archetype"),
        "filter_profile": new_job.get("filter_profile") or existing.get("filter_profile"),
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
        "location": combine_listing_locations(
            listing_instances,
            f"{existing.get('location') or ''}; {new_job.get('location') or ''}",
        ),
        "detail_metadata_checked_at": new_job.get("detail_metadata_checked_at") or existing.get("detail_metadata_checked_at"),
        "same_id_relist_count": int(existing.get("same_id_relist_count") or 0) + int(accepted_same_id_relist),
    }
    description = new_job.get("description")
    if description is not None:
        content_hash = make_description_content_hash(description)
        if content_hash and content_hash != make_description_content_hash(existing.get("description")):
            payload["description"] = description
            payload["description_fingerprint"] = make_description_fingerprint(description)
    if (is_new_listing or accepted_same_id_relist) and existing.get("job_state") in {"expired", "removed"}:
        payload["is_active"] = True
        payload["job_state"] = "new"
    if existing.get("provider") == "linkedin" or new_job.get("provider") == "linkedin":
        latest_values = {key: value for key, value in new_job.items() if value is not None}
        merged = {**existing, **latest_values, **payload, "job_id": existing["job_id"]}
        input_hash = freehire_compat.compute_classification_hash(merged)
        is_remote, evidence = freehire_compat.classify_remote(merged)
        payload["is_remote"] = is_remote
        payload["freehire_remote_evidence"] = evidence
        payload["freehire_compat_import_hash"] = freehire_compat.compute_import_hash(
            merged, is_remote=is_remote
        )
        if input_hash != existing.get("freehire_compat_input_hash"):
            payload["freehire_compat_input_hash"] = input_hash
            payload["freehire_compat_status"] = "pending"
            payload["freehire_compat_schema_version"] = config.FREEHIRE_COMPAT_SCHEMA_VERSION
            payload["freehire_compat_error"] = None
    return payload


def find_canonical_match(job: dict, existing_rows: list[dict]) -> dict | None:
    target_company = normalize_company(job.get("company"))
    target_title = normalize_role_title(job.get("job_title"))
    target_location = normalize_location(job.get("location"))
    target_fp = make_description_fingerprint(job.get("description"))
    target_job_id = normalize_job_identifier(job.get("job_id"))

    ordered_rows = sorted(existing_rows, key=lambda row: str(row.get("job_id") or ""))
    for row in ordered_rows:
        known_ids = {
            identifier
            for identifier in (
                normalize_job_identifier(row.get("job_id")),
                normalize_job_identifier(row.get("latest_job_id")),
            )
            if identifier is not None
        }
        known_ids.update(
            identifier
            for instance in (row.get("listing_instances") or [])
            if isinstance(instance, Mapping)
            and (identifier := normalize_job_identifier(instance.get("job_id"))) is not None
        )
        if target_job_id and target_job_id in known_ids:
            return row

    matching_bucket = [
        row for row in ordered_rows
        if normalize_company(row.get("company")) == target_company
    ]
    if len(matching_bucket) > 200:
        return None

    for row in matching_bucket:
        matched, _, _ = is_high_confidence_repost_match(job, row)
        if matched:
            return row

    return None


def get_canonical_candidates(provider: str, page_size: int = 1000) -> list[dict]:
    candidates = []
    offset = 0
    while True:
        response = (
            supabase.table(config.SUPABASE_TABLE_NAME)
            .select(
                "job_id, canonical_revision, canonical_key, company, job_title, location, description, description_fingerprint, "
                "listing_instances, seen_count, posting_wave_count, repost_count, latest_job_id, last_seen_at, last_seen_posted_at, "
                "posted_relative_text, applicant_count, applicant_count_text, applicant_count_type, "
                "salary_text, salary_min, salary_max, salary_currency, recruiter_name, "
                "recruiter_profile_url, recruiter_identifier, detail_metadata_checked_at, "
                "is_active, job_state, same_id_relist_count, provider, level, "
                "search_query, archetype, filter_profile, location_scope, "
                "freehire_category, freehire_seniority, is_remote, freehire_remote_evidence, "
                "freehire_compat_status, freehire_compat_input_hash, freehire_compat_import_hash"
            )
            .eq("provider", provider)
            .order("job_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data or []
        candidates.extend(page)
        if len(page) < page_size:
            return candidates
        offset += page_size


def _membership_provenance(job: dict, archetype: str) -> dict:
    query_scope = job.get("query_scope")
    if isinstance(query_scope, str):
        try:
            query_scope = json.loads(query_scope)
        except (TypeError, json.JSONDecodeError):
            query_scope = {"raw": query_scope}
    if not isinstance(query_scope, dict):
        query_scope = {}
    query_scope = {
        **query_scope,
        "lane": archetype,
        "query_id": job.get("search_query_id") or query_scope.get("query_id"),
        "query_type": job.get("search_query_type") or query_scope.get("query_kind"),
        "query": job.get("search_query") or query_scope.get("search_query"),
        "language": job.get("search_query_language") or query_scope.get("language"),
        "location_scope": (
            query_scope.get("location_scope")
            or job.get("search_location_scope")
            or query_scope.get("search_location_scope")
        ),
        "geography_id": job.get("geography_id") or query_scope.get("geography_id"),
    }
    return {key: value for key, value in query_scope.items() if value is not None}


def _membership_archetypes(job: dict) -> list[str]:
    raw_archetype = job.get("lane") or job.get("archetype")
    if raw_archetype:
        archetype = canonical_lane_slug(raw_archetype)
        if archetype not in CANONICAL_LANE_SLUGS:
            raise ValueError(f"Unknown membership archetype '{raw_archetype}'")
        return [archetype]
    # Migration compatibility: preserve legacy lane representation even when a
    # historical caller omitted explicit provenance.
    if job.get("provider") == "linkedin":
        return [canonical_lane_slug(config.DEFAULT_ARCHETYPE)]
    return []


def _upsert_single_job_archetype_membership(
    canonical_job_id: str,
    archetype: str,
    job: dict,
) -> None:
    provenance = _membership_provenance(job, archetype)
    rpc_args = {
        "p_job_id": canonical_job_id,
        "p_archetype": archetype,
        "p_query_scope": provenance,
        "p_query_id": provenance.get("query_id"),
        "p_query": provenance.get("query"),
        "p_query_type": provenance.get("query_type"),
        "p_language": provenance.get("language"),
        "p_location_scope": provenance.get("location_scope"),
        "p_geography_id": provenance.get("geography_id"),
    }
    observed_at = _valid_membership_observed_at(job, provenance)
    if observed_at is not None:
        rpc_args["p_first_matched_at"] = observed_at
        rpc_args["p_last_matched_at"] = observed_at
    supabase.rpc("record_job_archetype_membership", rpc_args).execute()


def _valid_membership_observed_at(job: dict, provenance: dict) -> str | None:
    """Return a valid source observation timestamp; otherwise use RPC defaults."""
    for value in (
        job.get("membership_observed_at"),
        job.get("observed_at"),
        provenance.get("observed_at"),
        job.get("detail_metadata_checked_at"),
        job.get("scraped_at"),
    ):
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            continue
        return parsed.isoformat()
    return None


def upsert_job_archetype_membership(canonical_job_id: str, job: dict) -> None:
    """Union scrape provenance into every canonical job/lane membership row."""
    for archetype in _membership_archetypes(job):
        _upsert_single_job_archetype_membership(canonical_job_id, archetype, job)


@dataclass(frozen=True)
class CanonicalSaveResult:
    """Canonical persistence result without implying that IDs preserve input order."""

    canonical_ids: list[str]
    canonical_by_source: dict[str, str]
    canonical_ids_by_input: list[str | None]


def _candidate_source_ids(candidate: Mapping[str, Any]) -> set[str]:
    source_ids = {
        normalized
        for value in (candidate.get("job_id"), candidate.get("latest_job_id"))
        if (normalized := normalize_job_identifier(value)) is not None
    }
    source_ids.update(
        identifier
        for instance in (candidate.get("listing_instances") or [])
        if isinstance(instance, Mapping)
        and (identifier := normalize_job_identifier(instance.get("job_id"))) is not None
    )
    return source_ids


@dataclass
class CanonicalRunContext:
    """Run-scoped canonical snapshots and indexes with write-through updates."""

    candidates_by_provider: dict[str, list[dict]] = field(default_factory=dict)
    existing_job_ids_by_provider: dict[str, set[str]] = field(default_factory=dict)
    company_title_keys_by_provider: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    canonical_by_source_by_provider: dict[str, dict[str, str]] = field(default_factory=dict)
    _load_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def candidates_for(self, provider: str) -> list[dict]:
        if provider not in self.candidates_by_provider:
            with self._load_lock:
                if provider not in self.candidates_by_provider:
                    candidates = get_canonical_candidates(provider=provider)
                    existing_job_ids: set[str] = set()
                    company_title_keys: set[tuple[str, str]] = set()
                    canonical_by_source: dict[str, str] = {}
                    for candidate in candidates:
                        canonical_id = normalize_job_identifier(candidate.get("job_id"))
                        source_ids = _candidate_source_ids(candidate)
                        existing_job_ids.update(source_ids)
                        if canonical_id is not None:
                            canonical_by_source.update(
                                {source_id: canonical_id for source_id in source_ids}
                            )
                        company = candidate.get("company")
                        job_title = candidate.get("job_title")
                        if company and job_title:
                            company_title_keys.add(
                                (company.strip().lower(), job_title.strip().lower())
                            )
                    self.existing_job_ids_by_provider[provider] = existing_job_ids
                    self.company_title_keys_by_provider[provider] = company_title_keys
                    self.canonical_by_source_by_provider[provider] = canonical_by_source
                    # Publish last so racing readers cannot observe an
                    # initialized snapshot with incomplete derived indexes.
                    self.candidates_by_provider[provider] = candidates
        return self.candidates_by_provider[provider]

    def existing_indexes(self, provider: str) -> tuple[set[str], set[tuple[str, str]]]:
        self.candidates_for(provider)
        return (
            self.existing_job_ids_by_provider[provider],
            self.company_title_keys_by_provider[provider],
        )

    def canonical_ids_for_sources(self, provider: str, source_job_ids: list[str]) -> dict[str, str]:
        self.candidates_for(provider)
        index = self.canonical_by_source_by_provider[provider]
        return {source_id: index[source_id] for source_id in source_job_ids if source_id in index}

    def refresh_candidate(self, provider: str, candidate: Mapping[str, Any]) -> None:
        """Update derived indexes after an insert or in-memory canonical mutation."""
        with self._load_lock:
            existing_ids = self.existing_job_ids_by_provider.setdefault(provider, set())
            canonical_by_source = self.canonical_by_source_by_provider.setdefault(provider, {})
            canonical_id = normalize_job_identifier(candidate.get("job_id"))
            source_ids = _candidate_source_ids(candidate)
            existing_ids.update(source_ids)
            if canonical_id is not None:
                canonical_by_source.update({source_id: canonical_id for source_id in source_ids})

            company = candidate.get("company")
            job_title = candidate.get("job_title")
            if company and job_title:
                self.company_title_keys_by_provider.setdefault(provider, set()).add(
                    (company.strip().lower(), job_title.strip().lower())
                )

    def add_candidate(self, provider: str, candidate: dict) -> None:
        self.candidates_for(provider)
        with self._load_lock:
            self.candidates_by_provider[provider].append(candidate)
            self.refresh_candidate(provider, candidate)

    def invalidate_provider(self, provider: str) -> None:
        with self._load_lock:
            self.candidates_by_provider.pop(provider, None)
            self.existing_job_ids_by_provider.pop(provider, None)
            self.company_title_keys_by_provider.pop(provider, None)
            self.canonical_by_source_by_provider.pop(provider, None)


class CanonicalTaskApplyAmbiguous(RuntimeError):
    pass


class CanonicalTaskLeaseLost(RuntimeError):
    pass


class CanonicalTaskReceiptConflict(RuntimeError):
    pass


def _provider_candidate_set_revision(candidates: Sequence[Mapping[str, Any]]) -> str:
    canonical_revisions = {}
    for candidate in candidates:
        canonical_id = normalize_job_identifier(candidate.get("job_id"))
        if canonical_id is None:
            continue
        revision = int(candidate.get("canonical_revision") or 0)
        if revision < 0:
            raise ValueError("canonical candidate revision must be non-negative")
        canonical_revisions[canonical_id] = str(revision)
    ordered_ids = sorted(canonical_revisions, key=lambda value: value.encode("utf-8"))
    revision_input = "".join(
        f"{len(job_id.encode('utf-8'))}:{job_id}"
        f"{len(canonical_revisions[job_id].encode('utf-8'))}:{canonical_revisions[job_id]}"
        for job_id in ordered_ids
    )
    return hashlib.sha256(revision_input.encode("utf-8")).hexdigest()


def _exact_source_candidate(job_id: str, candidates: list[dict]) -> dict | None:
    for candidate in sorted(candidates, key=lambda row: str(row.get("job_id") or "")):
        if job_id in _candidate_source_ids(candidate):
            return candidate
    return None


def _canonical_membership_provenances(
    task: Mapping[str, Any], job: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_provenances = task.get("membership_provenances")
    if raw_provenances is None:
        raw_provenances = []
    if not isinstance(raw_provenances, list):
        raise ValueError("adaptive discovery task membership provenances must be an array")
    if not raw_provenances:
        archetypes = _membership_archetypes(dict(job))
        if len(archetypes) != 1:
            raise ValueError("adaptive detail application requires membership provenance")
        archetype = archetypes[0]
        provenance = _membership_provenance(dict(job), archetype)
        observed_at = _valid_membership_observed_at(dict(job), provenance)
        if observed_at is None:
            observed_at = str(
                task.get("latest_observed_at")
                or task.get("first_observed_at")
                or datetime.now(timezone.utc).isoformat()
            )
        raw_provenances = [{
            **provenance,
            "lane": archetype,
            "archetype": archetype,
            "observed_at": observed_at,
        }]

    canonical_by_json: dict[str, dict[str, Any]] = {}
    for raw_provenance in raw_provenances:
        if not isinstance(raw_provenance, Mapping):
            raise ValueError("adaptive discovery membership provenance must be an object")
        provenance = dict(raw_provenance)
        raw_archetype = provenance.get("lane") or provenance.get("archetype")
        if not isinstance(raw_archetype, str) or not raw_archetype.strip():
            raise ValueError("adaptive discovery membership provenance lacks a lane")
        archetype = canonical_lane_slug(raw_archetype)
        if archetype not in CANONICAL_LANE_SLUGS:
            raise ValueError(f"Unknown membership archetype '{raw_archetype}'")
        provenance["lane"] = archetype
        provenance["archetype"] = archetype
        observed_at = _valid_membership_observed_at({}, provenance)
        if observed_at is None:
            raise ValueError("adaptive discovery membership provenance has invalid observed_at")
        provenance["observed_at"] = observed_at
        encoded = json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        canonical_by_json[encoded] = provenance
    return [canonical_by_json[key] for key in sorted(canonical_by_json)]


def build_linkedin_discovery_task_application(
    task: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    run_context: CanonicalRunContext,
    runtime_profile: dict | None = None,
    runtime_profiles: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    source_job_id = normalize_job_identifier(job.get("job_id"))
    task_source_job_id = normalize_job_identifier(task.get("source_job_id"))
    if source_job_id is None or source_job_id != task_source_job_id:
        raise ValueError("adaptive detail source does not match its discovery task")
    provider = str(job.get("provider") or task.get("provider") or "linkedin")
    if provider != "linkedin":
        raise ValueError("adaptive canonical application requires provider linkedin")

    prepared_job = dict(job)
    prepared_job["job_id"] = source_job_id
    prepared_job["provider"] = provider
    ingestion_run_id = str(task.get("first_ingestion_run_id") or "").strip()
    if not ingestion_run_id:
        raise ValueError("adaptive discovery task is missing its ingestion run")
    prepared_job["scrape_run_id"] = ingestion_run_id

    candidates = run_context.candidates_for(provider)
    match = (
        find_canonical_match(prepared_job, candidates)
        if getattr(config, "ENABLE_REPOST_DEDUP", True)
        else _exact_source_candidate(source_job_id, candidates)
    )
    accepted_relist = False
    if match is None:
        action = "insert"
        canonical_job_id = source_job_id
        canonical_payload = prepare_canonical_insert_payload(prepared_job)
        expected: dict[str, Any] = {}
    else:
        canonical_job_id = str(match["job_id"])
        canonical_payload = prepare_repost_update_payload(match, prepared_job)
        canonical_payload.pop("job_id", None)
        accepted_relist = bool(
            getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
            and prepared_job.get("same_id_relist_candidate")
            and int(canonical_payload.get("same_id_relist_count") or 0)
            > int(match.get("same_id_relist_count") or 0)
        )
        action = "accepted_relist" if accepted_relist else "update"
        expected = {"last_seen_at": match.get("last_seen_at")}
        if accepted_relist:
            expected["listing_instances"] = match.get("listing_instances") or []

    provenances = _canonical_membership_provenances(task, prepared_job)
    provenance_lanes = {str(provenance["archetype"]) for provenance in provenances}
    if runtime_profiles is None and len(provenance_lanes) > 1:
        raise ValueError("cross-lane adaptive membership requires runtime profiles by lane")
    memberships = []
    for provenance in provenances:
        archetype = str(provenance["archetype"])
        if runtime_profiles is not None:
            if archetype not in runtime_profiles:
                raise ValueError(f"adaptive detail has no runtime profile for lane {archetype!r}")
            lane_runtime_profile = runtime_profiles[archetype]
        else:
            lane_runtime_profile = runtime_profile
        filter_state = evaluate_lane_filter(
            prepared_job,
            archetype=archetype,
            runtime_profile=lane_runtime_profile,
        )
        observed_at = str(provenance["observed_at"])
        memberships.append({
            "archetype": archetype,
            "query_scope": provenance,
            "query_id": provenance.get("query_id"),
            "query": provenance.get("query") or provenance.get("search_query"),
            "query_type": provenance.get("query_type") or provenance.get("query_kind"),
            "language": provenance.get("language"),
            "location_scope": provenance.get("location_scope"),
            "geography_id": provenance.get("geography_id"),
            "first_matched_at": observed_at,
            "last_matched_at": observed_at,
            "filter_status": filter_state["filter_status"],
            "is_filtered": filter_state["is_filtered"],
            "filter_reason": filter_state["filter_reason"],
        })
    memberships.sort(key=lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ))
    latest_membership_observed_at = max(
        (str(membership["last_matched_at"]) for membership in memberships),
        key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
    )

    content_version = None
    content_hash = make_description_content_hash(prepared_job.get("description"))
    if getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True) and content_hash:
        content_version = {
            "content_hash": content_hash,
            "description": prepared_job.get("description"),
            "description_fingerprint": make_description_fingerprint(
                prepared_job.get("description")
            ),
            "observed_at": (
                prepared_job.get("detail_metadata_checked_at")
                or latest_membership_observed_at
            ),
            "ingestion_run_id": ingestion_run_id,
        }

    relist = None
    if accepted_relist:
        relist = {
            "relisted_on": _date_part(
                prepared_job.get("same_id_relist_date") or prepared_job.get("posted_at")
            ),
            "observed_at": (
                prepared_job.get("detail_metadata_checked_at")
                or latest_membership_observed_at
            ),
            "evidence": prepared_job.get("same_id_relist_evidence") or {},
        }

    membership_provenance_revision = int(task.get("membership_provenance_revision") or 0)
    if membership_provenance_revision < 0:
        raise ValueError("adaptive discovery membership provenance revision must be non-negative")

    return {
        "version": "linkedin-canonical-task-apply-v3",
        "provider_candidate_set_revision": _provider_candidate_set_revision(candidates),
        "membership_provenance_revision": membership_provenance_revision,
        "source": {
            "provider": provider,
            "source_job_id": source_job_id,
            "ingestion_run_id": ingestion_run_id,
        },
        "canonical": {
            "action": action,
            "canonical_job_id": canonical_job_id,
            "payload": {
                key: value for key, value in canonical_payload.items()
                if key in JOB_WRITE_FIELDS
            },
            "expected": expected,
        },
        "content_version": content_version,
        "memberships": memberships,
        "relist": relist,
    }


def _is_ambiguous_canonical_apply_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError)):
        return True
    status = getattr(error, "status_code", None)
    if status in {502, 503, 504, 520}:
        return True
    if error.args and isinstance(error.args[0], Mapping):
        status = error.args[0].get("status") or error.args[0].get("status_code")
        if status in {502, 503, 504, 520, "502", "503", "504", "520"}:
            return True
    return False


def _is_canonical_task_lease_error(error: Exception) -> bool:
    codes = [getattr(error, "code", None)]
    values: list[Any] = [str(error), getattr(error, "message", None)]
    if error.args and isinstance(error.args[0], Mapping):
        codes.append(error.args[0].get("code"))
        values.append(error.args[0].get("message"))
    text = " ".join(str(value or "") for value in values).lower()
    normalized_codes = {str(code) for code in codes if code is not None}
    return (
        "40001" in normalized_codes
        or ("55000" in normalized_codes and "lease" in text)
        or "task lease lost" in text
        or "task lease expired" in text
    )


def _is_canonical_task_receipt_conflict(error: Exception) -> bool:
    values: list[Any] = [str(error)]
    if error.args and isinstance(error.args[0], Mapping):
        values.extend((error.args[0].get("code"), error.args[0].get("message")))
    text = " ".join(str(value or "") for value in values).lower()
    return "23505" in text and "receipt" in text


def _refresh_context_after_task_apply(
    run_context: CanonicalRunContext,
    application: Mapping[str, Any],
    returned_canonical_job_id: str,
    returned_canonical_revision: Any,
) -> None:
    canonical = application["canonical"]
    planned_id = str(canonical["canonical_job_id"])
    if returned_canonical_job_id != planned_id:
        run_context.invalidate_provider("linkedin")
        return
    payload = dict(canonical["payload"])
    try:
        canonical_revision = int(returned_canonical_revision)
    except (TypeError, ValueError):
        run_context.invalidate_provider("linkedin")
        return
    if canonical_revision < 0:
        run_context.invalidate_provider("linkedin")
        return
    payload["canonical_revision"] = canonical_revision
    if canonical["action"] == "insert":
        payload["job_id"] = returned_canonical_job_id
        run_context.add_candidate("linkedin", payload)
        return
    candidates = run_context.candidates_for("linkedin")
    candidate = next(
        (row for row in candidates if str(row.get("job_id")) == returned_canonical_job_id),
        None,
    )
    if candidate is None:
        run_context.invalidate_provider("linkedin")
        return
    candidate.update(payload)
    run_context.refresh_candidate("linkedin", candidate)


def apply_linkedin_discovery_task_canonical(
    task: Mapping[str, Any],
    worker_id: str,
    job: Mapping[str, Any],
    *,
    run_context: CanonicalRunContext,
    runtime_profile: dict | None = None,
    runtime_profiles: Mapping[str, dict] | None = None,
    db: Any = None,
    max_stale_replans: int = 2,
) -> str:
    client = db or supabase
    task_snapshot = dict(task)
    for stale_attempt in range(max_stale_replans + 1):
        application = build_linkedin_discovery_task_application(
            task_snapshot,
            job,
            run_context=run_context,
            runtime_profile=runtime_profile,
            runtime_profiles=runtime_profiles,
        )
        response = None
        for ambiguous_attempt in range(2):
            try:
                response = client.rpc("apply_linkedin_discovery_task_canonical", {
                    "p_task_id": task_snapshot["id"],
                    "p_worker_id": worker_id,
                    "p_lease_token": task_snapshot["lease_token"],
                    "p_application": application,
                }).execute()
                break
            except Exception as error:
                if _is_canonical_task_lease_error(error):
                    raise CanonicalTaskLeaseLost(
                        "adaptive discovery task lease was lost before publication"
                    ) from error
                if _is_canonical_task_receipt_conflict(error):
                    raise CanonicalTaskReceiptConflict(
                        "completed adaptive task has a conflicting canonical receipt"
                    ) from error
                if not _is_ambiguous_canonical_apply_error(error):
                    raise
                if ambiguous_attempt == 1:
                    raise CanonicalTaskApplyAmbiguous(
                        "canonical task application outcome is ambiguous"
                    ) from error
        if response is None:
            raise CanonicalTaskApplyAmbiguous(
                "canonical task application returned no response"
            )
        result = _single_rpc_record(
            response.data, "apply_linkedin_discovery_task_canonical"
        )
        outcome = result.get("outcome")
        if outcome == "stale_plan":
            run_context.invalidate_provider("linkedin")
            refreshed_provenances = result.get("task_membership_provenances")
            refreshed_revision = result.get("task_membership_provenance_revision")
            if isinstance(refreshed_provenances, list) and refreshed_revision is not None:
                task_snapshot["membership_provenances"] = refreshed_provenances
                task_snapshot["membership_provenance_revision"] = int(refreshed_revision)
            if stale_attempt == max_stale_replans:
                raise RuntimeError("canonical task application remained stale after replanning")
            heartbeat_linkedin_discovery_task(
                int(task_snapshot["id"]), worker_id,
                str(task_snapshot["lease_token"]), db=client
            )
            continue
        if outcome not in {"applied", "replayed"}:
            raise RuntimeError(
                f"apply_linkedin_discovery_task_canonical returned invalid outcome {outcome!r}"
            )
        canonical_job_id = normalize_job_identifier(result.get("canonical_job_id"))
        if canonical_job_id is None:
            raise RuntimeError(
                "apply_linkedin_discovery_task_canonical returned no canonical job ID"
            )
        if outcome == "replayed":
            run_context.invalidate_provider("linkedin")
        else:
            _refresh_context_after_task_apply(
                run_context, application, canonical_job_id,
                result.get("canonical_revision"),
            )
        return canonical_job_id
    raise RuntimeError("canonical task application exhausted stale-plan retries")


def save_jobs_canonicalized_with_mapping(
    jobs_data: list,
    run_context: CanonicalRunContext | None = None,
) -> CanonicalSaveResult:
    """Save jobs and retain the canonical ID for every source input position."""
    candidates_cache = {}
    saved_job_ids: set[str] = set()
    canonical_by_source: dict[str, str] = {}
    canonical_ids_by_input: list[str | None] = []
    scrape_run_id = datetime.now(timezone.utc).isoformat()
    for raw_job in jobs_data:
        job = dict(raw_job)
        job_id = normalize_job_identifier(job.get("job_id"))
        if job_id is None:
            logging.warning("Skipping job with missing or non-scalar job_id: %r", job.get("job_id"))
            canonical_ids_by_input.append(None)
            continue
        job["job_id"] = job_id
        job.setdefault("scrape_run_id", scrape_run_id)
        if not getattr(config, "ENABLE_REPOST_DEDUP", True):
            persisted = prepare_canonical_insert_payload(job)
            result = save_job_to_supabase(persisted)
            saved_job_id = normalize_job_identifier(result)
            if saved_job_id is not None:
                saved_job_ids.add(saved_job_id)
                canonical_by_source[job_id] = saved_job_id
                upsert_job_archetype_membership(saved_job_id, job)
                if run_context is not None:
                    persisted["job_id"] = saved_job_id
                    run_context.add_candidate(job.get("provider"), persisted)
            canonical_ids_by_input.append(saved_job_id)
            continue

        cache_key = job.get("provider")
        candidates = run_context.candidates_for(cache_key) if run_context is not None else candidates_cache.get(cache_key)
        if candidates is None:
            candidates = get_canonical_candidates(provider=cache_key)
            candidates_cache[cache_key] = candidates
        match = find_canonical_match(job, candidates)

        if match:
            payload = prepare_repost_update_payload(match, job)
            accepted_relist = bool(
                job.get("provider") == "linkedin"
                and getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
                and job.get("same_id_relist_candidate")
            )
            if accepted_relist:
                apply_accepted_relist(
                    job,
                    canonical_job_id=match["job_id"],
                    projection=payload,
                    expected_listing_instances=match.get("listing_instances") or [],
                    expected_last_seen_at=match.get("last_seen_at"),
                )
            else:
                query = (
                    supabase.table(config.SUPABASE_TABLE_NAME)
                    .update({key: value for key, value in payload.items() if key != "job_id"})
                    .eq("job_id", match["job_id"])
                )
                # last_seen_at is the compare-and-swap token for this update. Do
                # not also filter on listing_instances: PostgREST puts filters
                # in the PATCH URL, and historical JSON arrays can make that URL
                # large enough for the server/proxy to reject it with HTTP 400.
                last_seen_at = match.get("last_seen_at")
                query = query.is_("last_seen_at", None) if last_seen_at is None else query.eq("last_seen_at", last_seen_at)
                response = query.execute()
                if len(response.data or []) != 1:
                    raise RuntimeError(
                        f"Concurrent canonical update detected for job_id={match['job_id']}"
                    )
                if job.get("provider") == "linkedin" and getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True):
                    save_listing_content_version(job, canonical_job_id=match["job_id"])
            matched_job_id = normalize_job_identifier(match.get("job_id"))
            if matched_job_id is not None:
                saved_job_ids.add(matched_job_id)
                canonical_by_source[job_id] = matched_job_id
                upsert_job_archetype_membership(matched_job_id, job)
            canonical_ids_by_input.append(matched_job_id)
            match.update(payload)
            if run_context is not None:
                run_context.refresh_candidate(cache_key, match)
        else:
            payload = prepare_canonical_insert_payload(job)
            result = save_job_to_supabase(payload)
            saved_job_id = normalize_job_identifier(result)
            if saved_job_id is not None:
                payload["job_id"] = saved_job_id
                saved_job_ids.add(saved_job_id)
                canonical_by_source[job_id] = saved_job_id
                upsert_job_archetype_membership(saved_job_id, job)
                if run_context is not None:
                    run_context.add_candidate(cache_key, payload)
                else:
                    candidates.append(payload)
            canonical_ids_by_input.append(saved_job_id)
            if job.get("provider") == "linkedin" and getattr(config, "ENABLE_LINKEDIN_RELIST_TRACKING", True):
                if saved_job_id is not None:
                    save_listing_content_version(job, canonical_job_id=saved_job_id)
    return CanonicalSaveResult(
        canonical_ids=sorted(saved_job_ids),
        canonical_by_source=canonical_by_source,
        canonical_ids_by_input=canonical_ids_by_input,
    )


def save_jobs_canonicalized(
    jobs_data: list,
    run_context: CanonicalRunContext | None = None,
) -> list[str]:
    """Save jobs and return sorted, deduplicated canonical IDs."""
    return save_jobs_canonicalized_with_mapping(jobs_data, run_context=run_context).canonical_ids


def save_linkedin_jobs_canonicalized(
    jobs_data: list,
    run_context: CanonicalRunContext | None = None,
) -> list[str]:
    """Save LinkedIn jobs and return the same list[str] canonical-ID contract."""
    return save_jobs_canonicalized(jobs_data, run_context=run_context)


def save_linkedin_jobs_canonicalized_with_mapping(
    jobs_data: list,
    run_context: CanonicalRunContext | None = None,
) -> CanonicalSaveResult:
    """Save LinkedIn jobs with exact source/input-to-canonical correspondence."""
    return save_jobs_canonicalized_with_mapping(jobs_data, run_context=run_context)

# --- Supabase Functions ---
def start_ingestion_run(
    run_id: str,
    provider: str,
    search_query: str | None = None,
    archetype: str | None = None,
    filter_profile: str | None = None,
    query_scope: str | None = None,
) -> None:
    supabase.table("ingestion_runs").upsert({
        "id": run_id,
        "provider": provider,
        "search_query": search_query,
        "archetype": archetype,
        "filter_profile": filter_profile,
        "query_scope": query_scope or "",
    }, on_conflict="id", ignore_duplicates=True, returning=ReturnMethod.minimal).execute()


def finish_ingestion_run(run_id: str, **metrics: Any) -> None:
    payload = {key: value for key, value in metrics.items() if value is not None}
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    (
        supabase.table("ingestion_runs")
        .update(payload, returning=ReturnMethod.minimal)
        .eq("id", run_id)
        .execute()
    )


def _single_rpc_record(data: Any, rpc_name: str) -> dict[str, Any]:
    if isinstance(data, Mapping):
        return dict(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) == 1 and isinstance(data[0], Mapping):
            return dict(data[0])
    raise RuntimeError(f"{rpc_name} returned an invalid response")


def create_linkedin_discovery_cycle(
    *,
    execution_id: str,
    configuration_revision: int | None,
    configuration_hash: str,
    user_agent: str,
    scopes: list[dict[str, Any]],
    db: Any = None,
) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("create_linkedin_discovery_cycle", {
        "p_execution_id": execution_id,
        "p_config_revision": configuration_revision,
        "p_config_content_hash": configuration_hash,
        "p_user_agent": user_agent,
        "p_scopes": scopes,
    }).execute()
    record = _single_rpc_record(response.data, "create_linkedin_discovery_cycle")
    if not isinstance(record.get("cycle_id"), int) or not isinstance(record.get("scopes"), list):
        raise RuntimeError("create_linkedin_discovery_cycle returned an incomplete response")
    return record


def get_linkedin_scope_coverage_states(
    scope_keys: list[str], *, db: Any = None
) -> dict[str, dict[str, Any]]:
    if not scope_keys:
        return {}
    client = db or supabase
    response = (
        client.table("linkedin_scope_coverage_state")
        .select(
            "scope_key,last_operational_success_at,recommended_pages,"
            "coverage_debt,last_deep_sweep_at"
        )
        .in_("scope_key", scope_keys)
        .execute()
    )
    rows = response.data or []
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("linkedin scope coverage state query returned an invalid response")
    return {str(row["scope_key"]): dict(row) for row in rows}


def get_pending_linkedin_coverage_debt(
    scope_keys: list[str], *, db: Any = None
) -> dict[str, dict[str, Any]]:
    if not scope_keys:
        return {}
    client = db or supabase
    response = (
        client.table("linkedin_coverage_debt")
        .select("scope_key,source_window_earliest_at,source_window_latest_at,created_at")
        .in_("scope_key", scope_keys)
        .eq("status", "pending")
        .order("source_window_earliest_at")
        .execute()
    )
    rows = response.data or []
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("linkedin coverage debt query returned an invalid response")
    oldest_by_scope: dict[str, dict[str, Any]] = {}
    for row in rows:
        oldest_by_scope.setdefault(str(row["scope_key"]), dict(row))
    return oldest_by_scope


def expire_linkedin_coverage_debt(
    scope_key: str, recovery_floor: str, *, db: Any = None
) -> int:
    client = db or supabase
    response = client.rpc("expire_linkedin_coverage_debt", {
        "p_scope_key": scope_key,
        "p_recovery_floor": recovery_floor,
    }).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("expire_linkedin_coverage_debt returned an invalid response")
    return value


def accept_linkedin_coverage_debt(
    debt_id: int, reviewer: str, reason: str, *, db: Any = None
) -> bool:
    client = db or supabase
    response = client.rpc("accept_linkedin_coverage_debt", {
        "p_debt_id": debt_id,
        "p_reviewer": reviewer,
        "p_reason": reason,
    }).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if value is not True:
        raise RuntimeError("accept_linkedin_coverage_debt returned an invalid response")
    return True


def accept_linkedin_discovery_requirement(
    requirement: Mapping[str, Any], reviewer: str, reason: str, *, db: Any = None
) -> bool:
    client = db or supabase
    response = client.rpc("accept_linkedin_discovery_requirement", {
        "p_discovery_cycle_id": requirement["discovery_cycle_id"],
        "p_ingestion_run_id": requirement["ingestion_run_id"],
        "p_provider": requirement["provider"],
        "p_source_job_id": requirement["source_job_id"],
        "p_task_kind": requirement["task_kind"],
        "p_requirement_key": requirement["requirement_key"],
        "p_reviewer": reviewer,
        "p_reason": reason,
    }).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if value is not True:
        raise RuntimeError(
            "accept_linkedin_discovery_requirement returned an invalid response"
        )
    return True


def commit_linkedin_discovery_page(payload: dict[str, Any], db: Any = None) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("commit_linkedin_discovery_page", {"p_page": payload}).execute()
    record = _single_rpc_record(response.data, "commit_linkedin_discovery_page")
    required = {"cards", "new_source_ids", "new_workflow_source_ids", "tasks_created"}
    if not required.issubset(record):
        raise RuntimeError("commit_linkedin_discovery_page returned an incomplete response")
    return record


def finish_linkedin_discovery_scope(
    ingestion_run_id: str,
    coverage_status: str,
    *,
    coverage_reason: str,
    db: Any = None,
) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("finish_linkedin_discovery_scope", {
        "p_ingestion_run_id": ingestion_run_id,
        "p_coverage_status": coverage_status,
        "p_coverage_reason": coverage_reason,
    }).execute()
    return _single_rpc_record(response.data, "finish_linkedin_discovery_scope")


def fail_linkedin_discovery_cycle(cycle_id: int, reason: str, db: Any = None) -> None:
    client = db or supabase
    client.rpc("fail_linkedin_discovery_cycle", {
        "p_cycle_id": cycle_id,
        "p_reason": reason[:2000],
    }).execute()


def seal_linkedin_discovery_cycle(
    cycle_id: int,
    *,
    advance_watermark: bool,
    db: Any = None,
) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("seal_linkedin_discovery_cycle", {
        "p_cycle_id": cycle_id,
        "p_advance_watermark": advance_watermark,
    }).execute()
    return _single_rpc_record(response.data, "seal_linkedin_discovery_cycle")


def resolve_eligible_failed_linkedin_discovery_cycles(
    resolving_cycle_id: int, *, db: Any = None
) -> int:
    client = db or supabase
    response = client.rpc(
        "resolve_eligible_failed_linkedin_discovery_cycles",
        {"p_resolving_cycle_id": resolving_cycle_id},
    ).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(
            "resolve_eligible_failed_linkedin_discovery_cycles returned an invalid response"
        )
    return value


def acquire_linkedin_request_grant(
    producer: str,
    request_kind: str,
    request_key: str,
    *,
    db: Any = None,
) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("acquire_linkedin_request_grant", {
        "p_producer": producer,
        "p_request_kind": request_kind,
        "p_request_key": request_key,
    }).execute()
    return _single_rpc_record(response.data, "acquire_linkedin_request_grant")


def consume_linkedin_request_grant(grant_id: str, producer: str, *, db: Any = None) -> dict[str, Any]:
    client = db or supabase
    response = client.rpc("consume_linkedin_request_grant", {
        "p_grant_id": grant_id,
        "p_producer": producer,
    }).execute()
    return _single_rpc_record(response.data, "consume_linkedin_request_grant")


def finish_linkedin_request_grant(
    grant_id: str,
    producer: str,
    response_class: str,
    http_status: int | None,
    *,
    db: Any = None,
) -> bool:
    client = db or supabase
    response = client.rpc("finish_linkedin_request_grant", {
        "p_grant_id": grant_id,
        "p_producer": producer,
        "p_response_class": response_class,
        "p_http_status": http_status,
    }).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, bool):
        raise RuntimeError("finish_linkedin_request_grant returned an invalid response")
    return value


def open_linkedin_source_circuit(
    grant_id: str,
    producer: str,
    reason: str,
    http_status: int | None,
    *,
    db: Any = None,
) -> bool:
    client = db or supabase
    response = client.rpc("open_linkedin_source_circuit", {
        "p_grant_id": grant_id,
        "p_producer": producer,
        "p_reason": reason[:1000],
        "p_http_status": http_status,
    }).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, bool):
        raise RuntimeError("open_linkedin_source_circuit returned an invalid response")
    return value


def claim_linkedin_discovery_tasks(
    worker_id: str,
    *,
    limit: int,
    order_mode: str = "oldest",
    db: Any = None,
) -> list[dict[str, Any]]:
    client = db or supabase
    response = client.rpc("claim_linkedin_discovery_tasks", {
        "p_worker_id": worker_id,
        "p_limit": limit,
        "p_order_mode": order_mode,
    }).execute()
    rows = response.data or []
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("claim_linkedin_discovery_tasks returned an invalid response")
    return [dict(row) for row in rows]


def transition_linkedin_discovery_task(
    task_id: int,
    worker_id: str,
    lease_token: str,
    status: str,
    *,
    canonical_job_id: str | None = None,
    error_code: str | None = None,
    db: Any = None,
) -> dict[str, Any]:
    client = db or supabase
    try:
        response = client.rpc("transition_linkedin_discovery_task", {
            "p_task_id": task_id,
            "p_worker_id": worker_id,
            "p_lease_token": lease_token,
            "p_status": status,
            "p_canonical_job_id": canonical_job_id,
            "p_error_code": error_code,
        }).execute()
    except Exception as error:
        if _is_canonical_task_lease_error(error):
            raise CanonicalTaskLeaseLost(
                "adaptive discovery task lease was lost during transition"
            ) from error
        raise
    return _single_rpc_record(response.data, "transition_linkedin_discovery_task")


def heartbeat_linkedin_discovery_task(
    task_id: int,
    worker_id: str,
    lease_token: str,
    *,
    db: Any = None,
) -> str:
    client = db or supabase
    try:
        response = client.rpc("heartbeat_linkedin_discovery_task", {
            "p_task_id": task_id,
            "p_worker_id": worker_id,
            "p_lease_token": lease_token,
        }).execute()
    except Exception as error:
        if _is_canonical_task_lease_error(error):
            raise CanonicalTaskLeaseLost(
                "adaptive discovery task lease was lost during heartbeat"
            ) from error
        raise
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value:
        raise RuntimeError("heartbeat_linkedin_discovery_task returned an invalid response")
    return value


SCRAPE_RUN_STATE_ID = 1


def get_last_successful_scrape_at() -> Optional[str]:
    """Read the singleton scraper watermark from the historical state table."""
    try:
        response = (
            supabase.table("scrape_run_state")
            .select("last_successful_scrape_at")
            .eq("id", SCRAPE_RUN_STATE_ID)
            .limit(1)
            .execute()
        )
    except Exception as error:
        logging.error("Failed to read scrape success watermark: %s", error)
        return None

    rows = response.data or []
    if not rows:
        return None
    return rows[0].get("last_successful_scrape_at")


def _scrape_run_state_matches(finished_at: str) -> bool:
    """Confirm that the singleton row contains the exact timestamp written."""
    try:
        response = (
            supabase.table("scrape_run_state")
            .select("last_successful_scrape_at")
            .eq("id", SCRAPE_RUN_STATE_ID)
            .limit(1)
            .execute()
        )
    except Exception as error:
        logging.warning("Failed to verify scrape success watermark: %s", error)
        return False

    rows = response.data or []
    if not rows:
        return False

    persisted_at = rows[0].get("last_successful_scrape_at")
    return _scrape_timestamps_match(persisted_at, finished_at)


def _scrape_timestamps_match(persisted_at: Any, finished_at: str) -> bool:
    """Compare timestamp values as instants despite Postgres formatting changes."""
    if persisted_at == finished_at:
        return True
    try:
        # PostgreSQL may omit insignificant fractional-second zeroes. Compare
        # exact instants rather than requiring identical timestamp formatting.
        return datetime.fromisoformat(persisted_at.replace("Z", "+00:00")) == (
            datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _rpc_timestamp(data: Any) -> Any:
    """Unwrap scalar RPC results across PostgREST client response shapes."""
    if isinstance(data, Mapping):
        return data.get("record_scrape_success")
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) != 1:
            return None
        return _rpc_timestamp(data[0])
    return data


def _record_scrape_success_rpc_is_absent(error: Exception) -> bool:
    """Only permit the legacy direct-write fallback for a missing RPC."""
    if isinstance(error, AttributeError) and "rpc" in str(error):
        return True
    code = getattr(error, "code", None)
    if code is None and error.args and isinstance(error.args[0], Mapping):
        code = error.args[0].get("code")
    return code == "PGRST202"


def _record_scrape_success_direct(finished_at: str) -> bool:
    """Write directly for deployments that have not installed the RPC yet."""
    payload = {"last_successful_scrape_at": finished_at}

    try:
        (
            supabase.table("scrape_run_state")
            .update(payload)
            .eq("id", SCRAPE_RUN_STATE_ID)
            .execute()
        )
        if _scrape_run_state_matches(finished_at):
            return True
    except Exception as error:
        logging.warning("Failed to update scrape success watermark: %s", error)

    try:
        # A fresh database may have the table but not its singleton row yet.
        (
            supabase.table("scrape_run_state")
            .upsert(
                {"id": SCRAPE_RUN_STATE_ID, **payload},
                on_conflict="id",
            )
            .execute()
        )
        if _scrape_run_state_matches(finished_at):
            return True
    except Exception as error:
        logging.error("Failed to persist scrape success watermark: %s", error)
        return False

    logging.error("Scrape success watermark did not match after update and upsert")
    return False


def record_scrape_success() -> bool:
    """Persist and verify the required top-level scraper watermark."""
    finished_at = datetime.now(timezone.utc).isoformat()

    try:
        response = supabase.rpc(
            "record_scrape_success",
            {"p_finished_at": finished_at},
        ).execute()
    except Exception as error:
        if _record_scrape_success_rpc_is_absent(error):
            logging.warning(
                "record_scrape_success RPC is absent; using legacy direct watermark write"
            )
            return _record_scrape_success_direct(finished_at)
        logging.error("Failed to persist scrape success watermark via RPC: %s", error)
        return False

    returned_at = _rpc_timestamp(response.data)
    if not _scrape_timestamps_match(returned_at, finished_at):
        logging.error(
            "record_scrape_success RPC returned an unexpected timestamp: %r",
            returned_at,
        )
        return False

    if not _scrape_run_state_matches(finished_at):
        logging.error("Scrape success watermark did not match after RPC write")
        return False

    return True


def get_listing_tracking_context(
    provider: str,
    source_job_ids: list[str],
    canonical_by_source: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    ids = list(dict.fromkeys(str(job_id) for job_id in source_job_ids if job_id is not None))
    if not ids:
        return {}
    context: dict[str, dict] = {}
    state_response = (
        supabase.table("listing_states")
        .select("source_job_id,canonical_job_id,latest_trusted_posted_date,last_seen_at,same_id_relist_count,pending_relist_on")
        .eq("provider", provider)
        .in_("source_job_id", ids)
        .execute()
    )
    for row in state_response.data or []:
        context[str(row["source_job_id"])] = dict(row)
    unresolved = set(ids) - set(context)
    if canonical_by_source is not None:
        for source_id in unresolved:
            canonical_id = normalize_job_identifier(canonical_by_source.get(source_id))
            if canonical_id is not None:
                context.setdefault(source_id, {})["canonical_job_id"] = canonical_id
        # The supplied index is authoritative for this run. IDs absent from it
        # are genuinely new and must not trigger a repeated full JSON scan.
        unresolved.clear()
    if unresolved:
        offset = 0
        while unresolved:
            response = (
                supabase.table(config.SUPABASE_TABLE_NAME)
                .select("job_id,latest_job_id,listing_instances")
                .eq("provider", provider)
                .range(offset, offset + 999)
                .execute()
            )
            rows = response.data or []
            for row in rows:
                canonical_id = str(row["job_id"])
                known_ids = {canonical_id, str(row.get("latest_job_id"))}
                known_ids.update(
                    str(instance["job_id"])
                    for instance in (row.get("listing_instances") or [])
                    if instance.get("job_id") is not None
                )
                for source_id in unresolved & known_ids:
                    context.setdefault(source_id, {})["canonical_job_id"] = canonical_id
            unresolved -= set(context)
            if len(rows) < 1000:
                break
            offset += 1000
    by_source: dict[str, list[dict]] = {}
    offset = 0
    while True:
        observations = (
            supabase.table("listing_observations")
            .select("source_job_id,posted_at,observed_at,ingestion_run_id,query_scope")
            .eq("provider", provider)
            .in_("source_job_id", ids)
            .order("observed_at", desc=False)
            .range(offset, offset + 999)
            .execute()
        )
        page = observations.data or []
        for row in page:
            by_source.setdefault(str(row["source_job_id"]), []).append(row)
        if len(page) < 1000:
            break
        offset += len(page)
    for source_id, rows in by_source.items():
        context.setdefault(source_id, {})["observations"] = rows
    offset = 0
    while True:
        response = (
            supabase.table("listing_relist_events")
            .select("source_job_id,relisted_on")
            .eq("provider", provider)
            .in_("source_job_id", ids)
            .is_("superseded_by", "null")
            .range(offset, offset + 999)
            .execute()
        )
        page = response.data or []
        for row in page:
            context.setdefault(str(row["source_job_id"]), {}).setdefault(
                "accepted_relist_dates", []
            ).append(_date_part(row.get("relisted_on")))
        if len(page) < 1000:
            break
        offset += len(page)
    return context


def save_listing_observations(
    cards: list[dict],
    run_id: str,
    provider: str = "linkedin",
    query_scope: str | None = None,
    canonical_by_source: dict[str, str | None] | None = None,
) -> dict:
    canonical_by_source = canonical_by_source or {}
    observed_at = datetime.now(timezone.utc).isoformat()
    payloads = []
    skipped_missing_date = 0
    for card in cards:
        source_id = card.get("job_id")
        posted_at = _date_part(card.get("posted_at"))
        if source_id is None:
            skipped_missing_date += 1
            continue
        payloads.append({
            "provider": provider,
            "source_job_id": str(source_id),
            "canonical_job_id": canonical_by_source.get(str(source_id)),
            "ingestion_run_id": run_id,
            "observed_at": observed_at,
            "posted_at": posted_at,
            "posted_relative_text": card.get("posted_relative_text"),
            "location": card.get("location"),
            "card_label": card.get("card_label"),
            "result": "seen",
            "query_scope": query_scope,
            "trigger_evidence": card.get("trigger_evidence") or {},
            "page_number": card.get("page_number"),
            "page_start": card.get("page_start"),
            "position_on_page": card.get("position_on_page"),
            "position_in_scope": card.get("position_in_scope"),
        })
    if payloads:
        (
            supabase.table("listing_observations")
            .upsert(
                payloads,
                on_conflict="provider,source_job_id,ingestion_run_id,query_scope,result",
                ignore_duplicates=True,
                returning=ReturnMethod.minimal,
            )
            .execute()
        )
    return {"attempted": len(payloads), "skipped_missing_date": skipped_missing_date}


def save_listing_states(
    cards: list[dict],
    prior_context: dict[str, dict],
    canonical_by_source: dict[str, str | None] | None = None,
    provider: str = "linkedin",
) -> None:
    canonical_by_source = canonical_by_source or {}
    now_iso = datetime.now(timezone.utc).isoformat()
    payloads = []
    for card in cards:
        source_id = str(card.get("job_id"))
        posted_at = _date_part(card.get("posted_at"))
        if source_id == "None":
            continue
        if posted_at is None:
            prior = prior_context.get(source_id) or {}
            payloads.append({
                "provider": provider,
                "source_job_id": source_id,
                "canonical_job_id": canonical_by_source.get(source_id) or prior.get("canonical_job_id"),
                "first_seen_at": prior.get("first_seen_at") or now_iso,
                "last_seen_at": now_iso,
                "latest_trusted_posted_date": prior.get("latest_trusted_posted_date"),
                "pending_relist_on": prior.get("pending_relist_on"),
                "same_id_relist_count": int(prior.get("same_id_relist_count") or 0),
            })
            continue
        prior = prior_context.get(source_id) or {}
        fold = relist_tracking.fold_observations(
            list(prior.get("observations") or []) + [{
                "posted_at": posted_at,
                "observed_at": now_iso,
                "ingestion_run_id": card.get("scrape_run_id"),
            }],
            min_forward_days=getattr(config, "LINKEDIN_RELIST_MIN_FORWARD_DAYS", 2),
            stable_observations=getattr(config, "LINKEDIN_RELIST_STABLE_OBSERVATIONS", 2),
        )
        pending_event = relist_tracking.latest_pending_event(
            fold, prior.get("accepted_relist_dates")
        )
        payloads.append({
            "provider": provider,
            "source_job_id": source_id,
            "canonical_job_id": canonical_by_source.get(source_id) or prior.get("canonical_job_id"),
            "first_seen_at": prior.get("first_seen_at") or now_iso,
            "last_seen_at": now_iso,
            "latest_trusted_posted_date": fold["latest_trusted_posted_date"] or posted_at,
            "pending_relist_on": (
                pending_event.get("relisted_on") if pending_event
                else prior.get("pending_relist_on")
            ),
            "same_id_relist_count": max(
                int(prior.get("same_id_relist_count") or 0),
                len(prior.get("accepted_relist_dates") or []),
            ),
        })
    if payloads:
        (
            supabase.table("listing_states")
            .upsert(
                payloads,
                on_conflict="provider,source_job_id",
                returning=ReturnMethod.minimal,
            )
            .execute()
        )


def save_listing_content_version(job: dict, canonical_job_id: str | None) -> str | None:
    description = job.get("description")
    source_id = job.get("job_id")
    content_hash = make_description_content_hash(description)
    if source_id is None or content_hash is None:
        return None
    observed_at = job.get("detail_metadata_checked_at") or datetime.now(timezone.utc).isoformat()
    existing = (
        supabase.table("listing_content_versions")
        .select("observation_count,last_ingestion_run_id")
        .eq("provider", job.get("provider") or "linkedin")
        .eq("source_job_id", str(source_id))
        .eq("content_hash", content_hash)
        .limit(1)
        .execute()
        .data
        or []
    )
    ingestion_run_id = job.get("scrape_run_id")
    if existing and ingestion_run_id and existing[0].get("last_ingestion_run_id") != ingestion_run_id:
        (
            supabase.table("listing_content_versions")
            .update({
                "last_observed_at": observed_at,
                "observation_count": int(existing[0].get("observation_count") or 1) + 1,
                "last_ingestion_run_id": ingestion_run_id,
            })
            .eq("provider", job.get("provider") or "linkedin")
            .eq("source_job_id", str(source_id))
            .eq("content_hash", content_hash)
            .execute()
        )
    elif not existing:
        supabase.table("listing_content_versions").insert({
            "provider": job.get("provider") or "linkedin",
            "source_job_id": str(source_id),
            "content_hash": content_hash,
            "canonical_job_id": canonical_job_id,
            "description": description,
            "description_fingerprint": make_description_fingerprint(description),
            "first_observed_at": observed_at,
            "last_observed_at": observed_at,
            "last_ingestion_run_id": ingestion_run_id,
        }).execute()
    (
        supabase.table("listing_states")
        .update({"canonical_job_id": canonical_job_id, "current_content_hash": content_hash})
        .eq("provider", job.get("provider") or "linkedin")
        .eq("source_job_id", str(source_id))
        .execute()
    )
    return content_hash


def save_relist_event(job: dict, canonical_job_id: str | None) -> None:
    relisted_on = _date_part(job.get("same_id_relist_date") or job.get("posted_at"))
    if not relisted_on or job.get("job_id") is None:
        return
    supabase.table("listing_relist_events").upsert({
        "provider": job.get("provider") or "linkedin",
        "source_job_id": str(job["job_id"]),
        "canonical_job_id": canonical_job_id,
        "relisted_on": relisted_on,
        "observed_at": job.get("detail_metadata_checked_at") or datetime.now(timezone.utc).isoformat(),
        "ingestion_run_id": job.get("scrape_run_id"),
        "evidence": job.get("same_id_relist_evidence") or {},
    }, on_conflict="provider,source_job_id,relisted_on", ignore_duplicates=True).execute()


def apply_accepted_relist(
    job: dict,
    canonical_job_id: str,
    projection: dict,
    expected_listing_instances: list[dict],
    expected_last_seen_at: str | None,
) -> None:
    content_hash = make_description_content_hash(job.get("description"))
    response = supabase.rpc("apply_linkedin_relist_projection", {
        "p_canonical_job_id": canonical_job_id,
        "p_source_job_id": str(job["job_id"]),
        "p_ingestion_run_id": job.get("scrape_run_id"),
        "p_relisted_on": _date_part(job.get("same_id_relist_date") or job.get("posted_at")),
        "p_observed_at": job.get("detail_metadata_checked_at") or datetime.now(timezone.utc).isoformat(),
        "p_projection": {key: value for key, value in projection.items() if key != "job_id"},
        "p_expected_listing_instances": expected_listing_instances,
        "p_expected_last_seen_at": expected_last_seen_at,
        "p_evidence": job.get("same_id_relist_evidence") or {},
        "p_description": job.get("description"),
        "p_content_hash": content_hash,
        "p_description_fingerprint": make_description_fingerprint(job.get("description")),
    }).execute()
    if response.data is not True and response.data != [True]:
        raise RuntimeError(f"Atomic relist projection rejected for job_id={canonical_job_id}")


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
                .select("job_id, latest_job_id, company, job_title, listing_instances")
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

                normalized_job_id = normalize_job_identifier(job_id)
                if normalized_job_id is not None:
                    existing_ids.add(normalized_job_id)

                latest_job_id = item.get("latest_job_id")
                normalized_latest_job_id = normalize_job_identifier(latest_job_id)
                if normalized_latest_job_id is not None:
                    existing_ids.add(normalized_latest_job_id)

                existing_ids.update(
                    identifier
                    for instance in (item.get("listing_instances") or [])
                    if isinstance(instance, Mapping)
                    and (identifier := normalize_job_identifier(instance.get("job_id"))) is not None
                )

                if company and job_title:
                    normalized_company = company.strip().lower()
                    normalized_title = job_title.strip().lower()
                    existing_company_title_keys.add((normalized_company, normalized_title))

            if len(data) < batch_size:
                break
            offset += batch_size

        print(f"Fetched {len(existing_ids)} job IDs and {len(existing_company_title_keys)} company-title pairs.")

    except Exception as e:
        print(f"Error fetching existing jobs from Supabase: {e}")

    return existing_ids, existing_company_title_keys


def get_canonical_job_ids_for_sources(source_job_ids: list[str]) -> dict[str, str]:
    """Resolve every already-known source/listing ID to its canonical jobs.job_id."""
    wanted = {str(source_id) for source_id in source_job_ids if source_id is not None}
    if not wanted:
        return {}
    resolved: dict[str, str] = {}
    offset = 0
    while True:
        response = (
            supabase.table(config.SUPABASE_TABLE_NAME)
            .select("job_id,latest_job_id,listing_instances")
            .eq("provider", "linkedin")
            .range(offset, offset + 999)
            .execute()
        )
        page = response.data or []
        for row in page:
            canonical_id = normalize_job_identifier(row.get("job_id"))
            if canonical_id is None:
                continue
            identifiers = {
                identifier for identifier in (
                    canonical_id,
                    normalize_job_identifier(row.get("latest_job_id")),
                ) if identifier is not None
            }
            identifiers.update(
                identifier
                for instance in (row.get("listing_instances") or [])
                if isinstance(instance, Mapping)
                and (identifier := normalize_job_identifier(instance.get("job_id"))) is not None
            )
            for identifier in wanted.intersection(identifiers):
                resolved[identifier] = canonical_id
        if len(page) < 1000 or wanted.issubset(resolved):
            return resolved
        offset += len(page)


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
    canonical = canonical_lane_slug(resolved)
    # software_tpm is the only compatibility alias. Prefer an active canonical
    # runtime profile so DB-configured filtering remains authoritative.
    profile = config.ARCHETYPE_CONFIGS.get(canonical) if resolved in LANE_ALIASES else None
    profile = profile or config.ARCHETYPE_CONFIGS.get(resolved)
    if profile is None:
        raise ValueError(
            f"Unknown archetype/filter profile '{resolved}'. "
            "Expected a configured canonical lane or the software_tpm alias."
        )
    return profile


def evaluate_lane_filter(job: dict, archetype: str | None = None, runtime_profile: dict | None = None) -> dict:
    """Evaluate one lane. Excludes filter; include terms are OR routing signals."""
    title = job.get("job_title") or ""
    company = job.get("company") or ""
    desc = job.get("description") or ""
    profile = runtime_profile or get_filter_profile(archetype or job.get("lane") or job.get("archetype"))

    for raw_pattern in profile["company_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(company):
            return {"filter_status": "filtered", "is_filtered": True, "filter_reason": f"company:{raw_pattern}", "is_entry_level_filtered": False}

    for raw_pattern in profile["title_entry_level_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(title):
            return {"filter_status": "filtered", "is_filtered": True, "filter_reason": f"title_entry_level:{raw_pattern}", "is_entry_level_filtered": True}

    for raw_pattern in profile["title_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(title):
            return {"filter_status": "filtered", "is_filtered": True, "filter_reason": f"title:{raw_pattern}", "is_entry_level_filtered": False}

    for raw_pattern in profile["desc_blocklist"]:
        if re.compile(raw_pattern, re.IGNORECASE).search(desc):
            return {"filter_status": "filtered", "is_filtered": True, "filter_reason": f"desc:{raw_pattern}", "is_entry_level_filtered": False}

    title_includes = profile.get("title_include", profile.get("title_context", []))
    desc_includes = profile.get("description_include", profile.get("description_context", []))
    include_signals = [(title, pattern) for pattern in title_includes] + [(desc, pattern) for pattern in desc_includes]
    # A lane may provide many context tokens. They are alternatives, not an
    # implicit AND. Missing positives routes to review and does not filter.
    if include_signals and not any(re.compile(pattern, re.IGNORECASE).search(value) for value, pattern in include_signals):
        return {"filter_status": "review", "is_filtered": False, "filter_reason": "include:no_route_signal", "is_entry_level_filtered": False}
    return {"filter_status": "included", "is_filtered": False, "filter_reason": None, "is_entry_level_filtered": False}


def match_filter_reason(job: dict, archetype: str | None = None, runtime_profile: dict | None = None) -> tuple[str | None, bool]:
    result = evaluate_lane_filter(job, archetype=archetype, runtime_profile=runtime_profile)
    return (result["filter_reason"] if result["is_filtered"] else None, result["is_entry_level_filtered"])


def persist_lane_filter_state(job_id: str, archetype: str, job: dict, runtime_profile: dict | None = None, db=None) -> dict:
    db = db or supabase
    lane = canonical_lane_slug(archetype)
    result = evaluate_lane_filter(job, archetype=lane, runtime_profile=runtime_profile)
    payload = {key: result[key] for key in ("filter_status", "is_filtered", "filter_reason")}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    db.table("job_archetype_memberships").update(payload).eq("job_id", str(job_id)).eq("archetype", lane).execute()
    return result

def save_job_to_supabase(job_data: dict) -> str | None:
    """Save one job and return only its stable scalar identifier."""
    job_id = normalize_job_identifier(job_data.get("job_id"))
    if job_id is None:
        logging.warning("Skipping job with missing or non-scalar job_id: %r", job_data.get("job_id"))
        return None
    job = dict(job_data)
    job["job_id"] = job_id
    saved_job_ids = save_jobs_to_supabase([job])
    return normalize_job_identifier(saved_job_ids)


def save_jobs_to_supabase(jobs_data: list) -> list[str]:
    """
    Saves or updates a list of job data dictionaries to the Supabase table using upsert.
    This avoids duplicate key errors by updating existing records based on job_id.
    """
    if not jobs_data:
        print("No job data provided to save/update.")
        return []

    # Enforce generated-column safety at the final generic write boundary too,
    # including callers that bypass canonical payload preparation.
    jobs_data = [_canonical_job_write_payload(job) for job in jobs_data]

    # Ensure job_id is present and potentially convert to the correct type if needed
    # (Assuming job_id in jobs_data is already the correct string type for your 'text' column)
    processed_jobs_data = []
    for job in jobs_data:
        job_id = normalize_job_identifier(job.get('job_id'))
        if job_id is not None:
             # If your Supabase job_id column was numeric, you'd convert here:
             # try:
             #     job['job_id'] = int(job['job_id'])
             #     processed_jobs_data.append(job)
             # except (ValueError, TypeError):
             #     print(f"Warning: Invalid job_id format found: {job.get('job_id')}. Skipping.")
             # Since it's text, just ensure it's a string (it likely already is)
             filtered_job = {key: value for key, value in job.items() if key in JOB_WRITE_FIELDS}
             filtered_job['job_id'] = job_id
             processed_jobs_data.append(filtered_job)
        else:
            print(f"Warning: Job data missing job_id. Skipping: {job}")


    if not processed_jobs_data:
        print("No valid job data remaining after processing.")
        return []

    print(f"Attempting to upsert {len(processed_jobs_data)} jobs to Supabase...")

    try:
        # Use table name from config
        # Use upsert instead of insert. It will insert new rows
        # or update existing rows if a job_id conflict occurs based on the primary key.
        # Ensure 'job_id' is the primary key or has a unique constraint in your Supabase table.
        # By default, supabase-py's upsert updates the row on conflict.
        response = supabase.table(config.SUPABASE_TABLE_NAME).upsert(processed_jobs_data).execute()
        response_job_ids = extract_job_identifiers(response)
        input_job_ids = [job["job_id"] for job in processed_jobs_data]
        saved_job_ids = response_job_ids or input_job_ids
        print(f"Successfully upserted/updated {len(processed_jobs_data)} jobs.")
        return saved_job_ids

    except Exception as e:
        print(f"Error upserting data to Supabase: {e}")
        # Consider logging the data that failed to upsert for debugging
        # print(f"Failed data: {processed_jobs_data}")
        return []


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


def get_jobs_to_score(
    limit: int, archetype: str | None = None, worker_id: str | None = None,
    lease_seconds: int = 900,
) -> list:
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
        if archetype:
            if not worker_id:
                raise ValueError("worker_id is required for lane scoring claims")
            response = supabase.rpc("get_lane_jobs_to_score", {
                "p_archetype": canonical_lane_slug(archetype), "p_limit": limit,
                "p_worker_id": worker_id, "p_lease_seconds": lease_seconds,
            }).execute()
        else:
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

def get_top_scored_jobs_for_resume_generation(
    limit: int, archetype: str | None = None, worker_id: str | None = None,
    lease_seconds: int = 900,
) -> list:
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
        rpc_name = "get_lane_jobs_for_resume_generation" if archetype else "get_jobs_for_resume_generation_custom_sort"
        if archetype and not worker_id:
            raise ValueError("worker_id is required for lane resume claims")
        rpc_args = ({"p_archetype": canonical_lane_slug(archetype), "p_limit": limit,
                     "p_worker_id": worker_id, "p_lease_seconds": lease_seconds}
                    if archetype else {"p_page_number": 1, "p_page_size": limit})
        response = supabase.rpc(rpc_name, rpc_args).execute()

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

def get_jobs_to_rescore(
    limit: int, archetype: str | None = None, worker_id: str | None = None,
    lease_seconds: int = 900,
) -> list:
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
        if archetype and not worker_id:
            raise ValueError("worker_id is required for lane rescore claims")
        response = supabase.rpc(
            "get_lane_jobs_for_rescore" if archetype else "get_jobs_for_rescore",
            ({"p_archetype": canonical_lane_slug(archetype), "p_limit": limit,
              "p_worker_id": worker_id, "p_lease_seconds": lease_seconds}
             if archetype else {"p_limit_val": limit})
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

def update_job_score(
    job_id: str, score: int, resume_score_stage: str = "initial",
    archetype: str | None = None, worker_id: str | None = None,
) -> bool:
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
        if archetype and worker_id:
            response = supabase.rpc("complete_lane_score_claim", {
                "p_job_id": job_id, "p_archetype": canonical_lane_slug(archetype),
                "p_worker_id": worker_id, "p_score": score,
                "p_score_stage": resume_score_stage,
            }).execute()
            return response.data is True
        update_payload = ({
            "match_score": score,
            "score_stage": resume_score_stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        } if archetype else {
            "resume_score": score,
            "resume_score_stage": resume_score_stage,
        })
        table = "job_archetype_memberships" if archetype else config.SUPABASE_TABLE_NAME
        query = supabase.table(table)\
                           .update(update_payload)\
                           .eq("job_id", job_id)
        if archetype:
            query = query.eq("archetype", canonical_lane_slug(archetype))
        response = query.execute()

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

def update_job_with_resume_link(job_id: str, customized_resume_id: str, new_status: Optional[str] = "resume_generated", archetype: str | None = None) -> bool:
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

        table = "job_archetype_memberships" if archetype else config.SUPABASE_TABLE_NAME
        query = supabase.table(table)\
                           .update(update_data)\
                           .eq("job_id", job_id)
        if archetype:
            query = query.eq("archetype", canonical_lane_slug(archetype))
        response = query.execute()

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


def release_lane_score_claim(job_id: str, archetype: str, worker_id: str, *, failed: bool = False) -> bool:
    """Release a score claim only when it is still owned by this worker."""
    try:
        response = supabase.rpc("fail_lane_score_claim" if failed else "release_lane_score_claim", {
            "p_job_id": job_id, "p_archetype": canonical_lane_slug(archetype),
            "p_worker_id": worker_id,
        }).execute()
        return response.data is True
    except Exception as exc:
        logging.error("Could not release score claim for %s/%s: %s", archetype, job_id, exc)
        return False


def complete_lane_resume_claim(
    job_id: str, archetype: str, worker_id: str, customized_resume_id: str,
    base_resume_id: str | None = None,
) -> bool:
    """Publish resume state only when the caller still owns its lease."""
    try:
        response = supabase.rpc("complete_lane_resume_claim", {
            "p_job_id": job_id, "p_archetype": canonical_lane_slug(archetype),
            "p_worker_id": worker_id, "p_customized_resume_id": customized_resume_id,
            "p_base_resume_id": base_resume_id,
        }).execute()
        return response.data is True
    except Exception as exc:
        logging.error("Could not complete resume claim for %s/%s: %s", archetype, job_id, exc)
        return False


def release_lane_resume_claim(job_id: str, archetype: str, worker_id: str, *, failed: bool = False) -> bool:
    """Release a resume claim only when it is still owned by this worker."""
    try:
        response = supabase.rpc("fail_lane_resume_claim" if failed else "release_lane_resume_claim", {
            "p_job_id": job_id, "p_archetype": canonical_lane_slug(archetype),
            "p_worker_id": worker_id,
        }).execute()
        return response.data is True
    except Exception as exc:
        logging.error("Could not release resume claim for %s/%s: %s", archetype, job_id, exc)
        return False

def save_customized_resume(resume_data: 'Resume', resume_path: str, archetype: str | None = None, base_resume_id: str | None = None, job_id: str | None = None, customized_resume_id: str | None = None) -> Optional[Any]:
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
        if customized_resume_id:
            data_to_insert['id'] = customized_resume_id
        if archetype:
            data_to_insert['archetype'] = canonical_lane_slug(archetype)
        if base_resume_id:
            data_to_insert['base_resume_id'] = base_resume_id
        if job_id:
            data_to_insert['job_id'] = str(job_id)

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
    Atomically replaces the parsed base resume through a database RPC.

    Args:
        resume_data: The parsed resume data as a dictionary.

    Returns:
        True if saved successfully, False otherwise.
    """
    if not resume_data:
        logging.error("No resume data provided to save.")
        return False

    try:
        response = supabase.rpc(
            "replace_base_resume",
            {"p_resume_data": resume_data},
        ).execute()
        if response.data is True or response.data == [True]:
            logging.info("Successfully replaced base resume atomically.")
            return True
        logging.error("Base resume replacement returned an unexpected response: %r", response.data)
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


def get_archetype_base_resume(archetype: str) -> Optional[dict]:
    """Return lane profile resume data plus its stable base_resume_id."""
    try:
        response = supabase.table("archetype_resume_profiles").select(
            "base_resume_id, profile_data, base_resume(resume_data)"
        ).eq("archetype", canonical_lane_slug(archetype)).eq("enabled", True).limit(1).execute()
        if not response.data:
            return None
        row = response.data[0]
        data = dict(row.get("profile_data") or (row.get("base_resume") or {}).get("resume_data") or {})
        if not data:
            return None
        data["base_resume_id"] = row.get("base_resume_id")
        return data
    except Exception as exc:
        logging.error("Error fetching archetype resume profile for %s: %s", archetype, exc)
        return None
