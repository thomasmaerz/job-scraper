BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

ALTER TABLE public.jobs
    ADD COLUMN IF NOT EXISTS same_id_relist_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS freehire_category text,
    ADD COLUMN IF NOT EXISTS freehire_seniority text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS is_remote boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS freehire_remote_evidence jsonb,
    ADD COLUMN IF NOT EXISTS freehire_compat_status text NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS freehire_compat_input_hash text,
    ADD COLUMN IF NOT EXISTS freehire_compat_import_hash text,
    ADD COLUMN IF NOT EXISTS freehire_compat_model text,
    ADD COLUMN IF NOT EXISTS freehire_compat_prompt_version text,
    ADD COLUMN IF NOT EXISTS freehire_compat_schema_version text,
    ADD COLUMN IF NOT EXISTS freehire_compat_confidence numeric,
    ADD COLUMN IF NOT EXISTS freehire_compat_classified_at timestamptz,
    ADD COLUMN IF NOT EXISTS freehire_compat_error text,
    ADD COLUMN IF NOT EXISTS freehire_compat_attempts integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS freehire_compat_claimed_at timestamptz,
    ADD COLUMN IF NOT EXISTS freehire_compat_claimed_by text,
    ADD COLUMN IF NOT EXISTS freehire_compat_provenance jsonb,
    ADD COLUMN IF NOT EXISTS freehire_compat_next_retry_at timestamptz;

UPDATE public.jobs
SET freehire_compat_status = 'pending'
WHERE provider = 'linkedin'
  AND freehire_compat_status = 'current'
  AND (
      freehire_compat_input_hash IS NULL
      OR freehire_compat_schema_version IS DISTINCT FROM 'freehire-compat-v1'
      OR freehire_compat_prompt_version IS DISTINCT FROM 'freehire-category-v1'
      OR freehire_category IS NULL
  );

ALTER TABLE public.jobs DROP CONSTRAINT IF EXISTS jobs_freehire_category_check;
ALTER TABLE public.jobs ADD CONSTRAINT jobs_freehire_category_check CHECK (
    freehire_category IS NULL OR freehire_category IN (
        'software_engineering', 'backend', 'frontend', 'fullstack', 'mobile',
        'devops', 'sre', 'network_engineering', 'data_engineering', 'data_science',
        'data_analytics', 'ml_ai', 'ai_engineering', 'qa', 'security', 'hardware',
        'embedded', 'blockchain', 'architecture', 'design', 'engineering_design',
        'product', 'project_management', 'management', 'marketing', 'sales',
        'support', 'business_analysis', 'solutions_engineering', 'developer_relations',
        'technical_writing', 'recruiting', 'hr', 'finance', 'legal', 'operations',
        'customer_success', 'other'
    )
);
ALTER TABLE public.jobs DROP CONSTRAINT IF EXISTS jobs_freehire_seniority_check;
ALTER TABLE public.jobs ADD CONSTRAINT jobs_freehire_seniority_check CHECK (
    freehire_seniority IN ('', 'intern', 'junior', 'middle', 'senior', 'lead', 'staff', 'principal', 'c_level')
);
ALTER TABLE public.jobs DROP CONSTRAINT IF EXISTS jobs_freehire_compat_status_check;
ALTER TABLE public.jobs ADD CONSTRAINT jobs_freehire_compat_status_check CHECK (
    freehire_compat_status IN ('pending', 'processing', 'current', 'failed')
);
ALTER TABLE public.jobs DROP CONSTRAINT IF EXISTS jobs_freehire_compat_confidence_check;
ALTER TABLE public.jobs ADD CONSTRAINT jobs_freehire_compat_confidence_check CHECK (
    freehire_compat_confidence IS NULL OR freehire_compat_confidence BETWEEN 0 AND 1
);

CREATE INDEX IF NOT EXISTS idx_jobs_freehire_compat_status
    ON public.jobs (provider, freehire_compat_status);
CREATE INDEX IF NOT EXISTS idx_jobs_freehire_compat_input_hash
    ON public.jobs (freehire_compat_input_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_freehire_compat_retry
    ON public.jobs (freehire_compat_status, freehire_compat_next_retry_at)
    WHERE provider = 'linkedin';

COMMENT ON COLUMN public.jobs.freehire_category IS 'Pinned Freehire category assigned by the compatibility classifier.';
COMMENT ON COLUMN public.jobs.freehire_seniority IS 'Pinned Freehire seniority; empty means unresolved rather than guessed.';
COMMENT ON COLUMN public.jobs.is_remote IS 'True only when standalone ASCII remote occurs in normalized visible title, location, or description text.';
COMMENT ON COLUMN public.jobs.freehire_remote_evidence IS 'Deterministic visible-text match field and normalized span for is_remote=true; null otherwise.';
COMMENT ON COLUMN public.jobs.freehire_compat_status IS 'Compatibility lifecycle: pending, processing, current, or failed.';
COMMENT ON COLUMN public.jobs.freehire_compat_input_hash IS 'SHA-256 of canonical classification inputs and schema version; independent of scraped_at.';
COMMENT ON COLUMN public.jobs.freehire_compat_import_hash IS 'SHA-256 of downstream classification, live ID, effective timestamps, and remote fields.';
COMMENT ON COLUMN public.jobs.freehire_compat_model IS 'Exact LLM model used for the latest classification attempt.';
COMMENT ON COLUMN public.jobs.freehire_compat_prompt_version IS 'Version of the strict Freehire classification prompt.';
COMMENT ON COLUMN public.jobs.freehire_compat_schema_version IS 'Version binding hash normalization and pinned vocabularies.';
COMMENT ON COLUMN public.jobs.freehire_compat_confidence IS 'Classifier confidence from zero through one.';
COMMENT ON COLUMN public.jobs.freehire_compat_classified_at IS 'Timestamp of the latest successful compatibility classification.';
COMMENT ON COLUMN public.jobs.freehire_compat_error IS 'Last retry-exhausted per-job classification error; null after success.';
COMMENT ON COLUMN public.jobs.freehire_compat_attempts IS 'Durable count of classification passes attempted for this canonical job.';
COMMENT ON COLUMN public.jobs.freehire_compat_claimed_at IS 'Worker lease timestamp while status is processing.';
COMMENT ON COLUMN public.jobs.freehire_compat_claimed_by IS 'Worker identifier owning the current processing lease.';
COMMENT ON COLUMN public.jobs.freehire_compat_provenance IS 'Auditable batch, model, prompt, schema, timestamp, or failure provenance.';
COMMENT ON COLUMN public.jobs.freehire_compat_next_retry_at IS 'Earliest time a failed classification may be claimed again.';

CREATE OR REPLACE FUNCTION public.invalidate_freehire_compat_input()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF pg_trigger_depth() = 1 AND NEW.provider = 'linkedin' AND (
        OLD.job_id IS DISTINCT FROM NEW.job_id
        OR OLD.job_title IS DISTINCT FROM NEW.job_title
        OR OLD.location IS DISTINCT FROM NEW.location
        OR OLD.description IS DISTINCT FROM NEW.description
        OR OLD.level IS DISTINCT FROM NEW.level
    ) THEN
        NEW.freehire_compat_status := 'pending';
        NEW.freehire_compat_input_hash := NULL;
        NEW.freehire_compat_import_hash := NULL;
        NEW.freehire_compat_error := NULL;
        NEW.freehire_compat_claimed_at := NULL;
        NEW.freehire_compat_claimed_by := NULL;
        NEW.freehire_compat_next_retry_at := NULL;
    ELSIF pg_trigger_depth() = 1 AND NEW.provider = 'linkedin' AND (
        OLD.latest_job_id IS DISTINCT FROM NEW.latest_job_id
        OR OLD.company IS DISTINCT FROM NEW.company
        OR OLD.posted_at IS DISTINCT FROM NEW.posted_at
        OR OLD.scraped_at IS DISTINCT FROM NEW.scraped_at
        OR OLD.first_seen_at IS DISTINCT FROM NEW.first_seen_at
        OR OLD.last_seen_at IS DISTINCT FROM NEW.last_seen_at
        OR OLD.last_seen_posted_at IS DISTINCT FROM NEW.last_seen_posted_at
        OR OLD.last_checked IS DISTINCT FROM NEW.last_checked
        OR OLD.detail_metadata_checked_at IS DISTINCT FROM NEW.detail_metadata_checked_at
        OR OLD.salary_text IS DISTINCT FROM NEW.salary_text
        OR OLD.salary_min IS DISTINCT FROM NEW.salary_min
        OR OLD.salary_max IS DISTINCT FROM NEW.salary_max
        OR OLD.salary_currency IS DISTINCT FROM NEW.salary_currency
        OR OLD.applicant_count IS DISTINCT FROM NEW.applicant_count
        OR OLD.applicant_count_text IS DISTINCT FROM NEW.applicant_count_text
        OR OLD.applicant_count_type IS DISTINCT FROM NEW.applicant_count_type
        OR OLD.recruiter_name IS DISTINCT FROM NEW.recruiter_name
        OR OLD.recruiter_profile_url IS DISTINCT FROM NEW.recruiter_profile_url
        OR OLD.recruiter_identifier IS DISTINCT FROM NEW.recruiter_identifier
        OR OLD.original_job_id IS DISTINCT FROM NEW.original_job_id
        OR OLD.seen_count IS DISTINCT FROM NEW.seen_count
        OR OLD.posting_wave_count IS DISTINCT FROM NEW.posting_wave_count
        OR OLD.repost_count IS DISTINCT FROM NEW.repost_count
        OR OLD.same_id_relist_count IS DISTINCT FROM NEW.same_id_relist_count
        OR OLD.listing_instances IS DISTINCT FROM NEW.listing_instances
        OR OLD.archetype IS DISTINCT FROM NEW.archetype
        OR OLD.search_query IS DISTINCT FROM NEW.search_query
        OR OLD.filter_profile IS DISTINCT FROM NEW.filter_profile
        OR OLD.is_filtered IS DISTINCT FROM NEW.is_filtered
        OR OLD.is_entry_level_filtered IS DISTINCT FROM NEW.is_entry_level_filtered
        OR OLD.filter_reason IS DISTINCT FROM NEW.filter_reason
        OR OLD.description_fingerprint IS DISTINCT FROM NEW.description_fingerprint
    ) THEN
        NEW.freehire_compat_import_hash := NULL;
    ELSIF pg_trigger_depth() = 1 AND NEW.provider = 'linkedin'
      AND OLD.freehire_compat_import_hash IS NOT DISTINCT FROM NEW.freehire_compat_import_hash
      AND (
        OLD.freehire_category IS DISTINCT FROM NEW.freehire_category
        OR OLD.freehire_seniority IS DISTINCT FROM NEW.freehire_seniority
        OR OLD.is_remote IS DISTINCT FROM NEW.is_remote
        OR OLD.freehire_remote_evidence IS DISTINCT FROM NEW.freehire_remote_evidence
        OR OLD.freehire_compat_confidence IS DISTINCT FROM NEW.freehire_compat_confidence
        OR OLD.freehire_compat_classified_at IS DISTINCT FROM NEW.freehire_compat_classified_at
        OR OLD.freehire_compat_model IS DISTINCT FROM NEW.freehire_compat_model
        OR OLD.freehire_compat_prompt_version IS DISTINCT FROM NEW.freehire_compat_prompt_version
        OR OLD.freehire_compat_schema_version IS DISTINCT FROM NEW.freehire_compat_schema_version
        OR OLD.freehire_compat_provenance IS DISTINCT FROM NEW.freehire_compat_provenance
      ) THEN
        NEW.freehire_compat_import_hash := NULL;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS jobs_invalidate_freehire_compat_input ON public.jobs;
CREATE TRIGGER jobs_invalidate_freehire_compat_input
    BEFORE UPDATE ON public.jobs
    FOR EACH ROW EXECUTE FUNCTION public.invalidate_freehire_compat_input();

CREATE OR REPLACE FUNCTION public.apply_freehire_compat_metadata(
    p_job_id text,
    p_expected_source_snapshot jsonb,
    p_payload jsonb
) RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    affected integer;
BEGIN
    UPDATE public.jobs AS j
    SET is_remote = (p_payload->>'is_remote')::boolean,
        freehire_remote_evidence = p_payload->'freehire_remote_evidence',
        freehire_compat_import_hash = p_payload->>'freehire_compat_import_hash'
    WHERE job_id = p_job_id
      AND provider = 'linkedin'
      AND freehire_compat_status = 'current'
      AND p_expected_source_snapshot <@ to_jsonb(j);
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected = 1;
END;
$$;

DROP FUNCTION IF EXISTS public.claim_freehire_compat_job(text, text, text);
DROP FUNCTION IF EXISTS public.claim_freehire_compat_job(text, text, jsonb, text);
DROP FUNCTION IF EXISTS public.persist_freehire_compat_result(text, text, text, jsonb);

CREATE OR REPLACE FUNCTION public.claim_freehire_compat_job(
    p_job_id text,
    p_expected_input_hash text,
    p_expected_source_snapshot jsonb,
    p_worker_id text,
    p_replacement_before timestamptz DEFAULT NULL
) RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    affected integer;
BEGIN
    UPDATE public.jobs AS j
    SET freehire_compat_status = 'processing',
        freehire_compat_input_hash = p_expected_input_hash,
        freehire_compat_claimed_at = now(),
        freehire_compat_claimed_by = p_worker_id
    WHERE job_id = p_job_id
      AND provider = 'linkedin'
      AND (
          freehire_compat_status = 'pending'
          OR (freehire_compat_status = 'failed' AND COALESCE(freehire_compat_next_retry_at, '-infinity'::timestamptz) <= now())
          OR (freehire_compat_status = 'processing' AND COALESCE(freehire_compat_claimed_at, '-infinity'::timestamptz) < now() - interval '30 minutes')
          OR (
              freehire_compat_status = 'current'
              AND p_replacement_before IS NOT NULL
              AND COALESCE(freehire_compat_classified_at, '-infinity'::timestamptz) < p_replacement_before
          )
          OR (
              freehire_compat_status = 'current'
              AND (
                  freehire_compat_input_hash IS NULL
                  OR freehire_compat_model IS NULL
                  OR freehire_category IS NULL
                  OR freehire_compat_schema_version IS DISTINCT FROM 'freehire-compat-v1'
                  OR freehire_compat_prompt_version IS DISTINCT FROM 'freehire-category-v1'
              )
          )
      )
      AND (freehire_compat_input_hash IS NULL OR freehire_compat_input_hash = p_expected_input_hash)
      AND p_expected_source_snapshot <@ to_jsonb(j);
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public.persist_freehire_compat_result(
    p_job_id text,
    p_expected_input_hash text,
    p_expected_source_snapshot jsonb,
    p_worker_id text,
    p_payload jsonb
) RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    affected integer;
BEGIN
    UPDATE public.jobs AS j
    SET freehire_category = CASE WHEN p_payload ? 'freehire_category' THEN p_payload->>'freehire_category' ELSE freehire_category END,
        freehire_seniority = CASE WHEN p_payload ? 'freehire_seniority' THEN p_payload->>'freehire_seniority' ELSE freehire_seniority END,
        is_remote = CASE WHEN p_payload ? 'is_remote' THEN (p_payload->>'is_remote')::boolean ELSE is_remote END,
        freehire_remote_evidence = CASE WHEN p_payload ? 'freehire_remote_evidence' THEN p_payload->'freehire_remote_evidence' ELSE freehire_remote_evidence END,
        freehire_compat_status = p_payload->>'freehire_compat_status',
        freehire_compat_import_hash = CASE WHEN p_payload ? 'freehire_compat_import_hash' THEN p_payload->>'freehire_compat_import_hash' ELSE freehire_compat_import_hash END,
        freehire_compat_model = p_payload->>'freehire_compat_model',
        freehire_compat_prompt_version = p_payload->>'freehire_compat_prompt_version',
        freehire_compat_schema_version = p_payload->>'freehire_compat_schema_version',
        freehire_compat_confidence = CASE WHEN p_payload ? 'freehire_compat_confidence' THEN (p_payload->>'freehire_compat_confidence')::numeric ELSE freehire_compat_confidence END,
        freehire_compat_classified_at = CASE WHEN p_payload ? 'freehire_compat_classified_at' THEN (p_payload->>'freehire_compat_classified_at')::timestamptz ELSE freehire_compat_classified_at END,
        freehire_compat_error = p_payload->>'freehire_compat_error',
        freehire_compat_attempts = (p_payload->>'freehire_compat_attempts')::integer,
        freehire_compat_claimed_at = NULL,
        freehire_compat_claimed_by = NULL,
        freehire_compat_next_retry_at = CASE WHEN p_payload ? 'freehire_compat_next_retry_at' AND p_payload->>'freehire_compat_next_retry_at' IS NOT NULL THEN (p_payload->>'freehire_compat_next_retry_at')::timestamptz ELSE NULL END,
        freehire_compat_provenance = p_payload->'freehire_compat_provenance'
    WHERE job_id = p_job_id
      AND freehire_compat_status = 'processing'
      AND freehire_compat_input_hash = p_expected_input_hash
      AND p_expected_source_snapshot <@ to_jsonb(j)
      AND freehire_compat_claimed_by = p_worker_id
      AND freehire_compat_claimed_at >= now() - interval '30 minutes';
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_freehire_compat_jobs(
    p_claims jsonb,
    p_worker_id text,
    p_replacement_before timestamptz DEFAULT NULL
) RETURNS TABLE(job_id text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    claim jsonb;
BEGIN
    IF pg_catalog.jsonb_typeof(p_claims) <> 'array' THEN
        RAISE EXCEPTION 'p_claims must be a JSON array' USING ERRCODE = '22023';
    END IF;
    FOR claim IN SELECT value FROM pg_catalog.jsonb_array_elements(p_claims)
    LOOP
        IF public.claim_freehire_compat_job(
            claim->>'job_id',
            claim->>'expected_input_hash',
            claim->'expected_source_snapshot',
            p_worker_id,
            p_replacement_before
        ) THEN
            job_id := claim->>'job_id';
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION public.persist_freehire_compat_results(
    p_results jsonb,
    p_worker_id text
) RETURNS TABLE(job_id text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    result jsonb;
BEGIN
    IF pg_catalog.jsonb_typeof(p_results) <> 'array' THEN
        RAISE EXCEPTION 'p_results must be a JSON array' USING ERRCODE = '22023';
    END IF;
    FOR result IN SELECT value FROM pg_catalog.jsonb_array_elements(p_results)
    LOOP
        IF public.persist_freehire_compat_result(
            result->>'job_id',
            result->>'expected_input_hash',
            result->'expected_source_snapshot',
            p_worker_id,
            result->'payload'
        ) THEN
            job_id := result->>'job_id';
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION public.apply_freehire_compat_metadata_batch(
    p_updates jsonb
) RETURNS TABLE(job_id text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    metadata_update jsonb;
BEGIN
    IF pg_catalog.jsonb_typeof(p_updates) <> 'array' THEN
        RAISE EXCEPTION 'p_updates must be a JSON array' USING ERRCODE = '22023';
    END IF;
    FOR metadata_update IN SELECT value FROM pg_catalog.jsonb_array_elements(p_updates)
    LOOP
        IF public.apply_freehire_compat_metadata(
            metadata_update->>'job_id',
            metadata_update->'expected_source_snapshot',
            metadata_update->'payload'
        ) THEN
            job_id := metadata_update->>'job_id';
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$;

CREATE OR REPLACE VIEW public.freehire_jobs AS
SELECT
    job_id,
    COALESCE(latest_job_id, job_id) AS live_listing_id,
    company,
    job_title,
    level,
    location,
    description,
    posted_at,
    scraped_at,
    first_seen_at,
    last_seen_at,
    last_seen_posted_at,
    last_checked,
    detail_metadata_checked_at,
    freehire_category,
    freehire_seniority,
    is_remote,
    freehire_remote_evidence,
    freehire_compat_status,
    freehire_compat_input_hash,
    freehire_compat_import_hash,
    freehire_compat_model,
    freehire_compat_prompt_version,
    freehire_compat_schema_version,
    freehire_compat_confidence,
    freehire_compat_classified_at,
    freehire_compat_provenance,
    salary_text,
    salary_min,
    salary_max,
    salary_currency,
    applicant_count,
    applicant_count_text,
    applicant_count_type,
    recruiter_name,
    recruiter_profile_url,
    recruiter_identifier,
    original_job_id,
    latest_job_id,
    seen_count,
    posting_wave_count,
    repost_count,
    same_id_relist_count,
    listing_instances,
    archetype,
    search_query,
    filter_profile,
    is_filtered,
    is_entry_level_filtered,
    filter_reason,
    description_fingerprint
FROM public.jobs
WHERE provider = 'linkedin'
  AND freehire_compat_status = 'current'
  AND freehire_category IS NOT NULL
  AND freehire_compat_input_hash IS NOT NULL
  AND freehire_compat_import_hash IS NOT NULL
  AND freehire_compat_model IS NOT NULL
  AND freehire_compat_prompt_version = 'freehire-category-v1'
  AND freehire_compat_schema_version = 'freehire-compat-v1';

REVOKE ALL ON public.freehire_jobs FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.freehire_jobs TO service_role;
REVOKE ALL ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.persist_freehire_compat_result(text, text, jsonb, text, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.apply_freehire_compat_metadata(text, jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_freehire_compat_jobs(jsonb, text, timestamptz) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.persist_freehire_compat_results(jsonb, text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.apply_freehire_compat_metadata_batch(jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz) TO service_role;
GRANT EXECUTE ON FUNCTION public.persist_freehire_compat_result(text, text, jsonb, text, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION public.apply_freehire_compat_metadata(text, jsonb, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION public.claim_freehire_compat_jobs(jsonb, text, timestamptz) TO service_role;
GRANT EXECUTE ON FUNCTION public.persist_freehire_compat_results(jsonb, text) TO service_role;
GRANT EXECUTE ON FUNCTION public.apply_freehire_compat_metadata_batch(jsonb) TO service_role;

COMMIT;
