-- Batch Freehire compatibility claims and writes to minimize Data API traffic.
BEGIN;

SET LOCAL ROLE postgres;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

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
            claim->>'job_id', claim->>'expected_input_hash',
            claim->'expected_source_snapshot', p_worker_id, p_replacement_before
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
            result->>'job_id', result->>'expected_input_hash',
            result->'expected_source_snapshot', p_worker_id, result->'payload'
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
            metadata_update->>'job_id', metadata_update->'expected_source_snapshot',
            metadata_update->'payload'
        ) THEN
            job_id := metadata_update->>'job_id';
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_freehire_compat_jobs(jsonb, text, timestamptz)
FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz)
FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.persist_freehire_compat_results(jsonb, text)
FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.apply_freehire_compat_metadata_batch(jsonb)
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_freehire_compat_jobs(jsonb, text, timestamptz)
TO service_role;
GRANT EXECUTE ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz)
TO service_role;
GRANT EXECUTE ON FUNCTION public.persist_freehire_compat_results(jsonb, text)
TO service_role;
GRANT EXECUTE ON FUNCTION public.apply_freehire_compat_metadata_batch(jsonb)
TO service_role;

COMMIT;
