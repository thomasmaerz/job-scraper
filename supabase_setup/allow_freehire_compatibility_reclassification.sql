-- Allow eligible rows to replace obsolete input hashes under the source snapshot fence.
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
      AND p_expected_source_snapshot <@ to_jsonb(j);
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected = 1;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz)
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_freehire_compat_job(text, text, jsonb, text, timestamptz)
TO service_role;

COMMIT;
