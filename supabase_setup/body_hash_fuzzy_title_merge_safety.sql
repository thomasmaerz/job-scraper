-- Production hardening required before applying body-hash/fuzzy-title folds.
-- The canonical merge function must set this transaction-local flag before
-- remapping append-only observation ownership and deleting source jobs.

ALTER TABLE public.listing_observations
    DROP CONSTRAINT IF EXISTS listing_observations_canonical_job_id_fkey;
ALTER TABLE public.listing_observations
    ADD CONSTRAINT listing_observations_canonical_job_id_fkey
    FOREIGN KEY (canonical_job_id) REFERENCES public.jobs(job_id)
    ON DELETE NO ACTION
    DEFERRABLE INITIALLY DEFERRED;

CREATE OR REPLACE FUNCTION public.prevent_listing_observation_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF current_setting('app.historical_repost_merge', true) = 'on'
       AND TG_OP = 'UPDATE'
       AND NEW.canonical_job_id IS DISTINCT FROM OLD.canonical_job_id
       AND to_jsonb(NEW) - 'canonical_job_id' = to_jsonb(OLD) - 'canonical_job_id'
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'listing_observations is append-only';
END;
$$;

-- merge_historical_repost_plan() additionally performs these operations in
-- one transaction, after locking every plan member:
--
--   PERFORM set_config('app.historical_repost_merge', 'on', true);
--   validate source/target provider and description_fingerprint equality;
--   UPDATE listing_observations SET canonical_job_id = survivor ...;
--   UPDATE listing_content_versions SET canonical_job_id = survivor ...;
--   UPDATE listing_relist_events SET canonical_job_id = survivor ...;
--   UPDATE listing_states SET canonical_job_id = survivor ...;
--   aggregate distinct locations with string_agg(..., '; ');
--   DELETE source jobs;
--   TRUNCATE job_repost_merge_plan;
--
-- replace_historical_repost_plan(jsonb) also uses TRUNCATE rather than an
-- unrestricted DELETE so the database's delete guard remains effective.
