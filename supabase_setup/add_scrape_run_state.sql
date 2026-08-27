BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '5min';

CREATE TABLE IF NOT EXISTS public.scrape_run_state (
    id integer PRIMARY KEY CHECK (id = 1),
    last_successful_scrape_at timestamp with time zone
);

CREATE OR REPLACE FUNCTION public.record_scrape_success(p_finished_at timestamptz)
RETURNS timestamptz
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    persisted_at timestamptz;
BEGIN
    IF p_finished_at IS NULL THEN
        RAISE EXCEPTION 'p_finished_at must not be null' USING ERRCODE = '22004';
    END IF;

    INSERT INTO public.scrape_run_state (id, last_successful_scrape_at)
    VALUES (1, p_finished_at)
    ON CONFLICT (id) DO UPDATE
    SET last_successful_scrape_at = EXCLUDED.last_successful_scrape_at
    RETURNING last_successful_scrape_at INTO persisted_at;

    RETURN persisted_at;
END;
$function$;

ALTER TABLE public.scrape_run_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS service_role_singleton_watermark_access ON public.scrape_run_state;
CREATE POLICY service_role_singleton_watermark_access
    ON public.scrape_run_state
    FOR ALL
    TO service_role
    USING (id = 1)
    WITH CHECK (id = 1);

REVOKE ALL ON TABLE public.scrape_run_state FROM PUBLIC, anon, authenticated;
GRANT ALL ON TABLE public.scrape_run_state TO service_role;

REVOKE ALL ON FUNCTION public.record_scrape_success(timestamptz) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.record_scrape_success(timestamptz) TO service_role;

INSERT INTO public.scrape_run_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

COMMIT;
