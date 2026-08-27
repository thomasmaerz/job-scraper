BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '5min';

CREATE TABLE IF NOT EXISTS public.scrape_run_state (
    id integer PRIMARY KEY CHECK (id = 1),
    last_successful_scrape_at timestamp with time zone
);

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

INSERT INTO public.scrape_run_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

COMMIT;
