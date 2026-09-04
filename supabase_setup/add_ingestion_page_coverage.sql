BEGIN;

ALTER TABLE public.ingestion_runs
    ADD COLUMN IF NOT EXISTS page_coverage jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE public.ingestion_runs
    DROP CONSTRAINT IF EXISTS ingestion_runs_page_coverage_json_check;

ALTER TABLE public.ingestion_runs
    ADD CONSTRAINT ingestion_runs_page_coverage_json_check CHECK (
        jsonb_typeof(page_coverage) = 'array'
        AND jsonb_array_length(page_coverage) <= 100
    );

COMMIT;
