BEGIN;

UPDATE public.scrape_settings
SET max_pages_per_query = 6,
    updated_at = now()
WHERE singleton IS TRUE
  AND max_pages_per_query = 3;

COMMIT;
