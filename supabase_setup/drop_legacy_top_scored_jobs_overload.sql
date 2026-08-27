BEGIN;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';
DROP FUNCTION IF EXISTS "public"."get_top_scored_jobs_custom_sort"(integer, integer, text, integer, integer, text);
COMMIT;
