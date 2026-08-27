-- Database Cleanup Script for job-scraper
-- Preserves rmc_* tables (backgrndy/resumemuncher)
-- Removes job-scraper specific tables and functions

BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

-- Step 1: Drop foreign key constraints
ALTER TABLE IF EXISTS "public"."jobs" DROP CONSTRAINT IF EXISTS jobs_customized_resume_id_fkey;

-- Step 2: Drop job-scraper tables (CASCADE will handle indexes)
DROP VIEW IF EXISTS "public"."freehire_jobs" CASCADE;
DROP TABLE IF EXISTS "public"."listing_relist_events" CASCADE;
DROP TABLE IF EXISTS "public"."listing_content_versions" CASCADE;
DROP TABLE IF EXISTS "public"."listing_observations" CASCADE;
DROP TABLE IF EXISTS "public"."listing_states" CASCADE;
DROP TABLE IF EXISTS "public"."ingestion_runs" CASCADE;
DROP TABLE IF EXISTS "public"."job_repost_merge_plan" CASCADE;
DROP TABLE IF EXISTS "public"."job_resume_links" CASCADE;
DROP TABLE IF EXISTS "public"."job_listing_archive" CASCADE;
DROP TABLE IF EXISTS "public"."job_keyword_insights" CASCADE;
DROP TABLE IF EXISTS "public"."keyword_insights" CASCADE;
DROP TABLE IF EXISTS "public"."jobs" CASCADE;
DROP TABLE IF EXISTS "public"."customized_resumes" CASCADE;
DROP TABLE IF EXISTS "public"."base_resume" CASCADE;

-- Step 3: Drop job-scraper functions
DROP FUNCTION IF EXISTS "public"."get_applied_jobs_sorted"(integer, integer) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_applied_jobs_sorted"(integer, integer, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_applied_jobs_sorted"(integer, integer, text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_applied_jobs_sorted"(integer, integer, text, text, text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_jobs_for_rescore"(integer) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_jobs_for_resume_generation_custom_sort"(integer, integer) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_top_scored_jobs_custom_sort"(integer, integer, text, integer, integer, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_top_scored_jobs_custom_sort"(integer, integer, text, integer, integer, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."update_last_updated_column"() CASCADE;
DROP FUNCTION IF EXISTS "public"."update_base_resume_updated_at_column"() CASCADE;
DROP FUNCTION IF EXISTS "public"."merge_historical_repost_plan"() CASCADE;
DROP FUNCTION IF EXISTS "public"."calculate_listing_posting_waves"(jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."prevent_listing_observation_mutation"() CASCADE;

-- Step 4: Drop storage bucket for personalized_resumes
DELETE FROM storage.objects WHERE bucket_id = 'personalized_resumes';
DELETE FROM storage.buckets WHERE id = 'personalized_resumes';

-- Step 5: Keep 'resumes' bucket (may be shared), but clean it if desired
-- Uncomment below to clear the resumes bucket:
-- DELETE FROM storage.objects WHERE bucket_id = 'resumes';

COMMIT;

-- Verification: List remaining non-rmc tables
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name NOT LIKE 'rmc_%'
ORDER BY table_name;

-- Verification: List remaining non-rmc functions
SELECT routine_name FROM information_schema.routines 
WHERE routine_schema = 'public' AND routine_name NOT LIKE 'rmc_%'
ORDER BY routine_name;
