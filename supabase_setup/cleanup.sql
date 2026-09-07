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
ALTER TABLE IF EXISTS "public"."freehire_publication_generations"
    DROP COLUMN IF EXISTS "source_discovery_cycle_id",
    DROP COLUMN IF EXISTS "source_discovery_sequence";
ALTER TABLE IF EXISTS "public"."freehire_publication_state"
    DROP COLUMN IF EXISTS "source_discovery_cycle_id",
    DROP COLUMN IF EXISTS "source_discovery_sequence";
ALTER TABLE IF EXISTS "public"."scrape_run_state"
    DROP COLUMN IF EXISTS "last_successful_discovery_cycle_id",
    DROP COLUMN IF EXISTS "last_successful_discovery_sequence";
DROP TABLE IF EXISTS "public"."linkedin_discovery_requirement_acceptances" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_task_attempts" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_coverage_debt_attempts" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_cycle_resolutions" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_requirements" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_tasks" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_cycle_sources" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_ingestion_page_sources" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_ingestion_pages" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_coverage_debt" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_cycle_scopes" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_discovery_cycles" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_scope_coverage_state" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_source_request_grants" CASCADE;
DROP TABLE IF EXISTS "public"."linkedin_source_request_policy" CASCADE;
DROP TABLE IF EXISTS "public"."canonical_provider_revisions" CASCADE;
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
DROP FUNCTION IF EXISTS "public"."claim_freehire_compat_jobs"(jsonb, text, timestamptz) CASCADE;
DROP FUNCTION IF EXISTS "public"."persist_freehire_compat_results"(jsonb, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."apply_freehire_compat_metadata_batch"(jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."claim_freehire_compat_job"(text, text, jsonb, text, timestamptz) CASCADE;
DROP FUNCTION IF EXISTS "public"."replace_job_keyword_facts_and_refresh_aggregates"(text[], text, jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."rebuild_keyword_insights_atomic"() CASCADE;
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
DROP FUNCTION IF EXISTS "public"."replace_historical_repost_plan"(jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."merge_historical_repost_plan"() CASCADE;
DROP FUNCTION IF EXISTS "public"."calculate_listing_posting_waves"(jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."prevent_listing_observation_mutation"() CASCADE;
DROP FUNCTION IF EXISTS "public"."increment_job_canonical_revision"() CASCADE;
DROP FUNCTION IF EXISTS "public"."bump_canonical_provider_revision"() CASCADE;
DROP FUNCTION IF EXISTS "public"."get_canonical_provider_revision"(text) CASCADE;
DROP FUNCTION IF EXISTS "public"."acquire_linkedin_request_grant"(text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."consume_linkedin_request_grant"(uuid, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."finish_linkedin_request_grant"(uuid, text, text, integer) CASCADE;
DROP FUNCTION IF EXISTS "public"."open_linkedin_source_circuit"(uuid, text, text, integer) CASCADE;
DROP FUNCTION IF EXISTS "public"."reset_linkedin_source_circuit"(text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."create_linkedin_discovery_cycle"(uuid, bigint, text, text, jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_resumable_linkedin_discovery_cycle"(boolean, text[]) CASCADE;
DROP FUNCTION IF EXISTS "public"."commit_linkedin_discovery_page"(jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."finish_linkedin_discovery_scope"(uuid, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."fail_linkedin_discovery_cycle"(bigint, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."expire_linkedin_coverage_debt"(text, timestamptz) CASCADE;
DROP FUNCTION IF EXISTS "public"."prepare_linkedin_discovery_scope_state"(text[], timestamptz) CASCADE;
DROP FUNCTION IF EXISTS "public"."accept_linkedin_coverage_debt"(bigint, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."advance_linkedin_discovery_watermark"() CASCADE;
DROP FUNCTION IF EXISTS "public"."resolve_failed_linkedin_discovery_cycle"(bigint, bigint, text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."resolve_eligible_failed_linkedin_discovery_cycles"(bigint) CASCADE;
DROP FUNCTION IF EXISTS "public"."accept_linkedin_discovery_requirement"(bigint, uuid, text, text, text, text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."seal_linkedin_discovery_cycle"(bigint, boolean) CASCADE;
DROP FUNCTION IF EXISTS "public"."claim_linkedin_discovery_tasks"(text, integer, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."heartbeat_linkedin_discovery_task"(bigint, text, uuid) CASCADE;
DROP FUNCTION IF EXISTS "public"."apply_linkedin_discovery_task_canonical"(bigint, text, uuid, jsonb) CASCADE;
DROP FUNCTION IF EXISTS "public"."reject_linkedin_requirement_provenance_change"() CASCADE;
DROP FUNCTION IF EXISTS "public"."transition_linkedin_discovery_task"(bigint, text, uuid, text, text, text) CASCADE;
DROP FUNCTION IF EXISTS "public"."finalize_freehire_publication_v2"(bigint) CASCADE;
DROP FUNCTION IF EXISTS "public"."get_linkedin_discovery_status"() CASCADE;

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
