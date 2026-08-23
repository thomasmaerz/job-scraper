from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_sql_drops_job_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "cleanup.sql").read_text()

    assert 'DROP TABLE IF EXISTS "public"."job_keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."jobs" CASCADE;' in sql
    assert 'DROP FUNCTION IF EXISTS "public"."update_last_updated_column"() CASCADE;' in sql


def test_add_job_insights_sql_is_self_contained_for_last_updated_trigger():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'CREATE OR REPLACE FUNCTION "public"."update_last_updated_column"() RETURNS "trigger"' in sql


def test_init_sql_uses_if_not_exists_for_setup_indexes():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_insights_analyzed_at"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_insights_reanalyzed_at"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_category"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_count"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_job_id"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_keyword_category"' in sql


def test_init_sql_adds_job_archetype_provenance_columns():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'ADD COLUMN IF NOT EXISTS "search_query" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "filter_profile" text' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_archetype"' in sql


def test_add_job_insights_sql_scopes_fact_and_aggregate_tables_by_archetype():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert '"archetype" text NOT NULL' in sql
    assert 'PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql
    assert 'PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql


def test_add_job_insights_sql_migrates_existing_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'ALTER TABLE "public"."keyword_insights"' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "provider" text' in sql
    assert "UPDATE \"public\".\"keyword_insights\"" in sql
    assert "SET \"archetype\" = 'software_tpm'" in sql
    assert "SET \"provider\" = 'unknown'" in sql
    assert 'ALTER COLUMN "provider" SET NOT NULL' in sql
    assert 'DROP CONSTRAINT IF EXISTS "keyword_insights_pkey"' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql


def test_add_job_insights_sql_migrates_existing_job_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'ALTER TABLE "public"."job_keyword_insights"' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "provider" text' in sql
    assert "UPDATE \"public\".\"job_keyword_insights\"" in sql
    assert "SET \"archetype\" = 'software_tpm'" in sql
    assert 'DROP CONSTRAINT IF EXISTS "job_keyword_insights_pkey"' in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql


def test_init_sql_scopes_insight_tables_by_archetype_for_fresh_installs():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'CREATE TABLE IF NOT EXISTS "public"."keyword_insights" (' in sql
    assert 'CREATE TABLE IF NOT EXISTS "public"."job_keyword_insights" (' in sql
    assert '"archetype" text NOT NULL' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_archetype_category"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_archetype"' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("keyword", "category")' not in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "keyword", "category")' not in sql


def test_init_sql_actionable_rpcs_exclude_filtered_jobs_consistently():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'AND j.is_filtered = FALSE\n        AND j.customized_resume_id IS NOT NULL' in sql
    assert 'AND j.is_filtered = FALSE\n        AND j.resume_score >= 50' in sql
    assert sql.count('AND j.is_filtered = FALSE') >= 4


def test_posting_wave_migration_adds_count_and_shared_calculator():
    sql = (ROOT / "supabase_setup" / "add_posting_wave_semantics.sql").read_text()

    assert "ADD COLUMN IF NOT EXISTS posting_wave_count" in sql
    assert "CREATE OR REPLACE FUNCTION public.calculate_listing_posting_waves" in sql
    assert "instance->>'location'" in sql
    assert "instance->>'scrape_run_id'" in sql
    assert "GREATEST(result.posting_wave_count - 1, 0)" in sql
    assert "GRANT EXECUTE ON FUNCTION public.calculate_listing_posting_waves(jsonb) TO service_role" in sql


def test_historical_merge_uses_location_and_posting_wave_calculator():
    sql = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()

    assert "'location', source_snapshot->>'location'" in sql
    assert "public.calculate_listing_posting_waves(lv.raw_listing_instances)" in sql
    assert "posting_wave_count = w.posting_wave_count" in sql
    assert "repost_count = w.repost_count" in sql
    assert "GREATEST(l.seen_count - 1, 0)" not in sql
