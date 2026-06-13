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
    assert 'PRIMARY KEY ("archetype", "keyword", "category")' in sql
    assert 'PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql


def test_add_job_insights_sql_migrates_existing_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'ALTER TABLE "public"."keyword_insights"' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "provider" text' in sql
    assert "UPDATE \"public\".\"keyword_insights\"" in sql
    assert "SET \"archetype\" = 'software_tpm'" in sql
    assert 'DROP CONSTRAINT IF EXISTS "keyword_insights_pkey"' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "keyword", "category")' in sql


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
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "keyword", "category")' in sql
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
