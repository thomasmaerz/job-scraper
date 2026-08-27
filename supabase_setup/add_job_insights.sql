-- Job insights analysis schema migration.
-- Run this once in Supabase SQL Editor for existing databases.

BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

ALTER TABLE "public"."jobs"
ADD COLUMN IF NOT EXISTS "insights_analyzed_at" timestamp with time zone;

ALTER TABLE "public"."jobs"
ADD COLUMN IF NOT EXISTS "insights_reanalyzed_at" timestamp with time zone;

CREATE OR REPLACE FUNCTION "public"."update_last_updated_column"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
   NEW.last_updated = now();
   RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS "public"."keyword_insights" (
    "archetype" text NOT NULL,
    "keyword" text NOT NULL,
    "category" text NOT NULL,
    "count" integer DEFAULT 0 NOT NULL,
    "provider" text DEFAULT 'unknown'::text NOT NULL,
    "last_updated" timestamp with time zone DEFAULT now(),
    CONSTRAINT "keyword_insights_category_check"
        CHECK ("category" IN ('skill', 'technology', 'certification', 'attribute')),
    CONSTRAINT "keyword_insights_count_check"
        CHECK ("count" >= 0)
);

CREATE TABLE IF NOT EXISTS "public"."job_keyword_insights" (
    "job_id" text NOT NULL,
    "archetype" text NOT NULL,
    "keyword" text NOT NULL,
    "category" text NOT NULL,
    "provider" text,
    "analyzed_at" timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category"),
    CONSTRAINT "job_keyword_insights_category_check"
        CHECK ("category" IN ('skill', 'technology', 'certification', 'attribute'))
);

ALTER TABLE "public"."keyword_insights"
ADD COLUMN IF NOT EXISTS "archetype" text;

ALTER TABLE "public"."keyword_insights"
ADD COLUMN IF NOT EXISTS "provider" text;

UPDATE "public"."keyword_insights"
SET "archetype" = 'software_tpm'
WHERE "archetype" IS NULL;

UPDATE "public"."keyword_insights"
SET "provider" = 'unknown'
WHERE "provider" IS NULL;

ALTER TABLE "public"."keyword_insights"
ALTER COLUMN "archetype" SET NOT NULL;

ALTER TABLE "public"."keyword_insights"
ALTER COLUMN "provider" SET DEFAULT 'unknown';

ALTER TABLE "public"."keyword_insights"
ALTER COLUMN "provider" SET NOT NULL;

ALTER TABLE "public"."keyword_insights"
DROP CONSTRAINT IF EXISTS "keyword_insights_pkey";

ALTER TABLE "public"."keyword_insights"
ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "provider", "keyword", "category");

ALTER TABLE "public"."job_keyword_insights"
ADD COLUMN IF NOT EXISTS "archetype" text;

ALTER TABLE "public"."job_keyword_insights"
ADD COLUMN IF NOT EXISTS "provider" text;

UPDATE "public"."job_keyword_insights"
SET "archetype" = 'software_tpm'
WHERE "archetype" IS NULL;

ALTER TABLE "public"."job_keyword_insights"
ALTER COLUMN "archetype" SET NOT NULL;

ALTER TABLE "public"."job_keyword_insights"
DROP CONSTRAINT IF EXISTS "job_keyword_insights_pkey";

ALTER TABLE "public"."job_keyword_insights"
ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category");

CREATE INDEX IF NOT EXISTS "idx_jobs_insights_analyzed_at"
ON "public"."jobs" USING btree ("insights_analyzed_at");

CREATE INDEX IF NOT EXISTS "idx_jobs_insights_reanalyzed_at"
ON "public"."jobs" USING btree ("insights_reanalyzed_at");

CREATE INDEX IF NOT EXISTS "idx_keyword_insights_category"
ON "public"."keyword_insights" USING btree ("category");

CREATE INDEX IF NOT EXISTS "idx_keyword_insights_archetype_category"
ON "public"."keyword_insights" USING btree ("archetype", "category");

CREATE INDEX IF NOT EXISTS "idx_keyword_insights_count"
ON "public"."keyword_insights" USING btree ("count" DESC);

CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_job_id"
ON "public"."job_keyword_insights" USING btree ("job_id");

CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_archetype"
ON "public"."job_keyword_insights" USING btree ("archetype");

CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_keyword_category"
ON "public"."job_keyword_insights" USING btree ("keyword", "category");

CREATE OR REPLACE TRIGGER "update_keyword_insights_last_updated"
BEFORE UPDATE ON "public"."keyword_insights"
FOR EACH ROW EXECUTE FUNCTION "public"."update_last_updated_column"();

ALTER TABLE "public"."keyword_insights" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "public"."job_keyword_insights" ENABLE ROW LEVEL SECURITY;

GRANT ALL ON TABLE "public"."keyword_insights" TO "anon";
GRANT ALL ON TABLE "public"."keyword_insights" TO "authenticated";
GRANT ALL ON TABLE "public"."keyword_insights" TO "service_role";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "anon";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "authenticated";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "service_role";

COMMIT;
