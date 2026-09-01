# AI State: Database-Configured Career Lanes

## Goal

Refactor `job-scraper` and `job-scraper-web` from one hardcoded `software_tpm` search into configurable, database-backed career lanes while preserving existing behavior and canonical job identity.

## Canonical lanes

- `technology_delivery` (`software_tpm` is an explicit compatibility alias)
- `systems_platform_ops`
- `network_infrastructure`
- `datacenter_operations`
- `ai_workflow_automation`
- `building_controls`

Definitions and initial search/filter context came from `/Users/tmaerz/projects/resgen/candidate/lanes.md`. Precision/recall queries are seeded in the web migration and editable in `/config`.

## Implemented here

- `lane_catalog.py`: canonical lane metadata and alias.
- `scrape_configuration.py`: strict contract for `get_scraper_configuration()`, DB-default source, complete file/env overrides, deterministic Canada/USA/EEA expansion.
- `scraper.py`: database-configured query execution and provenance.
- `supabase_utils.py`: atomic membership writes, lane-specific state helpers, source-to-canonical mapping, and generated geography write protection.
- `downstream_orchestration.py`: deterministic all-enabled-lane workers.
- `scheduled_scoring.py`: database `score_jobs` gate.
- `analyze_jobs.py`, `score_jobs.py`, `custom_resume_generator.py`: lane-scoped processing.
- `lane_resume_storage.py`: immutable lane/job/version resume paths.
- Score and resume workers use owner-checked, expiring database claims so overlapping runs cannot process the same membership.
- Workflows: blank archetype processes all enabled lanes; explicit canonical slug remains supported.

## Shared database contract

`job-scraper-web/supabase/migrations/202609010001_configurable_career_lanes.sql` defines configuration tables, audited revisions, `(job_id, archetype)` memberships, resume profiles, protected RPCs, membership-aware web reads, and historical `software_tpm` compatibility without rewriting `jobs.archetype`.

## Verification

- Changed Python modules compile with `python3 -m compileall`.
- Full Python suite passes: 305 tests, with two Supabase client deprecation warnings.
- Web tests, TypeScript, production build, and lint complete; lint reports pre-existing warnings.
- Multiple read-only release reviews were run and identified blockers were fixed.

## Working-tree warning

Unrelated user modifications existed before this task, including FreeHire workflow/compatibility changes and untracked agent configuration. Do not discard them. No commit, push, or remote migration was performed.

## Rollout

1. Review and apply the web migration through the normal Supabase workflow.
2. Configure web administrator authentication.
3. Review `/config` seeds, filters, and geographies.
4. Add reviewed base resume profiles for resume-dependent lanes.
5. Run manual samples and tune precision before broad scheduling.
