# LinkedIn ingestion relevance audit - 2026-09-07

## Scope and method

- Population: 11,649 `(job_id, archetype)` memberships covering 9,656 canonical jobs and all six enabled career lanes.
- Audit frame: all configurable-lane memberships for the five new lanes; the latest seven days for `technology_delivery` so its older legacy history did not dominate.
- Residual-noise sample: deterministic simple random sample without replacement of 25 non-filtered memberships per lane, ordered by `md5(job_id || ':audit-2026-09-07')`.
- Recall guard: deterministic samples of up to 15 already-filtered memberships per lane, with a census for the two network and 15 datacenter rows.
- Labels: relevant, borderline, clearly irrelevant, or insufficient evidence after reading the full title, company, and description. Borderline jobs were treated as relevant when validating new filters.
- Intervals: two-sided 95% Wilson intervals. At these sampling fractions, finite-population correction is negligible.
- Limitation: these are model-assisted role reviews, not human double-coded gold labels. The audit estimates precision among retrieved jobs; no retrieval-only audit can establish absolute LinkedIn recall.

## Baseline retrieval cost

Observed from 2026-09-01 through the audit:

| Archetype | Search pages | Cards | New workflow sources | Detail tasks first attributed |
|---|---:|---:|---:|---:|
| `technology_delivery` | 1,764 | 17,231 | 6,401 | 1,681 |
| `systems_platform_ops` | 1,040 | 10,043 | 2,805 | 701 |
| `network_infrastructure` | 809 | 7,750 | 1,282 | 229 |
| `datacenter_operations` | 650 | 6,132 | 1,555 | 210 |
| `ai_workflow_automation` | 998 | 9,700 | 2,980 | 777 |
| `building_controls` | 939 | 9,094 | 3,615 | 1,030 |
| **Total** | **6,200** | **59,950** | **18,638** | **4,628** |

`new workflow sources` is scope-level workflow evidence and is not additive across lanes. Detail-task attribution uses the first observing lane because detail requests are globally deduplicated.

## Baseline relevance results

| Archetype | Relevant | Borderline | Clearly irrelevant | Estimated clear irrelevance (95% CI) |
|---|---:|---:|---:|---:|
| `technology_delivery` | 6 | 5 | 14 | 56.0% (37.1%-73.3%) |
| `systems_platform_ops` | 7 | 3 | 15 | 60.0% (40.7%-76.6%) |
| `network_infrastructure` | 1 | 8 | 15 | 60.0% (40.7%-76.6%) |
| `datacenter_operations` | 1 | 0 | 24 | 96.0% (80.5%-99.3%) |
| `ai_workflow_automation` | 10 | 4 | 11 | 44.0% (26.7%-62.9%) |
| `building_controls` | 9 | 0 | 16 | 64.0% (44.5%-79.8%) |

Dominant clusters were non-technology program work, application/data/software engineering, SRE and generic IT, facilities and field-service work, ML research/platform work, financial/project controls, and generic electrical/mechanical work with incidental controls terminology.

## Existing-filter recall audit

| Archetype | Reviewed filtered rows | False exclusions | Borderline exclusions | Main unsafe rule |
|---|---:|---:|---:|---|
| `technology_delivery` | 15 | 5 | 1 | Description-wide `product manager` / `scrum master`, company `jobgether`, generic `coordinator` |
| `systems_platform_ops` | 15 | 9 | 2 | Description-wide `data platform` / `ML platform` |
| `network_infrastructure` | 2 | 0 | 0 | No observed false exclusion; description-wide `full stack` remained structurally risky |
| `datacenter_operations` | 15 | 0 | 0 | No observed false exclusion |
| `ai_workflow_automation` | 15 | 2 | 1 | Description-wide `model training`, including negated/incidental mentions |
| `building_controls` | 15 | 2 | 5 | `sales engineer` substring and description-wide `test automation` |

The broad description exclusions were removed. Unambiguous role-title exclusions replaced them. The hard-coded Jobgether company exclusion and generic coordinator exclusion were also removed.

## LinkedIn query behavior

LinkedIn documents uppercase `AND`, `OR`, `NOT`, quotes, and parentheses for its interactive search UI:

- https://www.linkedin.com/help/linkedin/answer/a524335

The guest endpoint did not enforce those semantics in direct tests. Contradictory queries such as `"Network Engineer" NOT "Network"` and additions such as `AND qzxxnonexistenttoken` continued returning Network Engineer jobs. One-page controlled probes also showed that extra keyword clauses often changed ranking without reliably enforcing constraints.

Operational conclusion: treat `jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=` as fuzzy free text. Use short, role-specific natural-language phrases and enforce unavoidable ambiguity after descriptions are fetched. Do not rely on Boolean expressions as scrape-time filters.

## Applied query set

The active query count was reduced from 40 to 13 (67.5%). Removed queries remain retired for provenance.

| Archetype | Active queries |
|---|---|
| `technology_delivery` | `technical program delivery manager`; `IT technology project manager`; `gestionnaire de projet informatique TI` |
| `systems_platform_ops` | `VMware virtualization infrastructure administrator`; `systems administrator infrastructure operations` |
| `network_infrastructure` | `Network Engineer`; `Network Administrator` |
| `datacenter_operations` | `Data Center Technician`; `Data Centre Technician` |
| `ai_workflow_automation` | `agentic AI engineer RAG workflow automation`; `AI solutions engineer LLM workflow integration` |
| `building_controls` | `building automation controls specialist BAS BACnet`; `industrial controls engineer PLC SCADA commissioning` |

The two datacenter spellings are retained because adding hardware/rack terms did not improve its 10-card live probe and the current configuration contract requires one enabled precision and recall query. Datacenter precision therefore depends primarily on the downstream evidence gate.

## Downstream changes

- Description-based routing now requires two lane-specific evidence families for systems, network, datacenter, AI workflow, and controls lanes. Regex lookaheads are anchored with `\A` to avoid quadratic scans of long descriptions.
- Ambiguous jobs remain `filter_status='review'` and visible for inspection.
- Scoring, keyword analysis, and custom-resume workers now claim only `filter_status='included'`; previously they accepted every row where `is_filtered=false`, including pending and review rows.
- The filter-state backfill changed 2,467 memberships for the first revision and 1,218 for evidence hardening. A repeat dry run reported zero changes.
- The current unscored worker queue is 2,839 included jobs. Old `is_filtered=false` semantics would expose 6,419 jobs, so 3,580 review jobs (55.8%) no longer consume downstream work.

## Post-change validation

Replaying the original samples through the final filters retained every sampled relevant job:

| Archetype | Sampled relevant retained | Clearly irrelevant still included |
|---|---:|---:|
| `technology_delivery` | 6/6 | 9/14 |
| `systems_platform_ops` | 7/7 | 0/15 |
| `network_infrastructure` | 1/1 | 0/15 |
| `datacenter_operations` | 1/1 | 0/24 |
| `ai_workflow_automation` | 10/10 | 0/11 |
| `building_controls` | 9/9 | 2/16 |

Technology Delivery's residual historical noise is mostly generic project/program titles that cannot be rejected safely from title and description terms alone. Its narrower source queries are the safer remedy. The two remaining Building Controls false positives are generic trade roles whose descriptions contain genuine control-system evidence; they remain preferable to false-excluding valid PLC-heavy electricians and commissioning roles.

Final membership state:

| Archetype | Included | Review | Filtered | Pending |
|---|---:|---:|---:|---:|
| `technology_delivery` | 5,362 | 395 | 345 | 0 |
| `systems_platform_ops` | 659 | 511 | 171 | 0 |
| `network_infrastructure` | 235 | 536 | 25 | 0 |
| `datacenter_operations` | 48 | 699 | 41 | 0 |
| `ai_workflow_automation` | 480 | 666 | 78 | 0 |
| `building_controls` | 400 | 963 | 35 | 0 |

## Rollback and monitoring

- Pre-audit configuration is preserved in revision 5. Revision 9 records the migration-backed final configuration.
- Monitor at least two completed discovery cycles. Compare cards, unique workflow sources, included rate, review rate, and manually reviewed precision against this baseline.
- Expand the audit by another 25 labels for any lane whose observed included precision degrades or whose relevant unique yield drops materially. Restore individual retired queries rather than restoring the full 40-query set.
