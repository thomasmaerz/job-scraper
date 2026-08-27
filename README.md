# Job Scraper & Application Assistant

This project is a comprehensive suite of tools designed to automate and enhance the job searching process, primarily focusing on LinkedIn. It scrapes job postings, parses resumes, scores job suitability against a candidate's resume, analyzes job market keywords, manages job application statuses, and can even generate custom PDF resumes. The system leverages AI through a unified LLM client and Supabase for data storage.

## Features

- **Job Scraping**: Automatically scrapes job postings. ([scraper.py](scraper.py))
- **Resume Parsing**:
  - Extracts text from PDF resumes using `pdfplumber`. ([resume_parser.py](resume_parser.py))
  - Utilizes Google Gemini AI to parse resume text into structured data ([parse_resume_with_ai.py](parse_resume_with_ai.py))
- **Job Scoring**: Scores job descriptions against a parsed resume using AI to determine suitability. ([score_jobs.py](score_jobs.py))
- **Job Market Insights**: Extracts recurring keywords from unanalyzed jobs into per-job facts in `job_keyword_insights` and aggregate totals in `keyword_insights` using `analyze_jobs.py` for downstream reporting and trend analysis. Scheduled runs analyze unanalyzed jobs in any state (new or expired) up to `JOB_INSIGHTS_MAX_JOBS` per run, with aggregates split by archetype and provider; manual replacement backfill is available for one-time reanalysis of previously analyzed jobs. ([analyze_jobs.py](analyze_jobs.py))
- **Universal LLM Support**: Supports 400+ model providers (Gemini, OpenAI, Anthropic, Ollama, Groq, etc.) via a unified abstraction layer. ([llm_client.py](llm_client.py))
- **Job Management**:
  - Tracks the status of job applications.
  - Marks old or inactive jobs as expired.
  - Periodically checks if active jobs are still available.
    ([job_manager.py](job_manager.py))
- **Data Storage**: Uses Supabase to store job data, resume details, and application statuses. (Utility functions in [supabase_utils.py](supabase_utils.py))
- **Canonical Listing History**: Preserves every source listing ID while distinguishing simultaneous variants, location variants, and confirmed later posting waves.
- **Custom PDF Resume Generation**: Generates ATS-friendly PDF resumes from structured resume data. ([pdf_generator.py](pdf_generator.py))
- **AI-Powered Text Processing**: Leverages any configured LLM for tasks like resume parsing and job description formatting.
- **Quota Management**: Built-in rate limiting, exponential backoff, and daily budget tracking for LLM API calls. Features task-specific Gemini fallback chains for job scoring and job insights extraction.
- **Automated Workflows**: Includes optimized GitHub Actions for running tasks on a schedule without exhausting quotas. ([workflows](.github/workflows/))

### LLM Routing Notes

- Job scoring uses a task-specific scoring chain via `job_scoring_client`.
- Job insights analysis uses a task-specific extraction chain via `job_insights_client`.
- Resume parsing and resume generation still use the generic `primary_client` and are not yet routed through dedicated task-specific chains.

## Tech Stack

- **Programming Language**: Python 3.11.9
- **Web Scraping/HTTP**:
  - `requests`
  - `httpx`
  - `BeautifulSoup4` (for HTML parsing)
  - `Playwright` (for browser automation)
- **PDF Processing**:
  - `pdfplumber` (for text extraction)
  - `ReportLab` (for PDF generation)
- **AI/LLM**: `litellm` (Universal proxy supporting Gemini, OpenAI, Claude, etc.), `google-genai`
- **Database**: Supabase (`supabase`)
- **Data Validation**: `Pydantic`
- **Environment Management**: `python-dotenv`
- **Text Conversion**: `html2text`
- **CI/CD**: GitHub Actions

## Setup and Installation

This project is designed to run primarily through GitHub Actions. Follow these steps to set it up for your own use:

1.  **Fork the Repository:**
    - Click the "Fork" button at the top right of this page to create a copy of this repository in your own GitHub account.

2.  **Create a Supabase Project:**
    - Go to [Supabase](https://supabase.com/) and create a new project.
    - Once your project is created, navigate to the "SQL Editor" section.
    - Open the `supabase_setup/init.sql` file from this repository, copy its content, and run it in your Supabase SQL Editor. This will set up the necessary tables (like `jobs`, `customized_resumes`, `keyword_insights`, `job_keyword_insights`, and `base_resume`) and storage buckets (`resumes`, `personalized_resumes`).
    - If you already have an existing Supabase project using an older schema, run `supabase_setup/add_job_insights.sql` once in the Supabase SQL Editor to add `jobs.insights_analyzed_at`, `jobs.insights_reanalyzed_at`, `keyword_insights`, and `job_keyword_insights`.
    - Scheduled `Analyze Job Insights` runs only process new unanalyzed jobs. Use the manual `replacement_backfill` workflow input only when you explicitly want a one-time reanalysis of previously analyzed jobs.

3.  **Obtain API Keys for Your LLM Provider:**
    - Get API key(s) from your chosen provider (e.g., [Google AI Studio](https://aistudio.google.com/app/apikey), [OpenAI](https://platform.openai.com/api-keys), [Anthropic](https://console.anthropic.com/), etc.).

4.  **Configure GitHub Repository Secrets and Variables:**
    - In your forked GitHub repository, go to "Settings".
    - In the left sidebar, navigate to "Secrets and variables" under the "Security" section, and then click on "Actions".
    - **Add Repository Secrets** (Click "New repository secret"):
      - `LLM_API_KEY`: Your primary LLM API key (e.g., for Gemini or Groq). Also accepts legacy `GEMINI_FIRST_API_KEY`.
      - `OPENAI_API_KEY`: (Optional) Your OpenAI API key if using GPT models.
      - `ANTHROPIC_API_KEY`: (Optional) Your Anthropic API key if using Claude models.
      - `GROQ_API_KEY`: (Optional) Your Groq API key if using Groq models.
      - `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase project's `service_role` key.
      - `SUPABASE_URL`: Your Supabase project's URL.

    - > **Note:** Other non-sensitive variables like `LLM_MODEL`, `LLM_MAX_RPM`, and `JOBS_TO_SCORE_PER_RUN` are now hardcoded in `config.py` as safe defaults. You only need to set them as GitHub Variables if you want to override the `config.py` defaults (though this is no longer the recommended approach).

5.  **Upload Your Resume to Supabase Storage:**
    - In your Supabase project dashboard, navigate to **Storage** in the left sidebar.
    - Find the **`resumes`** bucket (created by the `init.sql` script in step 2).
    - Click on the bucket, then click **"Upload files"** and upload your resume. **The file must be named `resume.pdf`**.
    - > **⚠️ Security Note:** Your resume is stored securely in your private Supabase Storage bucket — it is **never committed to the public GitHub repository**. This protects your personal information (name, email, phone, address, etc.) from being publicly visible.

6.  **Parse Your Resume:**
    - Go to the "Actions" tab in your forked GitHub repository.
    - Find the workflow named "Parse Resume Manually" in the list of workflows.
    - Click on it, and then click the "Run workflow" button. This will trigger the `resume_parser.py` script, which will download your `resume.pdf` from Supabase Storage, parse it using AI, and store the structured data securely in the `base_resume` table in your Supabase database.

7.  **Configure Job Search Parameters (Edit `config.py`):**
    - In your forked GitHub repository, navigate to the [config.py](config.py) file.
    - Edit the file to customize your job search preferences. The main variables you'll likely want to change are:

      ```python
      # --- LinkedIn Search Configuration ---
      LINKEDIN_SEARCH_QUERIES = ["maths lecturer", "statistics lecturer"] # Your keywords
      LINKEDIN_LOCATION = "Singapore" # Target location
      LINKEDIN_GEO_ID = 102454443 # Geo ID (Singapore: 102454443, Dubai: 100205264)
      LINKEDIN_JOB_TYPE = "F" # "F" for Full-time
      LINKEDIN_JOB_POSTING_DATE = "r86400" # "r86400" for past 24 hours

      # --- Careers Future Search Configuration ---
      CAREERS_FUTURE_SEARCH_QUERIES = ["IT Support", "Full Stack Web Developer"]
      CAREERS_FUTURE_SEARCH_CATEGORIES = ["Information Technology"]

      # --- LLM configuration ---
      # For a full list of 100+ supported providers and model naming schemes, see:
      # https://docs.litellm.ai/docs/providers

      LLM_MODEL = "gemini"            # Model to use
      LLM_MAX_RPM = 10                # Max requests per minute
      LLM_REQUEST_DELAY_SECONDS = 8   # Delay between calls

      # --- Processing Limits ---
      JOBS_TO_SCORE_PER_RUN = 1       # Scaled for free tier
      MAX_JOBS_PER_SEARCH = {
          "linkedin": 2,
          "careers_future": 10,
      }
      ```

    - **IMPORTANT**: Do not modify other variables in `config.py` as they are carefully calibrated to prevent rate limiting and potential account bans. Only edit the search queries and location parameters shown above.
    - Commit the changes to your `config.py` file in your repository.

8.  **Enable GitHub Actions:**
    - Go to the "Actions" tab in your forked GitHub repository.
    - You will see a message saying "Workflows aren't running on this repository". Click the "Enable Actions on this repository" button (or a similar prompt) to allow the scheduled workflows to run automatically.
    - Ensure all workflows listed (e.g., `scrape_jobs.yml`, `score_jobs.yml`, `job_manager.yml`, `analyze_jobs.yml`) are enabled. If any are disabled, you may need to enable them individually.

## Automated Workflows

Once the setup is complete and GitHub Actions are enabled, the workflows defined in [workflows](.github/workflows/) are scheduled to run automatically:

- **`scrape_jobs.yml`**: Periodically scrapes new job postings from LinkedIn and CareersFuture based on your `config.py` settings and saves them to your Supabase database.
- **`score_jobs.yml`**: Periodically scores the newly scraped jobs and jobs with custom resumes against your parsed resume / custom resume and updates the scores in the database.
- **`job_manager.yml`**: Periodically manages job statuses (e.g., marks old jobs as expired, checks if active jobs are still available).
- **`analyze_jobs.yml`**: Runs `analyze_jobs.py` on a schedule or manually to extract recurring market keywords from unanalyzed jobs and update `keyword_insights`. Manual dispatch also supports a `drain_backlog` input to keep processing until no eligible unanalyzed jobs remain.
- **`freehire_compat.yml`**: Frontfills pending or changed LinkedIn rows after scraping and every four hours using strict token-budgeted Freehire category batches. Manual dispatch supports bounded, draining, and replacement passes.
- **`hourly_resume_customization.yml`**: (If enabled and configured) May run tasks related to customizing resumes for specific jobs.

You can monitor the execution of these actions in the "Actions" tab of your repository.

## Usage

After the initial setup and the "Parse Resume Manually" action has successfully run, the system will operate automatically through the scheduled GitHub Actions.

You can interact with the data directly through your Supabase dashboard to view scraped jobs, your parsed resume, and job scores.

### Web Interface for Viewing Data

A Next.js web application is available to view and manage the scraped jobs, your resume details, job scores, and job market insights from the database.

- **Repository:** [jobs-scrapper-web](https://github.com/anandanair/jobs-scraper-web)
- **Setup:** To use the web interface, clone the `jobs-scrapper-web` repository and follow the setup instructions provided in its `README.md` file to run it locally. This will typically involve configuring it to connect to your Supabase instance.
- **Insights UI:** The companion `zeroluck/job-scraper-web` application displays records from `keyword_insights` on its Insights page. The `job_keyword_insights` table provides idempotent per-job source facts behind those aggregates. The UI category tabs are a client-side refinement only; they do not replace server-side archetype scoping in the underlying Supabase query or RPC.

The individual Python scripts can still be run locally for development or testing, but this requires setting up a local Python environment, installing dependencies from `requirements.txt`, and creating a local `.env` file with the necessary credentials (mirroring the GitHub secrets).

### Freehire compatibility contract

Apply `supabase_setup/add_freehire_compat.sql` before enabling the frontfill. `public.freehire_jobs` is the service-role-only publication contract. It keeps canonical `job_id` as downstream identity, exposes `COALESCE(latest_job_id, job_id)` as `live_listing_id`, preserves source timestamps and metadata sidecars, and excludes candidate workflow/resume fields. Only LinkedIn rows with `freehire_compat_status='current'` and a pinned category are published; pending, processing, and failed rows are excluded. The current-status view is a persisted hash/version contract: consumers must not republish independently from raw `public.jobs`, and should compare `freehire_compat_import_hash` during complete keyset sweeps.

`is_remote` is true only for standalone visible-text `remote`; it is never inferred. `freehire_compat_input_hash` binds classification to canonical normalized title, visible description, location, LinkedIn level, canonical `job_id`, and schema/taxonomy version. `freehire_compat_import_hash` tracks every published source/projection, classification, deterministic remote, live-ID, and effective timestamp field. Claims and writes compare the expected database source snapshot, and workers reread the claimed row before classification. `backfill_freehire_compat.py` performs a complete bounded keyset sweep and defaults to dry-run; use `--apply` only after reviewing counts.

The downstream private Freehire derive/restore implementation remains a dependency outside this repository. Run source classification and preservation before derive, then order restoration as `derive -> linkedin-restore -> workbc-restore -> reindex -> supabase_out`. Restore only hash-matched source facets; keep `external_id=job_id` and use `latest_job_id` only for the live URL. This repository does not modify stock Freehire. The unrelated WorkBC `closed_reason='missing'` constraint failure and `supabase_out` datetime JSON serialization failure are explicitly out of scope.

**Local Development Setup (Optional):**

1.  **Clone your forked repository locally:**
    ```bash
    git clone https://github.com/anandanair/linkedin-jobs-scrapper
    cd linkedin-jobs-scrapper
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    playwright install # Install browser drivers for Playwright
    ```
4.  **Create a `.env` file:**
    - In the root of your local repository, create a `.env` file.
    - Add the keys and values that you configured as GitHub secrets:

      ```env
      # Essential Keys
      LLM_API_KEY="YOUR_LLM_API_KEY"
      SUPABASE_URL="YOUR_SUPABASE_URL"
      SUPABASE_SERVICE_ROLE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"

      # Note: LLM settings (MODEL, RPM, etc.) can be configured in config.py
      ```

5.  **Run scripts locally (example):**
    ```bash
    python scraper.py
    python resume_parser.py
    python score_jobs.py
    python job_manager.py
    ```

### Safe backfill order for archetype-aware filtering and insights

1. Apply `supabase_setup/init.sql` and `supabase_setup/add_job_insights.sql`.
2. Run `python backfill_filter.py` to stamp legacy LinkedIn rows with `software_tpm`, clear the removed aerospace-defense filter, and reapply archetype-aware filters.
3. Run `JOB_INSIGHTS_REPLACEMENT_BACKFILL=true JOB_INSIGHTS_ARCHETYPE=software_tpm python analyze_jobs.py` to rebuild facts and aggregates from the clean corpus.

### End-to-end rebuild verification for clean insights

1. Start from a clean archetype-scoped corpus by completing the safe backfill order above.
2. Run `JOB_INSIGHTS_REPLACEMENT_BACKFILL=true JOB_INSIGHTS_ARCHETYPE=software_tpm python analyze_jobs.py` so `job_keyword_insights` is fully replaced for the scoped corpus before `keyword_insights` is rebuilt.
3. Verify the rebuilt `keyword_insights` rows remain separated by `archetype`, `keyword`, and `category` rather than collapsing shared keywords across archetypes.
4. Confirm the rebuilt aggregate contains the expected `software_tpm` / `Python` / `technology` row with count `2` when the source facts include two matching software TPM rows.
5. Verify the server query or RPC used by the web app is explicitly archetype-scoped before any UI tab filtering is applied.
6. In the web app, verify the Insights page category tabs filter only within the already-scoped `software_tpm` dataset returned from Supabase, so switching tabs never reintroduces rows from a different archetype.

### Listing identity and repost semantics

- `seen_count` is the number of distinct source listing IDs retained in `listing_instances`. A source ID is an observation or variant, not proof of a repost.
- A posting wave groups source IDs at the same known normalized location when they share an effective `posted_at` date or a `scrape_run_id`. If both are unavailable, observations on the same UTC scrape date are grouped conservatively. Instances with unknown location remain variants but never confirm chronological reposts.
- `posting_wave_count` is the maximum wave count for any one normalized location. `repost_count` is `max(posting_wave_count - 1, 0)`.
- Recruiter-only changes and multiple source IDs in one wave are `simultaneous_variant` instances and do not increment `repost_count`.
- Exact normalized locations are required for automatic canonical matching. Cross-location records are not silently merged. Historical grouped data can retain `location_variant` instances, which remain outside `repost_count`.
- Every new instance stores `location`, `normalized_location`, `posting_wave_key`, `posting_wave_index`, and `variant_type`. Missing historical locations remain null unless recovered from an archived source snapshot.
- Guest search cards are recorded before known-ID filtering in append-only `listing_observations`, scoped to an `ingestion_runs` row. Replaying one run is idempotent, while failed or partial coverage remains explicit and cannot prove that a listing disappeared.
- A known ID becomes a relist detail candidate only after its stable card date moves forward by at least two calendar days. At least two observations must establish the prior date. One-day moves, backward dates, and late/out-of-order dates remain auditable corrections and never create relist events.
- Detail work reserves bounded capacity for accepted-but-unprojected same-ID relists before new IDs, then handles stale metadata. Pending relists remain durable in `listing_states.pending_relist_on` and are retried after failed or deferred detail fetches. `LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY` and `LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN` cap relist detail requests; the existing guest detail retry and User-Agent path is reused.
- Exact description hashes are stored in `listing_content_versions`. New hashes retain one full version and update the canonical description/fingerprint; unchanged hashes update version timestamps and observation count only. Description edits alone are content edits, not relists.
- Accepted `listing_relist_events` are deterministic lower-bound evidence. `same_id_relist_count` means "relisted at least once" rather than an exact historical count. Same-ID evidence joins the location/date posting-wave fold without increasing distinct-ID `seen_count` or double-counting a simultaneous new-ID variant.
- `ENABLE_LINKEDIN_RELIST_TRACKING=false` is the observation kill switch; `ENABLE_LINKEDIN_RELIST_EFFECTS=false` keeps shadow collection while suppressing same-ID relist effects.

Apply `supabase_setup/add_posting_wave_semantics.sql` before deploying code that writes `posting_wave_count`. Then inspect the idempotent repair dry run:

```bash
python repair_repost_history.py
```

Only after reviewing its counts, apply it explicitly:

```bash
python repair_repost_history.py --apply
```

The repair reads archived source snapshots when available, preserves every distinct source ID, fills only recoverable locations, and recalculates wave annotations and counts. Do not run the migration or `--apply` command against production without explicit authorization.

Apply `supabase_setup/add_same_id_relist_tracking.sql` before enabling observation writes. Review the same-ID repair in dry-run mode:

```bash
python backfill_same_id_relists.py --limit 500
```

Only reviewed output should be applied with `--apply`. The repair uses stored observations and listing instances only, is idempotent, and cannot invent intermediate relists, employer intent, or missing evidence. Scope is prospective guest-card detection plus an honest lower-bound backfill; authenticated LinkedIn scraping is explicitly excluded.

After the archive repair, unresolved LinkedIn source IDs can be sampled with a conservative dry run:

```bash
python backfill_historical_linkedin_locations.py --limit 50
```

Apply only reviewed recoveries with:

```bash
python backfill_historical_linkedin_locations.py --limit 50 --apply
```

The rescrape command fetches exact source IDs, writes only previously missing locations, records `location_source=linkedin_rescrape` and observation time, and recalculates waves. A failed or unavailable historical page leaves the location null. It does not infer historical `posted_at`, replace existing locations, or create listing IDs.

Stored instance salary text can be normalized without network requests:

```bash
python backfill_listing_instance_salary.py
python backfill_listing_instance_salary.py --apply
```

This fills only missing `salary_min`, `salary_max`, and `salary_currency`, preserves existing structured values, and records `salary_metadata_source=salary_text_parser`.

### Scrape recovery window

- Scheduled LinkedIn runs use a 48-hour lookback, providing overlap beyond the daily schedule.
- GitHub Actions supplies the timestamp of the last successful scraper workflow. Failed or partial runs are excluded from this watermark.
- The next run expands its lookback to elapsed time since that watermark plus six hours, capped at seven days.
- Source listing-ID deduplication and posting-wave calculation make overlapping searches idempotent.
- A last-seen listing ID is not used as a cursor because LinkedIn search ordering is not a durable total order and one ID cannot represent all configured queries.

For one-off recovery, run the scraper workflow with `lookback_hours=96` or `168`. Normal scheduled runs return to the watermark-derived 48-hour minimum automatically.

## Project Structure

```
.
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       ├── analyze_jobs.yml
│       ├── freehire_compat.yml
│       ├── hourly_resume_customization.yml
│       ├── job_manager.yml
│       ├── parse_resume.yml
│       ├── score_jobs.yml
│       └── scrape_jobs.yml
├── analyze_jobs.py              # Extracts recurring keyword insights from new jobs
├── backfill_freehire_compat.py   # Dry-run-default compatibility sweep and classifier
├── freehire_compat.py            # Shared deterministic and LLM compatibility contract
├── frontfill_freehire_compat.py  # Continuous pending-row compatibility worker
├── .gitignore                  # Specifies intentionally untracked files that Git should ignore
├── README.md                   # This file
├── config.py                   # Configuration settings (API keys, search parameters)
├── custom_resume_generator.py  # Script to generate customized resumes (if applicable)
├── job_manager.py              # Manages job statuses
├── llm_client.py               # Universal LLM abstraction (LiteLLM) with rate limiting
├── models.py                   # Pydantic models for data validation
├── pdf_generator.py            # Generates PDF resumes
├── requirements.txt            # Python dependencies
├── resume_parser.py            # Parses resume PDF from Supabase Storage and saves to DB
├── score_jobs.py               # Scores job suitability against resumes
├── scraper.py                  # Core scraping logic for LinkedIn and CareersFuture
├── supabase_setup/             # SQL scripts for Supabase database initialization
│   ├── add_job_insights.sql
│   └── init.sql
├── supabase_utils.py           # Utility functions for interacting with Supabase
└── user_agents.py              # List of user-agents for web scraping
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  **Fork the Repository:** Create your own fork of the project on GitHub.
2.  **Create a Branch:** Create a new branch in your fork for your feature or bug fix (e.g., `git checkout -b feature/your-awesome-feature` or `git checkout -b fix/issue-description`).
3.  **Make Changes:** Implement your changes in your branch.
4.  **Test Your Changes:** Ensure your changes work as expected and do not break existing functionality.
5.  **Commit Your Changes:** Commit your changes with clear and descriptive commit messages (e.g., `git commit -m 'feat: Add awesome new feature'`).
6.  **Push to Your Fork:** Push your changes to your forked repository (`git push origin feature/your-awesome-feature`).
7.  **Open a Pull Request:** Go to the original repository and open a Pull Request from your forked branch to the main branch of the original repository. Provide a clear description of your changes in the Pull Request.

Please ensure your code adheres to the existing style and that any new dependencies are added to `requirements.txt`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- This project utilizes [LiteLLM](https://docs.litellm.ai/) as a universal proxy to support 400+ LLM providers.
- Originally built with the powerful [Google Gemini API](https://ai.google.dev/models/gemini) for AI-driven text processing.
- Data storage is managed with [Supabase](https://supabase.com/), an excellent open-source Firebase alternative.
- Web scraping capabilities are enhanced by [Playwright](https://playwright.dev/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/).
- PDF generation is handled by [ReportLab](https://www.reportlab.com/).
- PDF text extraction is performed using [pdfplumber](https://github.com/jsvine/pdfplumber).

## Disclaimer

This project is for educational and personal use only. Scraping websites like LinkedIn may be against their Terms of Service. Use this tool responsibly and at your own risk. The developers of this project are not responsible for any misuse or any action taken against your account by LinkedIn or other platforms.

## Contact

If you have any questions, suggestions, or issues, please open an issue on the GitHub repository.
