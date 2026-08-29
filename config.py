import os
from dotenv import load_dotenv

load_dotenv()

# --- DO NOT MODIFY THE BELOW SECTION ---

# =================================================================
# 1. CORE SYSTEM CONFIGURATION (Do Not Modify)
# =================================================================
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE_NAME: str = "jobs"
SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME = "customized_resumes"
SUPABASE_STORAGE_BUCKET="personalized_resumes"
SUPABASE_RESUME_STORAGE_BUCKET="resumes"
SUPABASE_BASE_RESUME_TABLE_NAME = "base_resume"
BASE_RESUME_PATH = "resume.json"

# API keys — set only the key(s) needed for your chosen provider.
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_FIRST_API_KEY")

# =================================================================
# 2. USER PREFERENCES (Editable)
# =================================================================

# --- LLM Settings ---
# Use any model supported by LiteLLM (gemini, openai/gpt-4o-mini, groq/llama-3.3-70b-versatile)
# Full list of supported models & naming: https://docs.litellm.ai/docs/providers
LLM_MODEL = "gemini"

# --- Search Configuration ---
LINKEDIN_LOCATION = "Canada"
LINKEDIN_GEO_ID = 101174742      # Canada
LINKEDIN_JOB_TYPE = "F" # F=Full-time, C=Contract, P=Part-time, T=Temporary, I=Internship
LINKEDIN_LOOKBACK_HOURS = max(24, int(os.environ.get("LINKEDIN_LOOKBACK_HOURS", "48")))
LINKEDIN_JOB_POSTING_DATE = f"r{LINKEDIN_LOOKBACK_HOURS * 3600}"
LINKEDIN_LOOKBACK_OVERLAP_HOURS = max(1, int(os.environ.get("LINKEDIN_LOOKBACK_OVERLAP_HOURS", "6")))
LINKEDIN_MAX_LOOKBACK_HOURS = max(
    LINKEDIN_LOOKBACK_HOURS,
    int(os.environ.get("LINKEDIN_MAX_LOOKBACK_HOURS", "168")),
)
LINKEDIN_LAST_SUCCESS_AT = os.environ.get("LINKEDIN_LAST_SUCCESS_AT")
LINKEDIN_F_WT = "1,2,3" # 1=Onsite, 2=Remote, 3=Hybrid

CAREERS_FUTURE_SEARCH_QUERIES = []
CAREERS_FUTURE_SEARCH_CATEGORIES = []
CAREERS_FUTURE_SEARCH_EMPLOYMENT_TYPES = []

# --- Filter Configuration ---
# Jobs matching these patterns are flagged is_filtered=True before LLM scoring.
# All regex patterns are case-insensitive. Edit lists here to tune without touching code.

ARCHETYPE_CONFIGS = {
    "software_tpm": {
        "provider": "linkedin",
        "location": LINKEDIN_LOCATION,
        "filter_profile": "software_tpm_v1",
        "search_queries": [
            "IT Project Manager",
            "Technical Project Manager",
            "Information Technology Project Manager",
            "Technical Program Manager",
        ],
        "company_blocklist": [
            r"jobgether",
        ],
        "title_blocklist": [
            r"\bconstruction\b",
            r"\bland development\b",
            r"\bm&e\b",
            r"\bsubcontract\b",
            r"\bsales manager\b",
            r"\baccount manager\b",
            r"\bcustomer success\b",
            r"\bICI\b",
            r"\bclinical\b",
            r"\bCNS\b",
        ],
        "title_entry_level_blocklist": [
            r"\bcoordinator\b",
            r"\bjr\.?\s+(project|program)\b",
            r"\bjunior\b",
            r"\bassistant project manager\b",
            r"\bstaff engineer\b",
        ],
        "desc_blocklist": [
            r"construction firm",
            r"construction company",
            r"infrastructure construction",
            r"industrial construction",
            r"\bEPCM?\b",
            r"\bEPC environment\b",
            r"civil engineering",
            r"natural and built assets",
            r"\bsubtrades?\b",
            r"(?s)\bsubcontractor.{0,3000}\b(?:general contractor|preconstruction|site inspection|specialty contractor|construction management)|\b(?:general contractor|preconstruction|site inspection|specialty contractor|construction management).{0,3000}\bsubcontractor",
            r"\bpreconstruction\b",
            r"\bgeneral contractor\b",
            r"\bMEP\b",
            r"nuclear facility|nuclear mega.?project",
            r"parliamentary precinct",
            r"real property programs",
            r"\bProcore\b",
            r"tenant improvement",
            r"building restoration",
            r"construction administration",
            r"\bshop drawings\b",
            r"mechanical construction",
            r"plumbing and mechanical",
            r"water purif|waste\s?water treatment",
            r"multifamily property",
            r"property management company",
            r"resource.sector client",
            r"pharmaceutical advertising",
            r"healthcare communications agency",
        ],
    }
}

DEFAULT_ARCHETYPE = "software_tpm"

LINKEDIN_SEARCH_QUERIES = ARCHETYPE_CONFIGS[DEFAULT_ARCHETYPE]["search_queries"]

# Block entire company by name (exact substring match, case-insensitive).
COMPANY_BLOCKLIST = ARCHETYPE_CONFIGS[DEFAULT_ARCHETYPE]["company_blocklist"]

# Block on job title. Matched before reading description (cheaper).
TITLE_BLOCKLIST = ARCHETYPE_CONFIGS[DEFAULT_ARCHETYPE]["title_blocklist"]

# Entry-level / below-PM titles. Sets is_entry_level_filtered=True (and is_filtered=True).
# UI exposes these separately — pivot from IT PM to coordinator is feasible; to Sr construction PM is not.
TITLE_ENTRY_LEVEL_BLOCKLIST = ARCHETYPE_CONFIGS[DEFAULT_ARCHETYPE]["title_entry_level_blocklist"]

# Block on full job description. Catches generic titles ("Senior Project Manager") at non-IT firms.
DESC_BLOCKLIST = ARCHETYPE_CONFIGS[DEFAULT_ARCHETYPE]["desc_blocklist"]

# --- Processing Limits ---
SCRAPING_SOURCES = ["linkedin"] # "linkedin", "careers_future"
JOBS_TO_SCORE_PER_RUN = int(os.environ.get("JOBS_TO_SCORE_PER_RUN", "25"))
JOBS_TO_CUSTOMIZE_PER_RUN = 1
MAX_JOBS_PER_SEARCH = {
    "linkedin": None,
    "careers_future": 10,
}

# =================================================================
# 3. ADVANCED SYSTEM SETTINGS (Modify with Caution)
# =================================================================
LLM_MAX_RPM = 10
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 10
LLM_DAILY_REQUEST_BUDGET = 0
LLM_REQUEST_DELAY_SECONDS = 8

JOB_SCORING_MODEL_CHAIN = [
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemma-4-31b-it",
    "gemini/gemini-3-flash-preview",
    "gemini/gemma-4-26b-a4b-it",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
]

JOB_INSIGHTS_MODEL_CHAIN = [
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemma-4-26b-a4b-it",
    "gemini/gemma-4-31b-it",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-3-flash-preview",
]

FREEHIRE_CATEGORIES = frozenset({
    "software_engineering", "backend", "frontend", "fullstack", "mobile",
    "devops", "sre", "network_engineering", "data_engineering", "data_science",
    "data_analytics", "ml_ai", "ai_engineering", "qa", "security", "hardware",
    "embedded", "blockchain", "architecture", "design", "engineering_design",
    "product", "project_management", "management", "marketing", "sales",
    "support", "business_analysis", "solutions_engineering", "developer_relations",
    "technical_writing", "recruiting", "hr", "finance", "legal", "operations",
    "customer_success", "other",
})
FREEHIRE_SENIORITY_LEVELS = frozenset({
    "", "intern", "junior", "middle", "senior", "lead", "staff", "principal", "c_level",
})
FREEHIRE_COMPAT_SCHEMA_VERSION = "freehire-compat-v1"
FREEHIRE_COMPAT_PROMPT_VERSION = "freehire-category-v1"
FREEHIRE_CLASSIFY_MODEL_CHAIN = JOB_INSIGHTS_MODEL_CHAIN
FREEHIRE_INPUT_TOKEN_BUDGET = max(
    1000, int(os.environ.get("FREEHIRE_INPUT_TOKEN_BUDGET", "32000"))
)
FREEHIRE_MAX_BATCH_JOBS = min(
    50, max(1, int(os.environ.get("FREEHIRE_MAX_BATCH_JOBS", "50")))
)
FREEHIRE_DESCRIPTION_MAX_CHARS = max(
    1000, int(os.environ.get("FREEHIRE_DESCRIPTION_MAX_CHARS", "12000"))
)
FREEHIRE_CHARS_PER_TOKEN = max(
    1.0, float(os.environ.get("FREEHIRE_CHARS_PER_TOKEN", "4"))
)
FREEHIRE_OUTPUT_TOKENS_PER_JOB = max(
    20, int(os.environ.get("FREEHIRE_OUTPUT_TOKENS_PER_JOB", "60"))
)
FREEHIRE_CLASSIFY_MAX_RETRIES = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_MAX_RETRIES", "3"))
)
FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS = max(
    0.0, float(os.environ.get("FREEHIRE_CLASSIFY_RETRY_BASE_SECONDS", "2"))
)
FREEHIRE_CLASSIFY_PAGE_SIZE = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_PAGE_SIZE", "500"))
)
FREEHIRE_CLASSIFY_LIMIT = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_LIMIT", "200"))
)
FREEHIRE_CLASSIFY_REQUEST_BUDGET = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_REQUEST_BUDGET", "40"))
)
FREEHIRE_CLASSIFY_MAX_DURABLE_ATTEMPTS = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_MAX_DURABLE_ATTEMPTS", "6"))
)
FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MINUTES = max(
    1, int(os.environ.get("FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MINUTES", "30"))
)
FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MAX_MINUTES = max(
    FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MINUTES,
    int(os.environ.get("FREEHIRE_CLASSIFY_RETRY_COOLDOWN_MAX_MINUTES", "1440")),
)

# --- Job Insights Analysis ---
# Keep scheduled analysis bounded. Recovery workflows can opt into repeated
# bounded passes without changing the hourly default.
JOB_INSIGHTS_MAX_JOBS = max(
    1, int(os.environ.get("JOB_INSIGHTS_MAX_JOBS", "100"))
)
JOB_INSIGHTS_BATCH_SIZE = 10
JOB_INSIGHTS_SLEEP_SECONDS = 6
JOB_INSIGHTS_MAX_RETRIES = 3
JOB_INSIGHTS_DB_PAGE_SIZE = 1000
JOB_INSIGHTS_UPSERT_BATCH_SIZE = 500

LINKEDIN_MAX_START = 30
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

JOB_EXPIRY_DAYS = 30
JOB_CHECK_DAYS = 3
JOB_DELETION_DAYS = 60
JOB_CHECK_LIMIT = 50
ACTIVE_CHECK_TIMEOUT = 20
ACTIVE_CHECK_MAX_RETRIES = 2
ACTIVE_CHECK_RETRY_DELAY = 10

ENABLE_REPOST_DEDUP = True
DESCRIPTION_FINGERPRINT_MIN_LENGTH = 500
REPOST_DESCRIPTION_SIMILARITY_THRESHOLD = 0.90
REPOST_TITLE_SIMILARITY_THRESHOLD = 0.60
LINKEDIN_METADATA_ENRICH_LIMIT_PER_QUERY = 5
ENABLE_LINKEDIN_RELIST_TRACKING = os.environ.get("ENABLE_LINKEDIN_RELIST_TRACKING", "true").lower() == "true"
ENABLE_LINKEDIN_RELIST_EFFECTS = os.environ.get("ENABLE_LINKEDIN_RELIST_EFFECTS", "true").lower() == "true"
LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY = max(
    0, int(os.environ.get("LINKEDIN_RELIST_REFRESH_LIMIT_PER_QUERY", "3"))
)
LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN = max(
    0, int(os.environ.get("LINKEDIN_RELIST_REFRESH_LIMIT_PER_RUN", "20"))
)
LINKEDIN_RELIST_MIN_FORWARD_DAYS = max(
    2, int(os.environ.get("LINKEDIN_RELIST_MIN_FORWARD_DAYS", "2"))
)
LINKEDIN_RELIST_STABLE_OBSERVATIONS = max(
    2, int(os.environ.get("LINKEDIN_RELIST_STABLE_OBSERVATIONS", "2"))
)

TITLE_NORMALIZATION_REPLACEMENTS = {
    "sr.": "senior",
    "sr": "senior",
    "jr.": "junior",
    "jr": "junior",
    "&": "and",
}
