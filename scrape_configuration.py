"""Strict adapter for public.get_scraper_configuration()."""

from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from lane_catalog import CANONICAL_LANE_SLUGS, canonical_lane_slug


class ScrapeConfigurationError(RuntimeError):
    """The configured source did not satisfy the migration's JSON contract."""


class QueryKind(str, Enum):
    precision = "precision"
    recall = "recall"


class LocationRegion(str, Enum):
    canada = "canada"
    usa = "usa"
    eea = "eea"


class LaneSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype: StrictStr
    query: StrictStr = Field(min_length=1, max_length=2000)
    query_type: QueryKind
    language: StrictStr = Field(pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    sort_order: StrictInt = Field(ge=0)
    enabled: StrictBool

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @property
    def query_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.language}\0{self.query_type.value}\0{self.query}".encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.language}:{self.query_type.value}:{self.sort_order}:{digest}"


class CareerLane(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype: StrictStr
    display_name: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr
    routing_guidance: StrictStr
    title_include: list[StrictStr]
    title_exclude: list[StrictStr]
    description_include: list[StrictStr]
    description_exclude: list[StrictStr]
    enabled: StrictBool
    # Informational only. Missing profiles never disable lane scraping.
    resume_profile_ready: StrictBool = False
    sort_order: StrictInt = Field(ge=0)
    locations: list[LocationRegion]
    queries: list[LaneSearchQuery]

    @field_validator("archetype")
    @classmethod
    def validate_archetype(cls, value: str) -> str:
        if value not in CANONICAL_LANE_SLUGS:
            raise ValueError(f"expected one of {', '.join(CANONICAL_LANE_SLUGS)}")
        return value

    @field_validator(
        "title_include", "title_exclude", "description_include", "description_exclude"
    )
    @classmethod
    def validate_context(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("context entries must not be blank")
        return values

    @field_validator("locations")
    @classmethod
    def validate_locations(cls, values: list[LocationRegion]) -> list[LocationRegion]:
        if len(values) != len(set(values)):
            raise ValueError("locations must be unique")
        order = {region: index for index, region in enumerate(LocationRegion)}
        return sorted(values, key=order.__getitem__)

    @model_validator(mode="after")
    def validate_enabled_lane(self) -> "CareerLane":
        if not self.enabled:
            return self
        if not self.locations:
            raise ValueError(f"enabled lane '{self.archetype}' has no locations")
        enabled_queries = [query for query in self.queries if query.enabled]
        if not enabled_queries:
            raise ValueError(f"enabled lane '{self.archetype}' has no enabled queries")
        kinds = {query.query_type for query in enabled_queries}
        if QueryKind.precision not in kinds or QueryKind.recall not in kinds:
            raise ValueError(
                f"enabled lane '{self.archetype}' must have enabled precision and recall queries"
            )
        identities = [(query.query, query.language) for query in self.queries]
        if len(identities) != len(set(identities)):
            raise ValueError(f"lane '{self.archetype}' has duplicate query/language pairs")
        if any(query.archetype != self.archetype for query in self.queries):
            raise ValueError(f"lane '{self.archetype}' contains a query for another archetype")
        return self


class ScrapeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scraping_enabled: StrictBool
    lookback_days: StrictInt = Field(ge=1, le=365)
    max_jobs_per_query: StrictInt = Field(ge=1, le=10_000)
    max_pages_per_query: StrictInt = Field(ge=1, le=100)
    request_delay_ms: StrictInt = Field(ge=0, le=60_000)
    concurrent_queries: StrictInt = Field(ge=1, le=50)
    deduplicate_jobs: StrictBool
    fetch_descriptions: StrictBool
    score_jobs: StrictBool
    options: dict[str, Any]
    updated_at: StrictStr

    @model_validator(mode="after")
    def validate_linkedin_pacing_options(self) -> "ScrapeSettings":
        interval = self.options.get("global_request_interval_ms")
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or not 2_500 <= interval <= 60_000
        ):
            raise ValueError(
                "options.global_request_interval_ms must be an integer from 2500 to 60000"
            )
        jitter = self.options.get("request_jitter_ms")
        if jitter is not None and (
            not isinstance(jitter, int)
            or isinstance(jitter, bool)
            or not 0 <= jitter <= 10_000
        ):
            raise ValueError(
                "options.request_jitter_ms must be an integer from 0 to 10000"
            )
        bounded_integer_options = {
            "min_pages_per_query": (1, 100),
            "soft_max_pages_per_query": (1, 100),
            "hard_max_pages_per_query": (1, 100),
            "max_adaptive_extra_requests": (0, 10_000),
            "max_detail_tasks_per_run": (0, 10_000),
            "max_source_http_attempts_per_run": (1, 10_000),
            "minimum_recent_window_hours": (1, 8_760),
            "indexing_overlap_hours": (0, 8_760),
            "maximum_normal_window_hours": (1, 8_760),
            "outage_recovery_cap_hours": (1, 8_760),
        }
        for name, (minimum, maximum) in bounded_integer_options.items():
            value = self.options.get(name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"options.{name} must be an integer from {minimum} to {maximum}"
                )
        return self


class ScrapeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: StrictInt
    revision: StrictInt | None
    aliases: dict[StrictStr, StrictStr]
    settings: ScrapeSettings
    lanes: list[CareerLane]

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, aliases: dict[str, str]) -> dict[str, str]:
        for alias, archetype in aliases.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", alias):
                raise ValueError(f"invalid alias '{alias}'")
            if alias == archetype or archetype not in CANONICAL_LANE_SLUGS:
                raise ValueError(f"invalid alias mapping '{alias}' -> '{archetype}'")
        if aliases.get("software_tpm") != "technology_delivery":
            raise ValueError("aliases must contain software_tpm -> technology_delivery")
        return aliases

    @model_validator(mode="after")
    def validate_lanes(self) -> "ScrapeConfiguration":
        archetypes = [lane.archetype for lane in self.lanes]
        if len(archetypes) != len(set(archetypes)):
            raise ValueError("lane archetypes must be unique")
        if set(archetypes) != set(CANONICAL_LANE_SLUGS):
            raise ValueError(
                "lanes must contain exactly the six canonical archetypes: "
                + ", ".join(CANONICAL_LANE_SLUGS)
            )
        if self.settings.scraping_enabled and not any(lane.enabled for lane in self.lanes):
            raise ValueError("scraping is enabled but no lane is enabled")
        return self


class LinkedInGeography(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location_scope: LocationRegion
    geography_id: str
    location: str
    geo_id: int | None = None


class LinkedInSearchExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lane: CareerLane
    query: LaneSearchQuery
    geography: LinkedInGeography


_EEA_COUNTRIES = (
    ("AT", "Austria"), ("BE", "Belgium"), ("BG", "Bulgaria"), ("HR", "Croatia"),
    ("CY", "Cyprus"), ("CZ", "Czechia"), ("DK", "Denmark"), ("EE", "Estonia"),
    ("FI", "Finland"), ("FR", "France"), ("DE", "Germany"), ("GR", "Greece"),
    ("HU", "Hungary"), ("IS", "Iceland"), ("IE", "Ireland"), ("IT", "Italy"),
    ("LV", "Latvia"), ("LI", "Liechtenstein"), ("LT", "Lithuania"),
    ("LU", "Luxembourg"), ("MT", "Malta"), ("NL", "Netherlands"), ("NO", "Norway"),
    ("PL", "Poland"), ("PT", "Portugal"), ("RO", "Romania"), ("SK", "Slovakia"),
    ("SI", "Slovenia"), ("ES", "Spain"), ("SE", "Sweden"),
)


def expand_location_scopes(scopes: list[LocationRegion]) -> tuple[LinkedInGeography, ...]:
    expanded: list[LinkedInGeography] = []
    for scope in scopes:
        if scope is LocationRegion.canada:
            expanded.append(LinkedInGeography(
                location_scope=scope, geography_id="CA", location="Canada", geo_id=101174742
            ))
        elif scope is LocationRegion.usa:
            expanded.append(LinkedInGeography(
                location_scope=scope, geography_id="US", location="United States", geo_id=103644278
            ))
        else:
            expanded.extend(
                LinkedInGeography(location_scope=scope, geography_id=code, location=country)
                for code, country in _EEA_COUNTRIES
            )
    return tuple(expanded)


def build_search_executions(
    configuration: ScrapeConfiguration,
    archetype_override: str | None = None,
) -> tuple[LinkedInSearchExecution, ...]:
    requested_archetype = (
        canonical_lane_slug(archetype_override)
        if archetype_override and archetype_override.strip()
        else None
    )
    if requested_archetype and requested_archetype not in CANONICAL_LANE_SLUGS:
        expected = ", ".join(CANONICAL_LANE_SLUGS)
        raise ScrapeConfigurationError(
            f"Unknown SCRAPE_ARCHETYPE '{archetype_override}'. Expected one of: {expected}"
        )

    lanes = sorted(
        (
            lane
            for lane in configuration.lanes
            if lane.enabled
            and (requested_archetype is None or lane.archetype == requested_archetype)
        ),
        key=lambda lane: (lane.sort_order, lane.archetype),
    )
    if requested_archetype and not lanes:
        raise ScrapeConfigurationError(
            f"SCRAPE_ARCHETYPE '{requested_archetype}' is disabled in scrape configuration"
        )
    return tuple(
        LinkedInSearchExecution(lane=lane, query=query, geography=geography)
        for lane in lanes
        for query in sorted(
            (query for query in lane.queries if query.enabled),
            key=lambda query: (query.sort_order, query.query, query.language),
        )
        for geography in expand_location_scopes(lane.locations)
    )


def parse_scrape_configuration(raw: Any, source: str = "configuration") -> ScrapeConfiguration:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScrapeConfigurationError(f"{source} is not valid JSON: {exc}") from exc
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ScrapeConfigurationError(f"{source} must contain a JSON object")
    try:
        return ScrapeConfiguration.model_validate(raw)
    except ValidationError as exc:
        raise ScrapeConfigurationError(f"Invalid {source}: {exc}") from exc


def load_scrape_configuration(db: Any = None, environ: dict[str, str] | None = None) -> ScrapeConfiguration:
    """Load the exact migration contract from DB, or an explicit full-document override."""
    env = os.environ if environ is None else environ
    source = env.get("SCRAPE_CONFIG_SOURCE", "db").strip().lower()
    if source == "env":
        payload = env.get("SCRAPE_CONFIG_JSON")
        if not payload:
            raise ScrapeConfigurationError(
                "SCRAPE_CONFIG_SOURCE=env requires non-empty SCRAPE_CONFIG_JSON"
            )
        return parse_scrape_configuration(payload, "SCRAPE_CONFIG_JSON")
    if source == "file":
        configured_path = env.get("SCRAPE_CONFIG_FILE")
        if not configured_path:
            raise ScrapeConfigurationError("SCRAPE_CONFIG_SOURCE=file requires SCRAPE_CONFIG_FILE")
        path = Path(configured_path).expanduser()
        try:
            return parse_scrape_configuration(
                path.read_text(encoding="utf-8"), f"SCRAPE_CONFIG_FILE '{path}'"
            )
        except OSError as exc:
            raise ScrapeConfigurationError(f"Could not read SCRAPE_CONFIG_FILE '{path}': {exc}") from exc
    if source != "db":
        raise ScrapeConfigurationError(
            f"Unknown SCRAPE_CONFIG_SOURCE '{source}'. Expected db, env, or file."
        )
    if db is None:
        raise ScrapeConfigurationError("SCRAPE_CONFIG_SOURCE=db requires a Supabase client")
    try:
        response = db.rpc("get_scraper_configuration").execute()
    except Exception as exc:
        raise ScrapeConfigurationError(f"Supabase RPC get_scraper_configuration() failed: {exc}") from exc
    if response.data in (None, []):
        raise ScrapeConfigurationError("get_scraper_configuration() returned no configuration")
    return parse_scrape_configuration(response.data, "get_scraper_configuration()")
