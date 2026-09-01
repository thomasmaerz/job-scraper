"""Dependency-light canonical storage keys for customized resume versions."""

from lane_catalog import canonical_lane_slug


def customized_resume_storage_path(
    archetype: str, job_id: str, customized_resume_id: str
) -> str:
    """Build the canonical, lane-isolated path for one immutable resume version."""
    lane = canonical_lane_slug(archetype)
    return f"{lane}/{job_id}/{customized_resume_id}.pdf"
