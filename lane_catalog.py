"""Canonical career-lane metadata derived from resgen/candidate/lanes.md.

Search enablement and scrape limits are deployment configuration.  The lane
meaning and filtering vocabulary are code-level contracts so a database typo
cannot silently change what a lane means.
"""

from __future__ import annotations

from copy import deepcopy


CANONICAL_LANE_SLUGS = (
    "technology_delivery",
    "systems_platform_ops",
    "network_infrastructure",
    "datacenter_operations",
    "ai_workflow_automation",
    "building_controls",
)
LANE_ALIASES = {"software_tpm": "technology_delivery"}


LANE_CATALOG: dict[str, dict] = {
    "technology_delivery": {
        "definition": "Technology project, program, delivery, and implementation management.",
        "route_when": "Schedule, budget, risk, vendors, governance, stakeholders, or cross-functional delivery is primary.",
        "precision_queries": (
            "Technical Project Manager", "Technical Program Manager", "IT Project Manager",
            "Information Technology Project Manager", "Technology Delivery Manager",
            "gestionnaire de projet", "gestionnaire de programme", "chef de projet", "directeur de projet",
        ),
        "recall_query": '("project manager" OR "program manager" OR "project delivery") AND (SaaS OR ERP OR cloud OR infrastructure OR security OR AI OR data)',
        "positive_signals": (
            "implementation", "transformation", "roadmap", "governance", "stakeholder",
            "vendor", "risk", "budget", "enterprise systems",
        ),
        "exclude_signals": (
            "construction", "civil", "land development", "clinical", "marketing operations",
            "sales operations", "product-only", "Scrum-only", "hands-on ERP configuration",
        ),
    },
    "systems_platform_ops": {
        "definition": "Systems, compute, virtualization, identity, storage, OS, and container-platform operations.",
        "route_when": "Operating compute infrastructure is primary.",
        "precision_queries": (
            "Systems Administrator", "System Administrator", "Infrastructure Engineer",
            "VMware Administrator", "Virtualization Engineer", "Platform Operations Engineer",
        ),
        "recall_query": "(VMware OR vSphere OR ESXi OR virtualization OR Kubernetes OR OpenShift) AND (infrastructure OR operations OR administrator)",
        "positive_signals": (
            "VMware", "vSphere", "ESXi", "Active Directory", "Linux", "Windows Server",
            "storage", "backup", "Kubernetes", "OpenShift",
        ),
        "exclude_signals": (
            "backend", "full-stack", "data platform", "ML platform", "product management",
            "sales engineering", "generic DevOps",
        ),
    },
    "network_infrastructure": {
        "definition": "Routing, switching, wireless, WAN/VPN, firewall, network operations, and NOC work.",
        "route_when": "Connectivity, topology, or network availability is primary.",
        "precision_queries": (
            "Network Engineer", "Network Administrator", "Network Operations Engineer",
            "Network Infrastructure Engineer", "NOC Engineer",
        ),
        "recall_query": "(Cisco OR Aruba OR Juniper OR Meraki) AND (routing OR switching OR wireless OR firewall OR SD-WAN)",
        "positive_signals": (
            "routing", "switching", "VLAN", "WAN", "VPN", "firewall", "wireless", "Cisco", "Aruba", "NOC",
        ),
        "exclude_signals": (
            "neural network", "social network", "data annotation", "software development",
            "data science", "telecom sales", "office administration",
        ),
    },
    "datacenter_operations": {
        "definition": "Rack/stack, structured cabling, server hardware, power/cooling, break/fix, and critical-facility operations.",
        "route_when": "Physical datacenter infrastructure is primary.",
        "precision_queries": (
            "Data Center Technician", "Data Centre Technician", "Datacenter Technician",
            "Critical Facilities Technician", "Critical Environment Technician",
        ),
        "recall_query": '("rack and stack" OR "structured cabling" OR "server hardware") AND (datacenter OR "data center" OR "data centre")',
        "positive_signals": (
            "rack and stack", "cabling", "break/fix", "PDU", "UPS", "power", "cooling", "hardware", "critical facilities",
        ),
        "exclude_signals": (
            "data science", "data engineering", "sales", "solutions", "design-only engineering", "facility manager", "project manager",
        ),
    },
    "ai_workflow_automation": {
        "definition": "Implementation of AI-enabled workflows, agents, integrations, RAG, and low-code business automation.",
        "route_when": "Building and operating automation is primary.",
        "precision_queries": (
            "AI Automation Engineer", "AI Solutions Engineer", "Power Platform Developer",
            "Power Automate Developer", "Copilot Studio Developer", "Low Code Developer",
        ),
        "recall_query": '(LLM OR RAG OR Copilot OR "Power Platform" OR n8n) AND (workflow OR automation OR integration OR agent)',
        "positive_signals": (
            "workflow orchestration", "API integration", "agents", "RAG", "Copilot Studio",
            "Power Platform", "n8n", "evaluation", "human review",
        ),
        "exclude_signals": (
            "ML research", "model training", "data science", "GPU infrastructure", "deep production software engineering",
        ),
    },
    "building_controls": {
        "definition": "BAS/BMS, HVAC controls, DDC/PLC, commissioning, and controls service work.",
        "route_when": "Commissioning, programming, installing, or servicing building controls is primary.",
        "precision_queries": (
            "Building Automation Technician", "Building Controls Technician", "Controls Technician",
            "BAS Technician", "BMS Technician", "HVAC Controls Technician",
        ),
        "recall_query": '(DDC OR PLC OR BAS OR BMS OR "building automation") AND (HVAC OR commissioning OR controls)',
        "positive_signals": (
            "BAS", "BMS", "DDC", "HVAC controls", "commissioning", "PLC", "control panels", "field service",
        ),
        "exclude_signals": (
            "marketing automation", "test automation", "DevOps", "process automation",
            "AI automation", "robotics", "security technician",
        ),
    },
}


def canonical_lane_slug(slug: str) -> str:
    normalized = str(slug or "").strip().lower()
    return LANE_ALIASES.get(normalized, normalized)


def canonical_context(slug: str) -> dict:
    canonical = canonical_lane_slug(slug)
    if canonical not in LANE_CATALOG:
        expected = ", ".join(CANONICAL_LANE_SLUGS)
        raise ValueError(f"Unknown career lane '{slug}'. Expected one of: {expected}; alias: software_tpm.")
    return deepcopy(LANE_CATALOG[canonical])
