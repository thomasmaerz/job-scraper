from types import SimpleNamespace

import supabase_utils
import pytest


def test_get_filter_profile_returns_software_tpm_profile_for_current_jobs():
    profile = supabase_utils.get_filter_profile("software_tpm")

    assert profile["filter_profile"] == "software_tpm_v1"
    assert r"construction firm" in profile["desc_blocklist"]
    assert r"aerospace.*defense|defense.*aerospace" not in profile["desc_blocklist"]


def test_get_filter_profile_fails_loudly_for_unknown_archetype():
    with pytest.raises(ValueError, match="Unknown archetype/filter profile 'not_real'"):
        supabase_utils.get_filter_profile("not_real")


def test_match_filter_reason_uses_archetype_specific_construction_rules():
    reason, entry_level = supabase_utils.match_filter_reason(
        {
            "job_title": "Senior Project Manager",
            "company": "BuildCo",
            "description": "Experience with Procore, subcontractors, and construction administration.",
            "archetype": "software_tpm",
        }
    )

    assert reason == r"desc:\bProcore\b"
    assert entry_level is False


def test_lane_include_terms_are_or_routing_signals_and_excludes_win():
    profile = {
        "company_blocklist": [], "title_entry_level_blocklist": [],
        "title_blocklist": [r"sales"], "desc_blocklist": [r"construction"],
        "title_include": [r"program manager", r"project manager"],
        "description_include": [r"cloud", r"infrastructure"],
    }
    included = supabase_utils.evaluate_lane_filter(
        {"job_title": "Technical Program Manager", "description": "Delivery role"},
        archetype="technology_delivery", runtime_profile=profile,
    )
    review = supabase_utils.evaluate_lane_filter(
        {"job_title": "Operations Lead", "description": "General operations"},
        archetype="technology_delivery", runtime_profile=profile,
    )
    excluded = supabase_utils.evaluate_lane_filter(
        {"job_title": "Technical Program Manager", "description": "Construction delivery"},
        archetype="technology_delivery", runtime_profile=profile,
    )
    assert included["filter_status"] == "included"
    assert review == {"filter_status": "review", "is_filtered": False, "filter_reason": "include:no_route_signal", "is_entry_level_filtered": False}
    assert excluded["filter_status"] == "filtered"


@pytest.mark.parametrize(("title", "pattern"), [
    ("Process Project Manager", r"\bprocess project manager\b"),
    ("Backend Engineer", r"\bbackend (?:software )?engineer\b"),
    ("Network Software Engineer", r"\bnetwork software engineer\b"),
    ("Software Engineer", r"\bsoftware engineer\b"),
    ("Senior Software Developer, Brokerage", r"\bsoftware (?:developer|engineer)\b.*\bbrokerage\b"),
    ("Maintenance Electrician", r"\bmaintenance electrician\b"),
    ("Senior Data Scientist", r"\bdata scientist\b"),
    ("Robotics Engineer", r"\brobotics engineer\b"),
])
def test_reviewed_cross_lane_false_positive_titles_are_excluded(title, pattern):
    profile = {
        "company_blocklist": [],
        "title_entry_level_blocklist": [],
        "title_blocklist": [pattern],
        "desc_blocklist": [],
        "title_include": [],
        "description_include": [],
    }

    result = supabase_utils.evaluate_lane_filter(
        {"job_title": title, "description": ""},
        runtime_profile=profile,
    )

    assert result["filter_status"] == "filtered"


def test_persist_lane_filter_state_keys_update_by_job_and_lane(monkeypatch):
    calls = []
    class Query:
        def update(self, payload): calls.append(("update", payload)); return self
        def eq(self, key, value): calls.append(("eq", key, value)); return self
        def execute(self): return SimpleNamespace(data=[{}])
    class Db:
        def table(self, name): calls.append(("table", name)); return Query()
    supabase_utils.persist_lane_filter_state(
        "job-1", "data_pm", {"job_title": "Data Product Manager", "description": "SQL"},
        runtime_profile={"company_blocklist": [], "title_entry_level_blocklist": [], "title_blocklist": [], "desc_blocklist": []},
        db=Db(),
    )
    assert ("table", "job_archetype_memberships") in calls
    assert ("eq", "job_id", "job-1") in calls
    assert ("eq", "archetype", "data_pm") in calls


def test_filter_updates_for_same_job_do_not_cross_lane():
    states = {}
    class Query:
        def __init__(self): self.payload = None; self.job = None; self.lane = None
        def update(self, payload): self.payload = payload; return self
        def eq(self, key, value):
            if key == "job_id": self.job = value
            if key == "archetype": self.lane = value
            return self
        def execute(self): states[(self.job, self.lane)] = self.payload; return SimpleNamespace(data=[{}])
    class Db:
        def table(self, name): assert name == "job_archetype_memberships"; return Query()
    base = {"company_blocklist": [], "title_entry_level_blocklist": [], "title_blocklist": [], "desc_blocklist": []}
    supabase_utils.persist_lane_filter_state("job-1", "data_pm", {"job_title": "Data PM"}, runtime_profile=base, db=Db())
    excluded = {**base, "title_blocklist": ["network"]}
    supabase_utils.persist_lane_filter_state("job-1", "network_infrastructure", {"job_title": "Network PM"}, runtime_profile=excluded, db=Db())
    assert states[("job-1", "data_pm")]["is_filtered"] is False
    assert states[("job-1", "network_infrastructure")]["is_filtered"] is True


class FakeQuery:
    def __init__(self, state):
        self.state = state
        self.mode = None
        self.update_payload = None
        self.selected_fields = None
        self.filters = []
        self.order_field = None
        self.gt_filters = []

    def select(self, fields):
        self.mode = "select"
        self.selected_fields = fields
        self.state["select_calls"].append(fields)
        return self

    def update(self, payload):
        self.mode = "update"
        self.update_payload = payload
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        if self.mode == "update" and field == "job_id":
            self.state["updates"].append((value, self.update_payload))
        return self

    def is_(self, field, value):
        self.filters.append(("is", field, value))
        return self

    def range(self, _start, _end):
        return self

    def order(self, field, desc=False):
        self.order_field = (field, desc)
        return self

    def gt(self, field, value):
        self.gt_filters.append((field, value))
        return self

    def execute(self):
        if self.mode == "select":
            if (
                self.state.get("fail_select_with_archetype")
                and "archetype" in (self.selected_fields or "")
            ):
                raise Exception('column "archetype" does not exist')
            if self.state["batches"]:
                return SimpleNamespace(data=self.state["batches"].pop(0))
            return SimpleNamespace(data=[])
        self.state["update_calls"].append(
            {
                "payload": self.update_payload,
                "filters": list(self.filters),
            }
        )
        return SimpleNamespace(data=self.state.get("update_response_data"))


class FakeSupabase:
    def __init__(self, state):
        self.state = state

    def table(self, _name):
        return FakeQuery(self.state)


def make_state(*batches, fail_select_with_archetype=False):
    return {
        "batches": list(batches),
        "updates": [],
        "update_calls": [],
        "update_response_data": [],
        "select_calls": [],
        "fail_select_with_archetype": fail_select_with_archetype,
    }


def test_flag_filtered_jobs_falls_back_when_archetype_column_missing(monkeypatch):
    state = make_state(
        [
            {
                "job_id": "job-1",
                "job_title": "Senior Project Manager",
                "company": "BuildCo",
                "description": "Experience with Procore, subcontractors, and construction administration.",
            }
        ],
        fail_select_with_archetype=True,
    )

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase(state))

    flagged_count = supabase_utils.flag_filtered_jobs()

    assert flagged_count == 1
    assert state["select_calls"][:2] == [
        "job_id, job_title, company, description, archetype",
        "job_id, job_title, company, description",
    ]
    assert state["updates"] == [
        (
            "job-1",
            {
                "is_filtered": True,
                "filter_reason": r"desc:\bProcore\b",
                "is_entry_level_filtered": False,
            },
        )
    ]


def test_flag_filtered_jobs_uses_row_archetype_when_present(monkeypatch):
    monkeypatch.setitem(
        supabase_utils.config.ARCHETYPE_CONFIGS,
        "non_default_pm",
        {
            "filter_profile": "non_default_pm_v1",
            "company_blocklist": [],
            "title_entry_level_blocklist": [],
            "title_blocklist": [],
            "desc_blocklist": [r"aerospace.*defense|defense.*aerospace"],
        },
    )
    state = make_state(
        [
            {
                "job_id": "job-2",
                "job_title": "Senior Program Manager",
                "company": "AeroBuild",
                "description": "Program leadership across aerospace and defense programs.",
                "archetype": "non_default_pm",
            }
        ]
    )

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase(state))

    flagged_count = supabase_utils.flag_filtered_jobs()

    assert flagged_count == 1
    assert state["select_calls"][0] == "job_id, job_title, company, description, archetype"
    assert state["updates"] == [
        (
            "job-2",
            {
                "is_filtered": True,
                "filter_reason": r"desc:aerospace.*defense|defense.*aerospace",
                "is_entry_level_filtered": False,
            },
        )
    ]


def test_flag_filtered_jobs_does_not_skip_rows_when_batches_shrink_after_updates(monkeypatch):
    rows = [
        {
            "job_id": f"job-{index}",
            "job_title": "Senior Project Manager",
            "company": "BuildCo",
            "description": "Experience with Procore, subcontractors, and construction administration.",
            "archetype": "software_tpm",
            "is_filtered": False,
        }
        for index in range(1001)
    ]

    class ShrinkingResultSetQuery:
        def __init__(self, state):
            self.state = state
            self.mode = None
            self.selected_fields = None
            self.update_payload = None
            self.range_start = 0
            self.range_end = None
            self.filters = []
            self.gt_filters = []
            self.order_field = None

        def select(self, fields):
            self.mode = "select"
            self.selected_fields = fields
            return self

        def update(self, payload):
            self.mode = "update"
            self.update_payload = payload
            return self

        def eq(self, field, value):
            self.filters.append((field, value))
            if self.mode == "update" and field == "job_id":
                self.state["updates"].append(value)
                for row in self.state["rows"]:
                    if row["job_id"] == value:
                        row.update(self.update_payload)
                        break
            return self

        def range(self, start, end):
            self.range_start = start
            self.range_end = end
            self.state["ranges"].append((start, end))
            return self

        def order(self, field, desc=False):
            self.order_field = (field, desc)
            return self

        def gt(self, field, value):
            self.gt_filters.append((field, value))
            return self

        def execute(self):
            if self.mode == "select":
                filtered_source_rows = [row for row in self.state["rows"] if row["is_filtered"] is False]
                for field, value in self.gt_filters:
                    filtered_source_rows = [row for row in filtered_source_rows if row[field] > value]
                if self.order_field is not None:
                    field, desc = self.order_field
                    filtered_source_rows = sorted(filtered_source_rows, key=lambda row: row[field], reverse=desc)
                filtered_rows = [
                    {
                        key: row.get(key)
                        for key in [part.strip() for part in self.selected_fields.split(",")]
                    }
                    for row in filtered_source_rows
                ]
                end = None if self.range_end is None else self.range_end + 1
                return SimpleNamespace(data=filtered_rows[self.range_start:end])
            return SimpleNamespace(data=[])

    class ShrinkingResultSetSupabase:
        def __init__(self, state):
            self.state = state

        def table(self, _name):
            return ShrinkingResultSetQuery(self.state)

    state = {"rows": rows, "updates": [], "ranges": []}
    monkeypatch.setattr(supabase_utils, "supabase", ShrinkingResultSetSupabase(state))

    flagged_count = supabase_utils.flag_filtered_jobs()

    assert flagged_count == 1001
    assert len(state["updates"]) == 1001
    assert state["ranges"][:2] == [(0, 999), (0, 999)]
    assert all(row["is_filtered"] is True for row in state["rows"])


def test_backfill_job_archetypes_updates_missing_archetype_rows(monkeypatch):
    state = make_state()
    state["update_response_data"] = [{"job_id": "job-1"}, {"job_id": "job-2"}]

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase(state))

    updated_count = supabase_utils.backfill_job_archetypes()

    assert updated_count == 2
    assert state["update_calls"] == [
        {
            "payload": {
                "archetype": "software_tpm",
                "filter_profile": "software_tpm_v1",
            },
            "filters": [
                ("eq", "provider", "linkedin"),
                ("is", "archetype", None),
            ],
        }
    ]


def test_clear_removed_aerospace_defense_filter_resets_removed_reason(monkeypatch):
    state = make_state()
    state["update_response_data"] = [{"job_id": "job-3"}]

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase(state))

    cleared_count = supabase_utils.clear_removed_aerospace_defense_filter()

    assert cleared_count == 1
    assert state["update_calls"] == [
        {
            "payload": {
                "is_filtered": False,
                "filter_reason": None,
                "is_entry_level_filtered": False,
            },
            "filters": [
                ("eq", "filter_reason", r"desc:aerospace.*defense|defense.*aerospace"),
                ("eq", "archetype", "software_tpm"),
            ],
        }
    ]
