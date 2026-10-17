"""Check structure and content of docs/design.md (no network)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "docs" / "design.md"

SECTION_TITLES = {
    1: "Goals and non-goals",
    2: "Work cycle",
    3: "Work phases",
    4: "Proposal format",
    5: "Choice of the metric and the evaluation protocol",
    6: "Real improvement, noise and overfitting",
    7: "Experiment registry",
    8: "Roles: model and deterministic code",
    9: "Error analysis",
    10: "Kaggle integration",
    11: "Stop criteria and budget",
    12: "Reproducibility",
    13: "Human interaction",
    14: "Architecture and repository structure",
    15: "Implementation plan",
    16: "Open risks and questions for the human",
}
DECISION_SECTIONS = [2, 5, 6, 7, 8, 10, 11, 14]
MIN_WORDS = 80


def parse_sections(text: str) -> list[tuple[int, str, str]]:
    """Return (number, title, body) for each `## N. Title` header."""
    parts = re.split(r"^## (\d+)\. (.+)$", text, flags=re.M)
    return [
        (int(parts[i]), parts[i + 1].strip(), parts[i + 2])
        for i in range(1, len(parts), 3)
    ]


@pytest.fixture(scope="module")
def sections() -> dict[int, str]:
    assert DESIGN.is_file()
    return {n: body for n, _, body in parse_sections(DESIGN.read_text(encoding="utf-8"))}


def test_design_has_all_16_sections_in_order():
    parsed = parse_sections(DESIGN.read_text(encoding="utf-8"))
    assert [n for n, _, _ in parsed] == list(range(1, 17))
    assert {n: t for n, t, _ in parsed} == SECTION_TITLES


def test_sections_have_concrete_content():
    for n, _, body in parse_sections(DESIGN.read_text(encoding="utf-8")):
        assert len(body.split()) >= MIN_WORDS, f"section {n} too short"
        assert not re.search(r"\b(TODO|TBD|FIXME|lorem ipsum)\b", body, re.I), n


def test_decision_sections_have_rejected_alternatives(sections):
    for n in DECISION_SECTIONS:
        low = sections[n].lower()
        assert "alternative" in low and "rejected" in low, f"section {n}"


PRESCRIPTIVE = [
    r"the metric (is|will be) (always )?[A-Z]{2,}",
    r"always uses",
    r"\b\d+-?fold\b",
    r"\balways\s+\d+\b",
    r"uses (always )?(lightgbm|xgboost|catboost)",
]


def test_design_prescribes_no_fixed_metrics_or_techniques(sections):
    text = sections[5] + sections[6]
    for pattern in PRESCRIPTIVE:
        assert not re.search(pattern, text, re.I if "lightgbm" in pattern else 0), pattern
    low = text.lower()
    assert "justif" in low and "derivation" in low
    assert "does not fix" in sections[5].lower()


def test_metric_and_protocol_are_frozen(sections):
    low = sections[5].lower()
    assert "frozen" in low and "hash" in low
    assert "explicit proposal" in low and "approved" in low


def test_stack_is_non_binding_starting_point(sections):
    for n in (1, 15):
        low = sections[n].lower()
        assert "starting point" in low, n
        assert "not binding" in low or "not a constraint" in low, n
    low1 = sections[1].lower()
    for criterion in ("without gpu", "paid", "without network", "maintenance"):
        assert criterion in low1, criterion
    assert "alternative" in sections[15].lower()


def test_pyproject_no_runtime_dependencies_or_modeling_libs():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project.get("dependencies", []) == []
    declared = " ".join(
        d.lower() for deps in project.get("optional-dependencies", {}).values() for d in deps
    )
    for lib in ("pandas", "scikit-learn", "lightgbm", "mlflow", "numpy"):
        assert lib not in declared


def test_subpackages_cited_in_section_14(sections):
    pkg = ROOT / "src" / "kaggle_agent"
    subs = [p.name for p in pkg.iterdir() if (p / "__init__.py").is_file()]
    assert subs
    for name in subs:
        assert f"`{name}`" in sections[14], name


def test_readme_states_purpose_and_links_design():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/design.md" in text
    assert "Kaggle" in text and "data scientist" in text


def test_implementation_plan_tickets_have_criteria_and_fakes(sections):
    body = sections[15]
    tickets = re.findall(r"^\d+\. \*\*.+$", body, flags=re.M)
    assert len(tickets) >= 10
    for t in tickets:
        assert "Goal:" in t and "Acceptance criteria:" in t, t[:50]
    low = body.lower()
    assert "without network" in low
    assert "fake" in low and "kaggle" in low and "claude" in low
    assert "simulation" in low


def test_credentials_and_evaluation_protection(sections):
    s10 = sections[10].lower()
    assert "credentials" in s10 and "kaggle.json" in s10 and "600" in s10
    assert ".gitignore" in s10 and "model context" in s10
    s8 = sections[8].lower()
    assert "read-only" in s8 and "hash" in s8 and "holdout" in s8 and "sandbox" in s8


def test_predict_optimize_as_extension_point(sections):
    low = sections[14].lower()
    assert "predict + optimize" in low
    assert "uplift" in low and "allocation" in low
    assert "extension point" in low
    assert "is not designed" in low


# --- Short plan, autonomous cycle, traceability (ticket #7) ---


def _all(sections, *nums):
    return " ".join(sections[n] for n in nums).lower()


def test_short_initial_plan_with_five_items(sections):
    low = _all(sections, 2, 3, 4)
    assert "one page" in low and "2 minutes" in low
    for item in ("metric and why", "validation scheme", "feature engineering", "models to try", "baseline"):
        assert item in sections[4].lower(), item


def test_plan_is_single_control_point(sections):
    s2 = sections[2]
    assert len(re.findall(r"`AWAITING_[A-Z_]+`", s2.split("Happy path")[0])) == 2
    assert "`AWAITING_PLAN_APPROVAL`" in s2 and "`AWAITING_APPROVAL`" not in s2
    assert "single point of control" in s2.lower()
    s5 = sections[5].lower()
    assert "contained in the initial plan" in s5 and "replaces the separate approval" in s5
    assert "does not validate the individual steps" in sections[8].lower()


def test_split_validation_and_holdout(sections):
    low = _all(sections, 5, 6, 8)
    assert "validation set" in low and "drives the iterations" in low
    assert "never used to decide" in low or "is not used to decide" in low
    assert "every read is logged" in low
    assert "at the end of the cycle" in low


def test_baseline_precedes_loop(sections):
    s2 = sections[2]
    assert "`BASELINE`" in s2
    assert s2.index("`SPLITTING`") < s2.index("`BASELINE`") < s2.index("`ITERATING`")
    assert "first model trained and evaluated" in s2.lower()


def test_loop_ten_configurable_iterations(sections):
    low = _all(sections, 2, 11)
    assert "10 by default" in low or "default 10" in low
    assert "configurable" in low
    for item in ("metric", "errors", "hypothesis", "feature engineering", "model", "hyperparameters", "data cleaning", "new attempt"):
        assert item in sections[2].lower(), item


def test_early_stop_after_k_iterations(sections):
    low = sections[11].lower()
    assert "k consecutive iterations" in low and "configurable" in low
    assert "automatically" in low and "final report" in low


def test_short_final_report(sections):
    low = sections[13].lower()
    assert "short final report" in low
    for item in ("best attempt", "test holdout", "what worked and what did not"):
        assert item in low, item


def test_review_with_suggestions_and_new_cycle(sections):
    low = _all(sections, 2, 7, 13)
    assert "accepts" in low and "suggestions" in low
    assert "new cycle" in low and "existing state" in low
    assert "best model and the history" in sections[7].lower()
    assert "cycles spawned from it" in sections[7].lower()


def test_deliverable_end_to_end_pipeline(sections):
    low = sections[13].lower()
    assert "end-to-end python pipeline" in low
    for item in ("data loading", "preprocessing", "feature engineering", "training", "evaluation", "prediction", "submission"):
        assert item in low, item
    assert "n pipelines" in low and "simpler alternative" in low


def test_pipeline_single_command_and_reproducible(sections):
    for n in (12, 13):
        low = sections[n].lower()
        assert "single command" in low, n
        assert "fixed seeds" in low and "declared dependencies" in low, n
        assert "notebook" in low and "in-memory state" in low, n


def test_pipeline_linked_to_attempt(sections):
    assert "pipeline reference" in sections[7].lower()
    low = sections[13].lower()
    assert "from the attempt to the corresponding pipeline" in low and "link to the pipeline" in low


def test_attempt_traceability_fields(sections):
    low = sections[7].lower()
    for field in ("hypothesis", "rationale", "parent_id", "what_changed", "configuration", "metrics", "duration", "outcome", "artifacts"):
        assert field in low, field
    for outcome in ("succeeded", "failed", "inconclusive"):
        assert outcome in low
    assert "`what_changed`" in sections[4]


def test_web_list_and_chart(sections):
    low = sections[13].lower()
    assert "list/timeline of attempts" in low
    assert "chart of the metric per iteration" in low and "highlighted" in low


def test_web_detail_and_suggestions_without_reading_code(sections):
    low = sections[13].lower()
    assert "rationale in natural language" in low
    assert "user suggestions" in low and "cycles spawned from each" in low
    assert "without reading code" in low


def test_graphic_design_deferred(sections):
    assert "deferred to a later ticket" in sections[13].lower()
    s15 = sections[15]
    tickets = " ".join(re.findall(r"^\d+\. \*\*.+$", s15, flags=re.M)).lower()
    assert "graphic design" not in tickets
    assert "outside this plan" in s15.lower()


def test_ticket_sequence_consistency_and_no_per_step_approval(sections):
    s15 = sections[15].lower()
    for word in ("autonomous cycle", "pipeline", "traceability", "web interface"):
        assert word in s15, word
    text = DESIGN.read_text(encoding="utf-8").lower()
    for forbidden in ("approval at every step", "awaiting_approval", "`proposing`", "the human sees the diff"):
        assert forbidden not in text, forbidden
    assert "actual submission to kaggle is not part of the autonomous cycle" in sections[10].lower()


def test_only_documentation_no_new_packages():
    pkg = ROOT / "src" / "kaggle_agent"
    subs = {p.name for p in pkg.iterdir() if (p / "__init__.py").is_file()}
    assert "pipeline" not in subs and "web" not in subs
