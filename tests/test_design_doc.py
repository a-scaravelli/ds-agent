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


def test_pyproject_runtime_dependencies_are_the_confirmed_libraries():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    names = sorted(d.split("==")[0] for d in project["dependencies"])
    assert names == ["kaggle", "numpy", "pandas", "scikit-learn"]
    declared = " ".join(
        d.lower()
        for deps in project.get("optional-dependencies", {}).values()
        for d in deps
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
    assert len(re.findall(r"`AWAITING_[A-Z_]+`", s2.split("Happy path")[0])) == 3
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


# --- Attempt registry with traceability (ticket 2 of section 15) ---


def test_code_fixes_happen_inside_the_attempt(sections):
    s2 = sections[2].lower()
    assert "is a new recorded attempt" not in s2 and "the fix is a new" not in s2
    assert "3 fix retries" in s2 and "configurable" in s2
    assert "`failure_kind = code_error`" in s2
    assert "do not consume one of the n iterations" in s2
    s7 = sections[7].lower()
    assert "on an execution record it only means that this run crashed" in s7
    assert "still failed after the cap of fix retries" in s7
    assert "refuses to close an attempt as `code_error` before the cap" in s7
    assert "open point" in s7 and "inconclusive" in s7 and "rejected proposals" in s7
    s11 = sections[11].lower()
    assert "closed as failed" in s11 and "fix retries per attempt" in s11


def test_registry_section_describes_failure_kind_and_executions(sections):
    s7 = sections[7]
    for kind in ("code_error", "hypothesis_falsified", "timeout", "below_threshold"):
        assert f"`{kind}`" in s7, kind
    low = s7.lower()
    assert "required when the outcome is not `succeeded` and absent when it is `succeeded`" in low
    assert "hash of normalized configuration, code, data and protocol" in low
    assert "execution record" in low and "linked to the attempt by its id" in low
    assert "outcome of the attempt is the outcome of its last execution" in low
    assert "list failures by `failure_kind`" in low


def test_storage_choice_documented_against_criteria(sections):
    s7 = sections[7]
    low = s7.lower()
    assert "none is decided now" not in low
    assert "sqlite" in low and "source of truth" in low and "`registry`" in low
    for alternative in ("mlflow", "jsonl", "both"):
        assert alternative in low, alternative
    for criterion in ("no network", "maintain", "reproducible"):
        assert criterion in low, criterion
    assert "mlflow" not in " ".join(
        __import__("tomllib").loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        .get("optional-dependencies", {}).get("dev", [])
    ).lower()
    s15 = sections[15].lower()
    assert "failure_kind" in s15 and "execution records" in s15
    assert "mlflow was not adopted" in sections[16].lower()
    assert "fix retries" in sections[16].lower()


def test_section_13_translation_fixed(sections):
    assert "a polished graphic design right away" in sections[13]
    assert "a polished chart right away" not in DESIGN.read_text(encoding="utf-8")


def test_library_decision_documented_in_design(sections):
    low = sections[5].lower()
    for word in ("pandas", "numpy", "scikit-learn", "pure python", "polars", "scipy", "section 1"):
        assert word in low, word
    assert "criteria" in low and "rejected alternatives" in low
    memory = ROOT / ".asl" / "process-memory.md"
    text = memory.read_text(encoding="utf-8")
    if "Implement ticket 3" not in text:
        pytest.skip("process memory is updated by the later documentation phase")
    entry = text.split("Implement ticket 3", 1)[1].lower()
    for word in ("pandas", "numpy", "scikit-learn", "polars", "scipy"):
        assert word in entry, word


def test_section_6_describes_one_holdout_session_per_cycle(sections):
    low = sections[6].lower()
    assert "one holdout session" in low
    assert "best attempt" in low and "variants requested by the user" in low
    assert "every read is logged" in low
    assert "no read is possible after the session is closed" in low
    assert "failed evaluation still consumes the session" in low
    assert "evaluated only once" not in low and "only one evaluation" not in low


def test_sections_5_and_6_document_deterministic_repetitions(sections):
    for number in (5, 6):
        low = sections[number].lower()
        assert "deterministic" in low and "only the seed" in low, number
    assert "noise of the model seed" in sections[5].lower()
    assert "noise of the model seed" in sections[6].lower()


def test_holdout_isolation_limit_documented():
    from kaggle_agent.evaluation import split

    design = DESIGN.read_text(encoding="utf-8").lower()
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    doc = split.__doc__.lower()
    for text in (design, readme, doc):
        assert "sandbox" in text and "ticket 6" in text
        assert "interface" in text and "log" in text


def test_section_10_credentials_token_and_lookup_order(sections):
    low = sections[10].lower()
    for word in ("generate new token", "account page", "access_token", "kaggle_api_token",
                 "kaggle.json", "600", "kaggle_username", "legacy", "lookup order"):
        assert word in low, word
    start = low.index("the lookup order")
    order = [low.index(w, start) for w in ("`kaggle_api_token`", "`~/.kaggle/access_token`",
                                           "`kaggle_username`", "`~/.kaggle/kaggle.json`")]
    assert order == sorted(order)
    assert "looser than `600`" in low and "raises an error" in low
    for word in ("model context", "logs", "registry", "sandbox", "only the kaggle client reads"):
        assert word in low, word
    assert "environment" in low and "never modified" in low


def test_section_10_daily_limit_score_and_key_scan_decisions(sections):
    low = sections[10].lower()
    assert "reported by kaggle" in low and "required argument, with no default" in low
    assert "submissionentry" in low and "recorded in the registry" in low
    assert "read and recorded together with the submission" in low
    assert "key-like strings" in low and "tickets that save artifacts" in low
    assert "decisions of the user" in low and "nothing in the code base calls `submit`" in low
    assert "explicit action of the user" in low


def test_kaggle_access_method_documented_against_criteria(sections):
    low = sections[10].lower()
    for word in ("kagglehub", "direct http", "official `kaggle` package", "section 1",
                 "pinned", "import kaggle", "authenticate"):
        assert word in low, word
    memory = ROOT / ".asl" / "process-memory.md"
    text = memory.read_text(encoding="utf-8")
    if "Implement ticket 4" not in text:
        pytest.skip("process memory is updated by the later documentation phase")
    entry = text.split("Implement ticket 4", 1)[1].lower()
    for word in ("kagglehub", "direct http", "kaggle_api_token", "access_token", "q1", "q2", "q3", "q4", "q5"):
        assert word in entry, word


def test_gitignore_lists_credentials_files():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for name in ("access_token", "kaggle.json", ".env", ".env.*"):
        assert name in lines, name


def test_design_mentions_kaggle_package_in_stack_and_plan(sections):
    assert "official `kaggle` package" in sections[1] or "kaggle" in sections[1].lower()
    assert "SubmissionEntry" in sections[7]
    assert "manual smoke test" in sections[15].lower()


# --- Headless Claude adapter (ticket 5 of section 15) ---


def test_section_8_documents_cli_version_flags_and_what_is_not_verified(sections):
    low = sections[8].lower()
    for word in ("version", "2.1.296", "--output-format json", "--model", "--tools", "--allowedtools",
                 "--permission-mode", "--resume", "stdin", "usage", "exit code", "rate limit",
                 "not verified", "envelope"):
        assert word in low, word


def test_section_8_documents_subscription_only_permissions_retries_and_model(sections):
    low = sections[8].lower()
    for word in ("anthropic_api_key", "strips", "warning", "never its value", "claude_code_oauth_token",
                 "toolpermissions", "dontask", "backoff", "invalidresponseerror", "stateless",
                 "quotaexhaustederror", "no model name is written", "enforces no budget"):
        assert word in low, word


def test_adapter_is_aligned_across_sections_7_11_14_15_16(sections):
    assert "llm_calls" in sections[7] and "model" in sections[7].lower()
    assert "reported by the adapter" in sections[11].lower() and "registry" in sections[11].lower()
    assert "`llm`" in sections[14] and "recording" in sections[14].lower()
    assert "headless claude adapter" in sections[15].lower() and "done:" in sections[15].lower()
    assert "not verified" in sections[16].lower() and "2.1.296" in sections[16]


def test_section_8_documents_rich_contract_and_rejection_reason(sections):
    low = sections[8].lower()
    for word in ("rich contract", "contract_hints", "any json value", "rejection reason", "rejection_reason",
                 "claude-sonnet-4-6", "not verified", "rate limit", "oauth"):
        assert word in low, word
    assert "rejection_reason" in sections[7] and "rejection_reason" not in sections[1]
