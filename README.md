# kaggle-agent

An agent that does the iterative work of a data scientist on tabular Kaggle competitions, from downloading the data to the submission. The user approves only a short initial plan; then the agent works autonomously (validation/test holdout split, baseline, 10 improvement iterations by default, with early stop) and the user steps in at the end of the cycle: accept the report or give suggestions, which start a new cycle from the existing state. The model that reasons is headless Claude Code with the subscription (no paid API); it runs on a Mac without GPU.

Convention: the entire repository is in English (code, documentation, comments, tests, commit and PR messages).

## Current status

Design plus the first pieces of code: the design document is in [docs/design.md](docs/design.md), the `proposals` subpackage of `kaggle_agent` holds the data model of plan, attempt and suggestion (ticket 1 of section 15 of the design) and the `registry` subpackage holds the attempt registry with traceability (ticket 2). The other subpackages are still empty. There is no modeling logic yet.

## Stack

Python 3.12, pandas, scikit-learn, LightGBM and local MLflow are a starting point, not a constraint: libraries are chosen ticket by ticket, with the criteria of section 1 of the design. For this reason `pyproject.toml` declares no runtime dependencies, and the data model uses only the standard library.

## Data model (`kaggle_agent.proposals`)

All classes are immutable dataclasses. Invalid data raises `ValidationError` (a `ValueError` with `field` and `reason`), so the loop can pass the reason for a rejection back to the model.

- `ShortPlan`: the short initial plan with five mandatory items: `metric`, `validation`, `feature_engineering` (list), `models` (list) and `baseline`. Empty or wrongly typed items are rejected, and so are plans longer than `MAX_PLAN_WORDS` (300 words over the five items).
- `Attempt`: a falsifiable hypothesis. Mandatory and non-empty: `id`, `phase`, `hypothesis`, `rationale`, `what_changed`, `measure`. Optional: `parent_id` (different from `id`), `evidence`, `exploratory`, `expected_effect`, `estimated_cost`, `risks`, `artifacts`. An attempt without `evidence` must have `exploratory=True`, otherwise it is rejected.
- `Suggestion`: `id`, `text`, `origin_cycle` and `spawned_cycles` (no duplicates, never containing `origin_cycle`).

Each class has `to_dict`, `from_dict`, `to_json` and `from_json`. Rereading gives an identical object, the JSON is deterministic, and unknown keys or malformed JSON are rejected with `ValidationError`.

```python
from kaggle_agent.proposals import Attempt, ShortPlan, Suggestion, ValidationError

plan = ShortPlan(
    metric="official competition metric",
    validation="partitions that mimic the train/test separation",
    feature_engineering=["missing-value indicators"],
    models=["boosted trees"],
    baseline="simple model with default parameters",
)
assert ShortPlan.from_json(plan.to_json()) == plan

try:
    Attempt.from_dict({"id": "T-1", "phase": "iteration", "hypothesis": "h",
                       "rationale": "r", "measure": "m", "exploratory": True})
except ValidationError as error:
    print(error)  # what_changed: is required
```

## Registry (`kaggle_agent.registry`)

The registry stores cycles, attempts, execution records and suggestions behind the replaceable `Registry` interface. `SqliteRegistry` is the implementation and the source of truth: standard library `sqlite3` only, no network. MLflow and JSONL were evaluated and not adopted (see section 7 of the design).

- The `Attempt` is the proposal; an `ExecutionRecord` is what happened when it ran, linked by `attempt_id`: configuration, `seed`, code/data/protocol hashes, metrics per partition (`PartitionMetric` with mean and dispersion), duration, resources, outcome, `failure_kind`, reason and artifacts. An attempt has one or more execution records, and its outcome is that of the last one.
- `failure_kind` (`code_error`, `hypothesis_falsified`, `timeout`, `below_threshold`) is required unless the outcome is `succeeded`, and absent otherwise.
- Code fixes are not attempts: `add_execution` accepts a rerun only after a `failed` / `code_error` execution, up to 1 + `Cycle.max_fix_retries` executions (3 fix retries by default). `close_attempt` refuses to close a `code_error` attempt before that cap, so at attempt level `code_error` means "still failing after the cap".
- `compute_fingerprint(config, code_hash, data_hash, protocol_hash)` is a deterministic SHA-256 of the normalized configuration and the three hashes. `add_attempt` raises `DuplicateAttemptError` (with `existing_attempt_id`) if any earlier attempt, of any cycle and outcome, has the same fingerprint.
- Queries: `list_attempts`, `get_attempt`, `metric_series`, `list_failures(failure_kind=None)`, `list_suggestions` and `cycles_of_suggestion`. Only closed attempts appear in outcome and failure queries.

```python
from kaggle_agent.proposals import Attempt, ShortPlan
from kaggle_agent.registry import (
    Cycle, ExecutionRecord, PartitionMetric, SqliteRegistry, compute_fingerprint,
)

plan = ShortPlan(metric="m", validation="v", feature_engineering=["f"], models=["x"], baseline="b")
registry = SqliteRegistry("registry.db")  # ':memory:' by default
registry.add_cycle(Cycle(id="C1", plan=plan, iterations=10, stagnation_k=3, state="ITERATING"))
attempt = Attempt(id="A1", phase="iteration", hypothesis="h", rationale="r",
                  what_changed="added feature f", measure="m", exploratory=True)
config = {"model": "boosted trees"}
registry.add_attempt("C1", attempt, compute_fingerprint(config, "code-hash", "data-hash", "protocol-hash"))
registry.add_execution(ExecutionRecord(
    id="A1-1", attempt_id="A1", config=config, seed=1, code_hash="code-hash",
    data_hash="data-hash", protocol_hash="protocol-hash",
    metrics=[PartitionMetric("validation", 0.8, 0.02)], duration_seconds=3.0, outcome="succeeded",
))
registry.close_attempt("A1", "the feature helps")
print(registry.metric_series("C1", "validation"))
```

## Evaluation engine

Runtime stack: pandas, numpy and scikit-learn, pinned with `==` in `pyproject.toml` (choice and rejected alternatives in section 5 of the design). Only `evaluation` imports them.

The agent proposes a `Protocol` next to the plan. Metric, direction, partition schemes, seed, repetitions, `threshold_k` and holdout share are all mandatory; metrics and schemes are chosen by name from the registries `METRICS` and `SCHEMES` (extend with `register_metric` and `register_scheme`). `EvaluationEngine.freeze()` stores the protocol hash (protocol plus SHA-256 of the `evaluation` sources) on the cycle; every evaluation and holdout read recomputes it and raises `ProtocolHashMismatchError` on a mismatch.

```python
engine = EvaluationEngine(registry, "C1", protocol)  # protocol built explicitly
engine.freeze()  # at approval
kwargs = dict(config=config, code_hash="c", data_hash="d")
base = engine.evaluate_validation(baseline_candidate, frame, attempt_id="A0", execution_id="A0-1", **kwargs)
new = engine.evaluate_validation(candidate, frame, attempt_id="A1", execution_id="A1-1", **kwargs)
comparison = engine.compare("A1", "A0", new.scores, base.scores, base.scores)
with engine.holdout_session(frame) as session:  # one session per cycle
    session.evaluate_holdout(candidate, "A1")
```

A candidate implements `fit_predict(x_train, y_train, x_valid, seed)`. The engine does not persist executions: `result.execution` goes to `Registry.add_execution` (the loop ticket does that).

## Deliverable and traceability

The output of every cycle is an end-to-end Python pipeline that runs with a single command and is reproducible (or several pipelines, to compare variants), linked to the attempt it comes from. Every attempt records hypothesis and rationale, originating attempt and what changed, configuration, metrics, duration, outcome and artifacts; a web interface (list/timeline, metric chart, detail, suggestions and cycles) makes them consultable without reading code. The polished graphic design is deferred to a later ticket.

## Installation and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

The tests do not use the network.

## Structure

- `docs/design.md`: design in 16 sections (cycle, frozen evaluation, registry, Kaggle, ticket plan).
- `src/kaggle_agent/`: package with the subpackages `proposals` (data model), `registry` (attempt registry), `evaluation` (protocol, split, holdout, threshold), `loop`, `kaggle`, `llm`, `sandbox`, `review`, `budget`; `evaluation` holds the evaluation engine; `loop`, `kaggle`, `llm`, `sandbox`, `review` and `budget` are still empty.
- `tests/`: check the 16 sections of the design, the data model and its examples, the language convention, the test map, that the package imports, and the registry behavior (deduplication, fix retries, queries, persistence), the evaluation engine (protocol, schemes, split, holdout session, threshold, hash), and that third-party libraries are imported only by `evaluation`.

## Known limits

- There is still no runnable command of the agent: only the data model, the registry and the evaluation engine exist.
- The holdout guarantee is one of interface and log (one session per cycle, every read logged in the registry). Real isolation from generated code arrives with the sandbox (ticket 6).
- The hash covers the bytes of the `evaluation` sources: any edit of the package invalidates the frozen protocols of open cycles. Library versions are outside the hash and recorded in `resources`. Raising the threshold with the number of comparisons is not implemented; the count is only recorded.
- The registry is meant for a single process; multi-process access is not tested. The cycle `state` is free text until the `loop` ticket, and an iteration recorded `inconclusive` after repeated rejected proposals cannot be represented yet (open point in section 7 of the design). The semantic level of deduplication (section 7) is left to the loop ticket.
- The tests on the design check structure and keywords, not the quality of the content. The language test is a heuristic (Italian function words and identifiers), not a proof that every sentence is English.
- The word count of the plan (split on whitespace) is a simple proxy for "one page". `id` and `phase` have no format constraints yet: they belong to the registry and loop tickets.
- The test that forbids third-party dependencies will have to be relaxed when the development tickets arrive.
