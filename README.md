# kaggle-agent

An agent that does the iterative work of a data scientist on tabular Kaggle competitions, from downloading the data to the submission. The design (not yet implemented) has three levels with the same five role agents (coordinator, data, feature engineering, modeling, reviewer); level 1 is built first. The lead approves a short initial plan; then the agent works autonomously in rounds of several candidates (validation/test holdout split, baseline, rounds with early stop; provisional defaults: 5 rounds of 3 candidates) and the lead gives guidance after each round and steps in at the end of the cycle: accept the report or give suggestions, which start a new cycle from the existing state. The code of tickets 1 to 5 now has the types, tables and arguments that the three-level design needs (rounds, goal criteria, guidance, reports, bundles, role traces); the sandbox, the bundle runner, the role agents, the round pipeline and the web interface are tickets 6 onward and are not built. The model that reasons is headless Claude Code with the subscription (no paid API); it runs on a Mac without GPU.

Convention: the entire repository is in English (code, documentation, comments, tests, commit and PR messages).

## Current status

Design plus the first pieces of code: the design document is in [docs/design.md](docs/design.md), the `proposals` subpackage of `kaggle_agent` holds the data model of plan, attempt and suggestion (ticket 1 of section 15 of the design) the `registry` subpackage holds the attempt registry with traceability (ticket 2), the `evaluation` subpackage the frozen evaluation engine (ticket 3) the `kaggle` subpackage the Kaggle client with its fake (ticket 4) and the `llm` subpackage the headless Claude adapter with its fake (ticket 5). The other subpackages are still empty. There is no modeling logic yet.

## Stack

Python 3.12, pandas, scikit-learn, LightGBM and local MLflow are a starting point, not a constraint: libraries are chosen ticket by ticket, with the criteria of section 1 of the design. The runtime dependencies are pinned with `==` in `pyproject.toml`: pandas, numpy and scikit-learn (only `evaluation` imports them) and the official `kaggle` package (only `kaggle/real.py` imports it). The data model, the registry, the rest of the Kaggle client and the Claude adapter use only the standard library (the adapter spawns the `claude` CLI, which is not a Python dependency).

## Data model (`kaggle_agent.proposals`)

All classes are immutable dataclasses. Invalid data raises `ValidationError` (a `ValueError` with `field` and `reason`), so the loop can pass the reason for a rejection back to the model.

- `ShortPlan`: the short initial plan with five mandatory items: `metric`, `validation`, `feature_engineering` (list), `models` (list) and `baseline`. Empty or wrongly typed items are rejected, and so are plans longer than `MAX_PLAN_WORDS` (300 words over the five items).
- `Attempt`: a falsifiable hypothesis. Mandatory and non-empty: `id`, `phase`, `hypothesis`, `rationale`, `what_changed`, `measure`. Optional: `parent_id` (different from `id`), `evidence`, `exploratory`, `expected_effect`, `estimated_cost`, `risks`, `artifacts`. An attempt without `evidence` must have `exploratory=True`, otherwise it is rejected.
- `Suggestion`: `id`, `text`, `origin_cycle` and `spawned_cycles` (no duplicates, never containing `origin_cycle`).
- `GoalCriteria`: one or more `GoalCriterion` (`metric` name, finite `threshold`, `direction` `maximize` or `minimize`), with unique metrics and the order kept. It is the stop condition of the design; `proposals` does not know which metrics exist, `evaluation` checks that.
- `Guidance` (`id`, `round_id`, `text`, `created_at`), `Round` (`id`, `tasks` assigned by the coordinator, `candidates` as attempt ids without duplicates) and `ReviewerReport` (`id`, `round_id`, `text`, `created_at`). The report has no run ids: runs are reached through the round, its candidates and their executions.
- `ROLES` (`coordinator`, `data`, `feature_engineering`, `modeling`, `reviewer`) and `validate_role`.
- `Attempt` also takes an optional `bundle_id`; JSON written before it existed stays valid.

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
- Projects and rounds: `Cycle.project_id` (nullable), `add_round` / `get_round` / `list_rounds_with_candidates` (`RoundDetail`) and `add_attempt(..., round_id=None)`. The candidates of a round are the attempts stored with its `round_id` (candidate id = attempt id). `metric_series_by_round(cycle_id, partition)` returns, per round, the metric points of its candidates (`RoundMetrics`); it does not pick the best, because the registry does not know the direction of the metric.
- Bundles: `BundleRecord` (`id`, `cycle_id`, `folder_hash`, `reference`, `schema`) with `add_bundle`, `get_bundle` and `list_bundles`; an attempt with a `bundle_id` needs the bundle to exist.
- Reports and guidance: `add_reviewer_report` / `list_reviewer_reports(cycle_id, round_id=None)` and `add_guidance(cycle_id, guidance)` / `list_guidance`. `add_guidance` also creates the `Suggestion` with the same id and text and the cycle as origin, in the same transaction; a rejected guidance writes neither.
- Role traces: `RoleTrace` (`cycle_id`, `role`, `round_id`, `candidate_id` or None, `input_type` and `input`, `output_type` and `output`, `outcome`, `llm_call_id`, `created_at`) with `add_role_trace` and `list_role_traces(cycle_id, role=None, round_id=None)`. They are append-only (there is no method to edit or delete them) and every reference must exist. `llm_call_id` is the row id that `add_llm_call` now returns. Input and output are typed JSON objects and the record has no field for a prompt, a raw response or a credential; keeping free text out of the dicts is a rule for the caller, which the tests cannot check.
- `LlmCallEntry.role` (nullable, validated against `ROLES`) and `list_llm_calls(cycle_id, role=None)`.
- Old SQLite files open as before: the new tables are created and the nullable columns `cycles.project_id`, `attempts.round_id` and `llm_calls.role` are added when missing, without `REFERENCES` (SQLite limits that for added columns), so the code checks those references.

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

A candidate implements `fit_predict(x_train, y_train, x_valid, seed)` (the `Candidate` interface in `evaluation/candidate.py`; the engine assumes nothing else about the object). The bundle runner of ticket 7 will implement it as `fit(train)` followed by `predict(bundle_id, X)`, so the labels and the metric stay in the engine. The engine does not persist executions: `result.execution` goes to `Registry.add_execution` (the loop ticket does that).

**Goal criteria.** `Protocol` takes an optional `goal` (`GoalCriteria`): several metrics with a threshold and a direction, for example a minimum accuracy and a minimum balanced accuracy. Each metric must be registered in `METRICS` and have its natural direction. `evaluate_validation` scores every goal metric on the same validation predictions and adds a `goal/<metric>` partition metric (mean and dispersion) to the execution record; `EvaluationResult.goal_scores` and `EvaluationResult.goal` (a `GoalStatus`) carry the result. `evaluate_goal(goal, observed)` is a pure function: maximize needs value >= threshold, minimize value <= threshold, all criteria must be met, and a missing metric is an error. `EvaluationEngine.goal_values_from_execution` reads the values back. The goal is part of the protocol JSON and hash only when present, so a protocol without a goal is unchanged. No metric was added to `METRICS` (so there is still no precision or recall); a task that needs one adds it with a later ticket. **Limit:** because the goal metrics are computed on the same predictions as the main metric, they must have the same `input_kind` as the main metric (labels with labels, probabilities with probabilities); a goal that mixes them is refused. This limit is to be reviewed at ticket 12, with the first real competition. The goal is evaluated on validation, not on the holdout.

## Kaggle client (`kaggle_agent.kaggle`)

Access to Kaggle sits behind the `KaggleClient` interface. `FakeKaggleClient` is deterministic, in memory and offline, and is what every test uses. `kaggle/real.py` (`RealKaggleClient`) is a thin adapter on the official `kaggle` package, the only module that imports it, and it is never run by pytest. The choice against `kagglehub` and direct HTTP is in section 10 of the design.

- `download(competition, dest_dir=None)`: files into `data/<competition>/` by default (ignored by git), with the SHA-256 and size of each file and a combined `data_hash`.
- `get_competition_info(competition)`: description, rules page, evaluation page and data description as text, plus a read-only `pages` mapping with every page by name as Kaggle returns it (for example the FAQ). Pages are listed once and matched case-insensitively, with spaces, hyphens and underscores treated alike. If a required page (description, rules, evaluation) is missing it raises `KaggleError` naming the page and listing the available page names (never content); `data_description` is an empty string when there is no such page. Accepting the rules is a manual step on the competition website: there is no operation for it, and without it the client raises `RulesNotAcceptedError`.
- `get_submission_spec`, `daily_limit`, `get_submission` and `wait_for_score(competition, submission_id, timeout, poll_interval)`: score read until it is complete or has failed on the Kaggle side; `ScoreTimeoutError` after the timeout.
- `submit(competition, file, message)`: a template method of the base class. It always runs `validate_submission` (header and column order, row count, identifier column without gaps, duplicates or unknown ids, no empty prediction, all against the sample submission) and the daily limit check, and only then calls the `_send` hook of the implementation. `SubmissionFormatError` and `DailyLimitReachedError` are raised before anything is sent. The limit is per UTC day and comes from Kaggle in the real client; `DailyLimitGuard` (used by the fake) needs it as a required argument.
- `record_submission(registry, cycle_id, record)` and `record_score(registry, record)` in `kaggle/recording.py` store the submission and its public score in the registry as a `SubmissionEntry` (the score is accepted once). The client itself never writes to the registry, and the score never changes the evaluation protocol.

**Credentials.** Generate an API token on your Kaggle account page ("Generate New Token") and keep it outside the repo. `load_credentials()` looks in this order, first match wins:

1. `KAGGLE_API_TOKEN` environment variable;
2. `~/.kaggle/access_token` (permissions `600`);
3. legacy `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables (both);
4. legacy `~/.kaggle/kaggle.json` (permissions `600`).

An empty value counts as absent. A credentials file readable by group or others raises `InsecureCredentialsError` (`chmod 600` it). `access_token` and `kaggle.json` are in `.gitignore`. Only the Kaggle client reads credentials; they are never logged, never put in an error message, never stored in the registry and never passed to the model or the sandbox. The secret is handed to the library object directly and `os.environ` is not modified.

```python
from kaggle_agent.kaggle import FakeCompetition, FakeKaggleClient, record_score, record_submission

sample = "id,target\na,0\nb,0\n"
client = FakeKaggleClient({"demo": FakeCompetition(
    files={"train.csv": b"id,x,target\na,1,0\nb,2,1\n", "sample_submission.csv": sample.encode()},
    description="Predict the target.", rules="Rules text.", evaluation="Metric: AUC.",
    sample_submission=sample, daily_limit=2, score_script=(None, 0.8),
)})
print(client.download("demo", "/tmp/demo-data").data_hash)
print(client.daily_limit("demo"))  # limit 2, used 0, remaining 2
with open("/tmp/submission.csv", "w") as handle:
    handle.write("id,target\na,0.3\nb,0.9\n")
sent = client.submit("demo", "/tmp/submission.csv", "first try")  # validated, then sent
final = client.wait_for_score("demo", sent.submission_id, timeout=60, poll_interval=5)
print(final.status, final.public_score)  # complete 0.8
# with a registry and a cycle: record_submission(registry, "C1", sent); record_score(registry, final)
```

### Manual smoke test against real Kaggle

This is not run by pytest and needs your account and network: the automated tests never contact Kaggle. It is the only real check of the adapter.

The first run of this smoke test found a bug that the stub-based tests could not catch: `AttributeError: module 'kaggle' has no attribute '__version__'` after the HTTP response (reproduced with `kaggle==2.2.4`). The fix and its regression test are in `kaggle/real.py` and `tests/test_kaggle_real.py`.

The second run found a second bug, again invisible to the stub tests: `get_competition_info('titanic')` returned an empty `description` and `evaluation`. Kaggle's page names are not consistently lowercase (`'rules'`, `'Description'`, `'Evaluation'`, `'data-description'`, `'Frequently Asked Questions'`) and the adapter asked for exact lowercase names. The fix lists the pages once per call and matches names case-insensitively with normalized separators; `data_description` and `pages` were added. The regression tests use a stub with exactly those page names. Rerun the check below to confirm all four fields are filled.

1. Generate a token on the account page, save it to `~/.kaggle/access_token` and run `chmod 600 ~/.kaggle/access_token` (or export `KAGGLE_API_TOKEN`).
2. Choose a competition and accept its rules by hand on the website.
3. Read-only checks (no submission): run the following from the repository.

```python
from kaggle_agent.kaggle.real import RealKaggleClient

client = RealKaggleClient()  # credentials resolved with the lookup order above
print(client.daily_limit("titanic"))
info = client.get_competition_info("titanic")
print(info.evaluation[:200])
print(list(info.pages))  # the page names as Kaggle returns them
print(info.data_description[:200])
print(client.download("titanic").data_hash)
```

4. Optional and only as an explicit action of the user after the review: a single `client.submit("titanic", "submission.csv", "smoke test")` followed by `wait_for_score`. It uses one of your daily submissions. Nothing in the code base calls `submit` automatically.

Check in particular that a competition whose rules you have not accepted gives `RulesNotAcceptedError`, and that no token appears in any output.

## Claude adapter (`kaggle_agent.llm`)

Headless Claude Code with the user's subscription, never the paid API. `LlmClient.ask(prompt, output_type, permissions, purpose, model=None, role=None)` asks the model for a JSON object, validates it with a proposal type (`ShortPlan`, `Attempt`, ...) and returns an `LlmResult` with the validated `value`, the `calls` (one `CallRecord` per CLI invocation) and the `consumption` (`Consumption`: calls, tokens when reported, duration, rate limit events). Details and the verified CLI facts (version 2.1.296) are in section 8 of the design.

- `FakeLlmClient(script)`: deterministic fake used by all tests. The script is a list of steps: `Reply`, `Malformed`, `RateLimited`, `QuotaExhausted` and `CliCrash`. `valid_plan_json()` and `valid_attempt_json()` build valid payloads.
- `RealClaudeClient`: runs `claude -p --output-format json` with the prompt on stdin. It is not exported by `kaggle_agent.llm`; import it from `kaggle_agent.llm.real`. It strips `ANTHROPIC_API_KEY` (and the other paid-API variables) from the environment of the process and logs a WARNING that names the variable and never its value; the names removed appear in `stripped_variables`. The access token (for example `CLAUDE_CODE_OAUTH_TOKEN`) is never logged, stored or passed on.
- `ToolPermissions(working_dir, allowed_tools=())`: the explicit, minimal set of tools and the working directory chosen by the caller. The default allows no tool. It is the only interface the sandbox (ticket 6) will need.
- `model`: optional on the client and on each call; `None` is the default of the CLI. No model name is in the code; the model used by each call is recorded.
- `role`: optional, one of `ROLES`, last after `model`. It is copied into every `CallRecord` of the call (retries included) and stored in `llm_calls.role`; an unknown role raises `ValidationError` before any invocation. It does not select a model: the caller chooses the model per role.
- Errors: `LlmError` and its subclasses `RateLimitError`, `QuotaExhaustedError`, `CliFailureError` and `InvalidResponseError`. An invalid answer is retried a bounded number of times with the validation error passed back; a rate limit is retried with `BackoffPolicy` and an injected sleep; an exhausted quota is raised at once. The client reports consumption and enforces no budget.
- `record_call(registry, cycle_id, call)` stores a `CallRecord` in the registry (`llm_calls` table, `LlmCallEntry`): outcome, duration, tokens, model, `role` and `rejection_reason`, never a prompt, a response or a token.
- Rich JSON contract: `json_contract(output_type)` states for each field its JSON type, taken from the annotations (`str` is a string, `tuple[str, ...]` a list of strings, `bool` a boolean, `X | None` is X or null; any other annotation falls back to "any JSON value"), whether it is required, and the validation constraints. The constraints live next to the types, in an optional classmethod `contract_hints()` that returns `{field: text}` plus the special key `__all__` for rules on the whole object (`ShortPlan` and `Attempt` have one; `Suggestion` has none). `llm` knows no concrete type, so to add constraints to a new proposal type, give it a `contract_hints()` and test that a payload violating each hint is really rejected.
- `rejection_reason`: when an answer is rejected (`invalid_output`), the `CallRecord` and the `llm_calls` row hold a sanitized 'field: reason' taken from the `ValidationError`, at most 200 characters, never the model's text. A field name that is not a field of the type, the type name or an object label (`OBJECT_LABELS`) becomes `response`. Examples: `what_changed: is required`; `response: unknown key (allowed: ...)`; `plan: invalid JSON: ...` (`attempt: ...` for an `Attempt`); `plan: has 312 words, over the limit of 300`. It is `None` for every other outcome. Existing registry files gain the nullable column when opened.

```python
import functools
import tempfile

from kaggle_agent.llm import FakeLlmClient, Reply, ToolPermissions, record_call, valid_plan_json
from kaggle_agent.proposals import ShortPlan

# with a registry and a cycle: on_call=functools.partial(record_call, registry, "C1")
client = FakeLlmClient([Reply(valid_plan_json())])
permissions = ToolPermissions(working_dir=tempfile.gettempdir(), allowed_tools=("Read",))
result = client.ask("Propose a plan.", ShortPlan, permissions, purpose="plan")
assert isinstance(result.value, ShortPlan) and result.consumption.calls == 1
```

### Manual smoke test against the real CLI

Not run by pytest. It uses your Claude subscription quota (a few calls).

1. Install Claude Code and log in with your subscription (`claude` once interactively, or set `CLAUDE_CODE_OAUTH_TOKEN`). Make sure `ANTHROPIC_API_KEY` is not needed: if set, the adapter removes it and warns.
2. Run, from a Python shell in an empty directory:

```python
import tempfile
from kaggle_agent.llm import ToolPermissions
from kaggle_agent.llm.real import RealClaudeClient
from kaggle_agent.proposals import ShortPlan

client = RealClaudeClient()   # optional: model="<name or alias>"
permissions = ToolPermissions(working_dir=tempfile.mkdtemp())   # no tool
result = client.ask("Propose a short plan for a tabular classification competition.",
                    ShortPlan, permissions, purpose="smoke")
print(result.value, result.consumption, result.calls[0].model, result.stripped_variables)
```

3. Rerun the smoke test and check that the first answer is already valid (a single call, `outcome` `ok`); if it is not, read `rejection_reason` in `result.calls` (or in `llm_calls`) to see which constraint the contract is still missing.
4. Check that the answer is a valid `ShortPlan`, that tokens and the model are reported (if not, they stay `None`), that no token appears in any output, and, with `ANTHROPIC_API_KEY` exported, that a WARNING names it and the call still uses the subscription.
5. Rate limit and quota messages are not verified: if you hit one, check that it is classified as `RateLimitError` or `QuotaExhaustedError` and report the exact message so the patterns in `llm/real.py` can be fixed.

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
- `src/kaggle_agent/`: package with the subpackages `proposals` (data model), `registry` (attempt registry), `evaluation` (protocol, split, holdout, threshold), `loop`, `kaggle`, `llm`, `sandbox`, `review`, `budget`; `evaluation` holds the evaluation engine, `kaggle` the Kaggle client (interface, fake, credentials, submission rules, recording, real adapter) and `llm` the Claude adapter (interface, fake, structured output, environment, recording, real adapter); `loop`, `sandbox`, `review` and `budget` are still empty.
- `tests/`: check the 16 sections of the design, the data model and its examples, the language convention, the test map, that the package imports, and the registry behavior (deduplication, fix retries, queries, persistence), the evaluation engine (protocol, schemes, split, holdout session, threshold, hash), the Kaggle client on the fake and on a stub of the library, the Claude adapter on the fake and on a fake process runner, and that third-party libraries are imported only by `evaluation` and by `kaggle/real.py`.

Result of a manual run on macOS (Claude Code with the subscription, Python 3.14; the CLI version on that machine was not recorded): a fake `ANTHROPIC_API_KEY` was removed with a WARNING that names only the variable; the call used the subscription; the success envelope was read correctly (model `claude-sonnet-4-6`, usage reported); the result was a valid `ShortPlan`. So the format of the success envelope assumed in the design is confirmed. The first answer, however, was rejected (`invalid_output`, 7113 input and 545 output tokens) and only the retry succeeded (`ok`, 7139 and 271): about 7k input tokens per call, so a first-try failure doubles the cost. This motivated the rich contract and `rejection_reason` above; the benefit is not yet measured.

## Known limits

- The placeholder import in `load_api_class()` depends on the internals of the pinned `kaggle` version (it skips the package `__init__` and supplies only `__version__`). Recheck it, together with the source-scan test in `tests/test_kaggle_real.py`, whenever the pin in `pyproject.toml` changes.
- Adding the goal and the new modules changes the hash of the `evaluation` sources, so a protocol frozen earlier in an open cycle no longer passes `verify`.
- `Kaggle` needed no change; the data hash and the per-file SHA-256 are already in `DownloadResult`, and there is no separate snapshot accessor (the data role of a later ticket will read these).
- There is still no runnable command of the agent: only the data model, the registry, the evaluation engine, the Kaggle client and the Claude adapter exist.
- The real Kaggle adapter is checked only by the manual smoke test: its mapping of responses is tested against a stub built from the source of the official package, and the exact error for unaccepted rules is to be confirmed by hand. The policy of when a submission is worthwhile and the reserve for the last hours are not implemented (loop and review tickets); the key-like-string check of the artifacts belongs to the tickets that save artifacts. The official package has several transitive dependencies that are not pinned individually.
- The holdout guarantee is one of interface and log (one session per cycle, every read logged in the registry). Real isolation from generated code arrives with the sandbox (ticket 6).
- The hash covers the bytes of the `evaluation` sources: any edit of the package invalidates the frozen protocols of open cycles. Library versions are outside the hash and recorded in `resources`. Raising the threshold with the number of comparisons is not implemented; the count is only recorded.
- The registry is meant for a single process; multi-process access is not tested. The cycle `state` is free text until the `loop` ticket, and an iteration recorded `inconclusive` after repeated rejected proposals cannot be represented yet (open point in section 7 of the design). The semantic level of deduplication (section 7) is left to the loop ticket.
- The tests on the design check structure and keywords, not the quality of the content. The language test is a heuristic (Italian function words and identifiers), not a proof that every sentence is English.
- The word count of the plan (split on whitespace) is a simple proxy for "one page". `id` and `phase` have no format constraints yet: they belong to the registry and loop tickets.
- The guard on third-party dependencies is relaxed one package at a time: `evaluation` may import pandas, numpy and scikit-learn, and only `kaggle/real.py` may import the official `kaggle` package.
- The Claude adapter was designed against the CLI help and an error envelope only (the CLI was not logged in when it was written). The envelope of a successful call and the reported model are now verified by a manual run (`claude-sonnet-4-6`). The reporting of rate limit and quota and the effect of `CLAUDE_CODE_OAUTH_TOKEN` (it is unclear whether the run used the token or the login) are *not verified*; the tests use canned envelopes built from those assumptions, so only a manual run can confirm them. The contract may still not prevent every first-try rejection: `rejection_reason` tells which constraint is missing. Session resume is documented but not used (the validation retry is stateless). The backoff defaults are provisional until the budget ticket, and the WARNING for a stripped variable appears at every call. Tool isolation relies on CLI flags and the working directory until the sandbox exists.
