# kaggle-agent

An agent that does the iterative work of a data scientist on tabular Kaggle competitions, from downloading the data to the submission. The design (not yet implemented) has three levels with the same five role agents (coordinator, data, feature engineering, modeling, reviewer); level 1 is built first. The lead approves a short initial plan; then the agent works autonomously in rounds of several candidates (validation/test holdout split, baseline, rounds with early stop; provisional defaults: 5 rounds of 3 candidates) and the lead gives guidance after each round and steps in at the end of the cycle: accept the report or give suggestions, which start a new cycle from the existing state. The code of tickets 1 to 5 now has the types, tables and arguments that the three-level design needs (rounds, goal criteria, guidance, reports, bundles, role traces); the sandbox, the bundle runner, the role agents, the round pipeline and the web interface are tickets 6 onward and are not built. The model that reasons is headless Claude Code with the subscription (no paid API); it runs on a Mac without GPU.

Convention: the entire repository is in English (code, documentation, comments, tests, commit and PR messages).

## Current status

Design plus the first pieces of code: the design document is in [docs/design.md](docs/design.md), the `proposals` subpackage of `kaggle_agent` holds the data model of plan, attempt and suggestion (ticket 1 of section 15 of the design) the `registry` subpackage holds the attempt registry with traceability (ticket 2), the `evaluation` subpackage the frozen evaluation engine (ticket 3) the `kaggle` subpackage the Kaggle client with its fake (ticket 4) and the `llm` subpackage the headless Claude adapter with its fake (ticket 5); the `sandbox` subpackage holds the light sandbox (ticket 6, level 1); the `web` subpackage holds the read-only web interface (step 10 of the design); the `agents` subpackage holds the five role agents as typed prompts (step 8 of the design); the `bundle` subpackage holds the bundle contract and runner (step 7 of the design); the `loop`, `budget` and `review` subpackages hold the round pipeline and guidance (step 9 of the design, level 1); the `pipeline` subpackage holds the holdout evaluation, the persisted bundle, the final report, the suggestions and the end-to-end flow (step 11, level 1). the `writer` subpackage holds the Claude-driven bundle writer (step 12, part 1, level 1). There is no real modeling run yet: the orchestrator, the command line, the real lead input and the real submission are the next part of step 12.

## Stack

Python 3.12, pandas, scikit-learn, LightGBM and local MLflow are a starting point, not a constraint: libraries are chosen ticket by ticket, with the criteria of section 1 of the design. The runtime dependencies are pinned with `==` in `pyproject.toml`: pandas, numpy and scikit-learn (only `evaluation` imports them) and the official `kaggle` package (only `kaggle/real.py` imports it). The data model, the registry, the rest of the Kaggle client and the Claude adapter use only the standard library (the adapter spawns the `claude` CLI, which is not a Python dependency). Flask is pinned too and only `web` imports it.

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

```python
from kaggle_agent.proposals import GoalCriteria, Guidance, ReviewerReport, Round

goal = GoalCriteria.from_dict({"criteria": [
    {"metric": "accuracy", "threshold": 0.8, "direction": "maximize"},
    {"metric": "log_loss", "threshold": 0.4, "direction": "minimize"},
]})
round_ = Round(id="R1", tasks=("try a tree model",), candidates=("A1", "A2"))
report = ReviewerReport(id="P1", round_id="R1", text="A2 is better", created_at="2026-10-30T10:00:00+00:00")
guidance = Guidance(id="G1", round_id="R1", text="try target encoding", created_at="2026-10-30T11:00:00+00:00")
for item in (goal, round_, report, guidance):
    assert type(item).from_json(item.to_json()) == item  # identical after rereading
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

## Web interface (`kaggle_agent.web`)

A read-only, server-rendered web interface over the registry. Start it with a single command:

```bash
python -m kaggle_agent.web --db PATH
```

It binds to `127.0.0.1:8000` (options `--host` and `--port`), has no authentication and never writes: the registry file is opened with SQLite `mode=ro`, only GET routes exist, and no page has a form or input. Guidance and approvals stay outside the web at level 1. A registry file created by an older version is not migrated in read-only mode: open it once with the current version first. The command exits with code 2 and a message if the file is missing or not a registry.

Views: the list of cycles (`/`); a cycle with its rounds, their candidates and the timeline of attempts (`/cycles/<id>`); a round with tasks, candidates, reports, guidance and traces; the metric series per round and per iteration with an SVG chart (`/cycles/<id>/metrics?partition=&best=`); reviewer reports; role traces with typed input and output and outcome (`?role=&round=`); lead guidance; suggestions with the cycles they spawned; and the candidate detail (`/attempts/<id>`) with rationale, hypothesis, runs, `failure_kind`, bundle and the link to the pipeline.

- Pipeline link: the web looks in the artifacts of the runs for an entry `pipeline:<reference>`. It is a link only for `http` and `https` references; anything else is shown as text, so local files are never served. Without one the page says "no pipeline yet". The convention lives in `pipeline_reference` in `web/views.py`, to be agreed with the pipeline step.
- Direction of the metric: the registry stores only `protocol_hash`, so the direction of the metric is not readable through the registry interface. The chart highlights the maximum by default and offers a link to switch (`best=min`).
- Library: Flask 3.1.3, pinned with `==` in `pyproject.toml` (the reasons and the rejected alternatives are in section 13 of the design). Python 3.12 was checked here; 3.13 and 3.14 could not be checked in the environment that built this ticket (no network access to those interpreters), so run `pip install .` and the test suite once on 3.14 before relying on it. The transitive dependencies of Flask (werkzeug, jinja2, markupsafe, itsdangerous, click, blinker) are not pinned.

## Light sandbox (`kaggle_agent.sandbox`)

Level 1 of the design (step 6): run generated code in a child process with a timeout, a minimal environment and a SHA-256 hash of the protected files. There are no containers. `Sandbox` is the small interface (one method, `run(policy, request)`) that a container implementation can replace at level 3; `SubprocessSandbox` is the standard-library implementation (POSIX only). The bundle contract and runner (step 7) are in `kaggle_agent.bundle`.

```python
from pathlib import Path
from kaggle_agent.sandbox import (
    ProtectedPath, SandboxPolicy, SandboxRequest, SubprocessSandbox,
)

policy = SandboxPolicy(
    timeout_seconds=60,  # mandatory, no default
    protected=(
        ProtectedPath("src/kaggle_agent/evaluation", Path("src/kaggle_agent/evaluation")),  # child sees a read-only copy
        ProtectedPath("data/holdout.csv", Path("data/holdout.csv"), expose=False),  # hashed, never copied
    ),
    hidden_paths=(Path("data/holdout.csv"), Path.home() / ".kaggle"),  # never passed to the child
)
request = SandboxRequest(entry_point=Path("candidate.py"), args=("--seed", "3"), diff=None)
result = SubprocessSandbox().run(policy, request)
if result.status != "succeeded":
    print(result.status, result.failure_kind, result.reason)
```

A run goes in this order: a diff that touches a protected file is rejected (`diff_rejected`) and nothing starts; the protected files are hashed and compared with `expected_hashes` when given (a difference is `protected_modified` and nothing starts); the child starts in its own process group in an empty working directory with read-only copies under `SANDBOX_PROTECTED_DIR`; on timeout the whole group is killed (`timed_out`); leftover descendants are killed after a normal exit; stdout and stderr are kept up to `max_output_bytes`; the hashes are computed again on the originals and the copies, and any change is `protected_modified` even when the child exits with 0; a non-zero exit or a signal is `crashed`.

Failure kinds. No new kind was added: `timed_out` is `timeout`, and `crashed`, `protected_modified` and `diff_rejected` are `code_error`, with the precise cause in `result.status` and `result.reason`. **Callers must check `status`, not only `failure_kind`: `protected_modified` and `diff_rejected` are integrity failures and must never be retried as code fixes, even though they map to `code_error`.** Only `crashed` is a candidate for fix retries. Nothing in this ticket enforces that on the future caller, so step 7 and the round pipeline must keep it.

| Enforced | By convention only |
| --- | --- |
| Timeout, with the child and its process group killed | The child is not told about credentials, the holdout or the registry, but a readable absolute path is still readable |
| Minimal environment built from scratch: no Kaggle, Claude or other tokens, no `PYTHONPATH`; extra variables with secret-looking names are rejected | The caller lists the protected and hidden paths correctly |
| `HOME` and `TMPDIR` inside the run folder, so `~/.kaggle` does not resolve to the real home | Generated code is expected not to look for files outside its working directory |
| Read-only copies (files `0o444`, folders `0o555`) of the exposed protected files; the holdout is never copied | Callers retry only `crashed` results |
| SHA-256 before and after on the originals and on the copies; a change blocks the run | |
| Diffs touching protected files are rejected before running; arguments and variables that mention a hidden path are refused | |
| Output limit, stdin closed, working directory removed afterwards | |

On macOS the system adds `__CF_USER_TEXT_ENCODING` to every child process (and the interpreter may add `LC_CTYPE`): these are not set by the sandbox and are not credentials, and the sandbox tests allow them as platform-added keys. The sandbox tests also do not depend on `/proc`, which does not exist on macOS.

The child runs with `python -I` (and `-B`), which ignores `PYTHONPATH` and the user site, and the environment never carries import paths. So step 7 cannot rely on the environment for imports: it must pass every import path explicitly, for example through `SANDBOX_PROTECTED_DIR` or absolute paths in the arguments.

What the light sandbox does **not** guarantee:

- the child is the same OS user as the agent: it can chmod its own copies and write to original paths it can guess (this is detected afterwards by the hash, not prevented);
- no network isolation and no filesystem isolation: any absolute path the user can read, including credential files and the holdout, is readable if the code guesses it;
- as root, permissions do not block writes;
- a process that calls `setsid` escapes the group kill;
- on a Mac without containers the isolation is partial; real isolation needs level 3;
- diff parsing reads the paths declared in headers only.

## Bundle contract and runner (`kaggle_agent.bundle`)

Step 7 of the design, level 1. A bundle is a folder with the code, the declared dependencies and two entry points, `fit(train)` and `predict(bundle_id, X)`. `BundleRunner` is fixed code that runs it in a child process through `kaggle_agent.sandbox` (the only way a process is started) with the policy timeout, and implements the `Candidate` interface of the evaluation engine, so the frozen engine scores a bundle without changes. The bundle never reports a score: the parent computes every metric from the validated predictions.

Layout of a bundle folder:

```
my_bundle/
    bundle.json   {"bundle_id": "B1", "dependencies": ["pandas"],
                   "entry_points": {"fit": "model:fit", "predict": "model:predict"}}
    model.py      def fit(train): ...   def predict(bundle_id, X): ...
```

`fit(train)` gets a DataFrame with the features and the target column (training rows only) and returns nothing; `predict(bundle_id, X)` gets a DataFrame without the target column and returns one prediction per row. `fit` and `predict` run in two different child processes, so nothing kept in memory survives between them: `fit` saves the trained model under the directory in the environment variable `BUNDLE_MODEL_DIR` (any file format, pickle included, because only the bundle code in the child reads it) and `predict` loads it from the same directory. The signatures `fit(train)` and `predict(bundle_id, X)` and the manifest did not change. A bundle that keeps its model only in a module global now fails with `code_error` in the predict process. `load_bundle(folder)` rejects a folder with a missing or invalid manifest, without `fit` or `predict` (checked with `ast`, the code is not imported), with a path or link that escapes the folder, or with a static import of a third-party library that `dependencies` does not declare (a heuristic: dynamic imports are not seen; declared dependencies are not installed, that is level 2). The folder hash reuses the sandbox hashing helper.

```python
from pathlib import Path
from kaggle_agent.bundle import BundleRunner, load_bundle, score_bundle
from kaggle_agent.sandbox import SandboxPolicy, SubprocessSandbox

loaded = load_bundle(Path("my_bundle"))
runner = BundleRunner(
    SubprocessSandbox(), SandboxPolicy(timeout_seconds=120), loaded,
    target_column="y", input_kind="labels",  # the prediction kind of the metric
)
outcome = score_bundle(engine, runner, frame, attempt_id="A1", execution_id="E1",
                       config={}, data_hash="...")  # engine: a frozen EvaluationEngine
print(outcome.execution.outcome, outcome.execution.failure_kind)
```

Data passing. `fit_predict` (what the frozen engine calls) is `fit_model` then `predict_with_model`, two sandbox calls: the `fit` stage stages a temporary folder with `train.csv` (features and target of the training rows), the dtypes and a copy of the bundle, runs `python -I -B child.py STAGING fit` and copies the saved `bundle/model/` out; the `predict` stage stages `x.csv` (features of the rows to predict, no target column) and a copy of the model, and runs `... predict` in a fresh process. Both folders are removed afterwards. The folder hash ignores a top-level `model/` folder (it equals the sandbox `hash_path` for folders without one), so a persisted model never triggers `folder_changed`. Validation and holdout labels stay with the engine: they are never staged, never in the environment, and the tests prove it for the staging folder, the working directory, the environment and the captured output on both paths. Because the child starts with `python -I`, which ignores `PYTHONPATH`, the child script (`bundle/child.py`) adds the staged copy of the bundle to `sys.path` itself. Predictions come back as JSON, never pickle.

Validation before scoring: one value per row, no null; finite numbers for `values`, `scores` and `probabilities` (within [0, 1] for the last); scalar labels of one kind for `labels`. For `probabilities` and `scores` a rectangular matrix is also accepted (multiclass): one row per input row, the same width of at least 2 in every row, finite numbers, and for `probabilities` each cell within [0, 1] and each row summing to 1 within 1e-6. `values` and `labels` stay one-dimensional. Anything else is `malformed_predictions`. The engine scores a matrix as it is (`log_loss` works); multiclass `roc_auc` and `average_precision` are NOT supported, because the frozen metric spec allows no `multi_class` or `labels` parameter and adding it means editing `evaluation/` and changing the hash of every frozen protocol (a decision for the lead): such a run is a recorded failure with `sandbox_status` `scoring_error`.

Failures become a failed `ExecutionRecord` (`score_bundle` catches `BundleRunError` and a `ValueError` of the metric; a protocol hash problem and a `ValidationError` still propagate):

| Cause | `resources.sandbox_status` | `failure_kind` | Retry as code fix |
| --- | --- | --- | --- |
| time limit | `timed_out` | `timeout` | no |
| exception, non-zero exit | `crashed` | `code_error` | yes |
| wrong length, NaN, null, invalid matrix, wrong type | `malformed_predictions` | `code_error` | yes |
| the metric cannot score the predictions (e.g. multiclass `roc_auc`) | `scoring_error` | `code_error` | no |
| protected file changed (before or during the run) | `protected_modified` | `code_error` | never retried |
| diff touching protected files | `diff_rejected` | `code_error` | never retried |
| bundle folder changed since registration | `folder_changed` | `code_error` | never retried |

`protected_modified` and `diff_rejected` are integrity failures: they map to `code_error` only because no new kind exists, and are never retried as code fixes (`resources.retryable` is False). The runner protects its own package and the evaluation package (named `kaggle_agent/bundle` and `kaggle_agent/evaluation`) next to the caller's protected paths, and takes the hashes when it is built: it passes them as `expected_hashes` on every run, so a protected file edited after that blocks the run before the child starts. A caller path with the same name and the same location is kept once; the same name with another location raises `BundleError`. `register_bundle` stores the `BundleRecord` (id, folder hash, reference, schema) and `attempt_with_bundle` sets `Attempt.bundle_id` (the bundle must be stored before the attempt). The evaluation package is unchanged, so its code hash is unchanged.

## Role agents (`kaggle_agent.agents`)

The five roles (coordinator, data, feature engineering, modeling, reviewer) as prompts with a typed input and a typed output on `LlmClient`. Each role has a typed input made of summaries (`DataSummary`, `RegistrySummary`, ...: numbers and short text, never labels, holdout data, credentials, paths or the registry), a typed output that reuses the proposal types (`CoordinatorOutput` with `ShortPlan` and `GoalCriteria`; `DataFindings`; `Attempt` for feature engineering and modeling; `ReviewerReport`), a `build_<role>_prompt` and a `run_<role>` function. `run_<role>` calls `LlmClient.ask` with the role and a purpose, validates the output, and relies on the client for retries (`rejection_reason`) and rate limits. Invalid output after the retries raises `RoleRejectedError` (role, attempts and the sanitized reason; never the model text). The model is the caller's choice (`model=` argument).

- Roles do not call or import each other: the caller passes the output of one role to the next (the pipeline is a later step).
- The reviewer writes text only: the model fills `ReviewerNotes(text)` and `run_reviewer(..., report_id=...)` builds the `ReviewerReport` with the given id, the round of the input and the date. No field of it can decide anything.
- Feature engineering and modeling get their `id` from the caller (`candidate_id` in the input), not from the model.
- Every call writes a `RoleTrace` (typed input and output, outcome, link to the `llm_calls` row). Feature engineering and modeling return a `CandidateRoleResult` with a `PendingTrace`: register the attempt, then call `record_trace(registry, result.trace, candidate_id=attempt.id)`. Nothing is written for a successful candidate call until you do. A failed or rejected call is always traced at once, with no candidate and no output.
- The helpers store the `llm_calls` rows themselves (the trace needs their id). Do not also build the client with `on_call=functools.partial(record_call, registry, ...)` for the same registry: that is refused with `RoleConfigurationError`. Other wrappers (a lambda, a custom function) are not detected and would store every row twice.

```python
import tempfile
from datetime import datetime, timezone

from kaggle_agent.agents import (
    AttemptDigest, ColumnSummary, DataInput, DataSummary, FeatureEngineeringInput,
    DataFindings, RegistrySummary, TraceContext, record_trace, run_data, run_feature_engineering,
)
from kaggle_agent.llm import FakeLlmClient, Reply, ToolPermissions, valid_attempt_json, valid_plan_json
from kaggle_agent.proposals import Round, ShortPlan
from kaggle_agent.registry import Cycle, SqliteRegistry

registry = SqliteRegistry(":memory:")
plan = ShortPlan.from_json(valid_plan_json())
registry.add_cycle(Cycle(id="C1", plan=plan, iterations=10, stagnation_k=3, state="ITERATING"))
registry.add_round("C1", Round("R1", ("task",)))
context = TraceContext("C1", "R1")
permissions = ToolPermissions(working_dir=tempfile.gettempdir())
summary = DataSummary(100, "y", (ColumnSummary("age", "int64", 0.1, 40),))

# a one-step role: the call and its trace are written together
reply = '{"profile": "100 rows", "quality_issues": [], "leakage_hints": [], "proposed_cleaning": []}'
findings = run_data(FakeLlmClient([Reply(reply)]), registry, context,
                    DataInput(summary, "profile the data"), permissions)
assert isinstance(findings, DataFindings) and len(registry.list_role_traces("C1", role="data")) == 1

# a role with a candidate: the trace is written after the attempt is registered
role_input = FeatureEngineeringInput("A1", plan, findings, "add features", RegistrySummary())
result = run_feature_engineering(FakeLlmClient([Reply(valid_attempt_json(id="ignored"))]),
                                 registry, context, role_input, permissions)
assert result.output.id == "A1" and registry.list_role_traces("C1", role="feature_engineering") == []
registry.add_attempt("C1", result.output, "fingerprint-A1")
record_trace(registry, result.trace, candidate_id="A1")
assert registry.list_role_traces("C1", role="feature_engineering")[0].candidate_id == "A1"
```

Known limits of the roles: the prompts, `PURPOSES` and the shape of the inputs are provisional until the round pipeline; the output contract was not measured with the real model (no Claude call here), so first-try rejections may still happen; the coordinator covers plan, tasks and goal criteria but not yet the final report; nothing stops a caller from putting sensitive data in a free-text field of a summary (the guarantee is by structure); a trace of a successful candidate call that is never committed with `record_trace` cannot be repaired later (traces are append-only).

## Round pipeline and guidance (`kaggle_agent.loop`, `budget`, `review`)

The autonomous cycle (design sections 2 and 11, step 9, level 1) is fixed code. `CycleMachine` in `loop` runs the states of the design (`PLANNING`, `AWAITING_PLAN_APPROVAL`, `SPLITTING`, `BASELINE`, `ITERATING`, `AWAITING_GUIDANCE`, `REPORTING`, `AWAITING_REVIEW`, `STOPPED`) and saves `Cycle.state` in the registry after every transition. All progress inside a state is derived from the registry (`CycleView`), so a crash, a suspension or a rate limit resumes from the saved state without losing work and without repeating a finished step; a cycle in `STOPPED` is inert. `run()` returns a `CycleOutcome`: `stopped`, `waiting` (the lead has not answered: there is never an implicit approval) or `suspended` (rate limit, quota, CLI failure or a rejected coordinator, state intact).

**A round.** The coordinator sets the tasks (it reads a compact `RegistrySummary`, the reviewer reports and all the guidance). For each of C candidates the loop runs the data role, feature engineering and modeling (the loop passes the outputs between them; roles never call each other), asks the `BundleWriter` for a bundle, loads and registers it, adds the attempt and then writes the deferred traces of the two candidate roles with `candidate_id` set. The bundle runs through the runner and the frozen evaluation engine; every run is an execution record. A code error is fixed inside the same attempt (the writer gets a `FixRequest`, each fix is a new bundle) up to the cap, then the attempt is closed `failed` with `code_error`. A timeout closes `failed(timeout)`; `protected_modified`, `diff_rejected` and `folder_changed` are never retried. An invalid or duplicate proposal goes back to the roles with the rejection reason; after the cap on consecutive rejections the candidate is recorded as an inconclusive placeholder (an exploratory attempt with one `inconclusive` execution whose reason starts with `no valid proposal after`). Then the reviewer writes its report (never a gate: a fixed-code report replaces a rejected one), one paired comparison per round is recorded (the best candidate against the best so far; the baseline attempt for round 1), and the stop rules run in a fixed priority: target (the goal criteria of the frozen protocol, all met on validation), cap on consecutive failed attempts (fires at once, mid-round), N, K. A stop goes to `REPORTING`, which only calls an injected hook (the report itself is step 11).

**Setup round.** The planning and baseline traces and the baseline attempt belong to a setup round `<cycle>-R0`, created at start and excluded from N, K and M. Iteration rounds are `<cycle>-R1`, `-R2`, ... and ids are deterministic (`<round>-c2`, `<candidate>-b1`, `<candidate>-b1-f2`, `<candidate>-e3`, `<round>-rep`, `<round>-g`).

**Parameters.** `budget.CycleLimits`: N rounds (default 5), C candidates (3), K rounds without real improvement (2), M rounds in a row without a pause for guidance (1), fix retries per attempt (3), cap on consecutive failed attempts (3) and cap on consecutive rejected proposals per candidate (3). The defaults are provisional and configurable. N, K and the fix cap live in the dedicated `Cycle` fields; C, M and the two caps are stored once on the cycle (`Cycle.parameters`, written by the machine at the first start), and a resumed run always uses the stored values: the injected limits are only a fallback for a cycle that has none. A pause for guidance is due after every M-th round, also after the last one (the guidance then feeds the next cycle).

**Guidance.** At `AWAITING_GUIDANCE` the injected `LeadInput` answers with `LeadResponse`: `continue` (also an input), `guidance` (recorded with `add_guidance`, which also creates the `Suggestion`, id `<round>-g`, and read by the coordinator in the next round) or `protocol_change`. A protocol change is not guidance: the pipeline refuses it and tells the lead through `notice` that it needs a new plan approval. `None` means no answer yet. The plan is approved through `plan_decision` (`Approve` freezes the protocol once, on the transition to `SPLITTING`; `Revise` goes back to `PLANNING`) and the report through `review_decision`.

**Open interfaces.** Bundle production is an open interface, not a feature: `BundleWriter` gets the attempt and the role outputs and must write a folder that satisfies the bundle contract. Only `FakeBundleWriter` (a minimal valid bundle) exists. **A Claude-driven code writer and real command or file input for the lead are not built: they need tools and a workspace and belong to step 12. Until then the cycle runs only with fakes, and this is an open question for the lead.**

```python
from kaggle_agent.loop import CycleContext, CycleMachine, FakeBundleWriter, RoleSet
from kaggle_agent.review import SimulatedLead

# registry, engine, frame, data summary, roles (client_roles(...) or fake callables), a runner
# factory (LoadedBundle -> Candidate) and a clock are injected; nothing external is touched
# (tests/loop_helpers.py `make_setup` builds a complete fake setup):
#     ctx = CycleContext(registry=..., cycle_id=..., engine=..., frame=..., data_hash=...,
#                        data_summary=..., roles=RoleSet(...), writer=FakeBundleWriter(),
#                        runner_factory=..., lead=SimulatedLead(), workspace=..., now=...)
#     outcome = CycleMachine(ctx).run()      # CycleOutcome(status='stopped', state='STOPPED', ...)
```

Known limits of the pipeline:

- A crash between `add_attempt` and the writing of the two candidate traces leaves that candidate's feature engineering and modeling traces absent (the pending traces live in memory); nothing is repeated and the cycle goes on.
- The placeholder convention (outcome plus reason prefix) is conventional; placeholders use up an iteration number in the registry, which is harmless because N counts rounds. Orphan bundle records of rejected proposals stay in the registry.
- The change requests of a `Revise` are kept in memory until the plan is approved. After a resume the data findings given to a fix request are rebuilt from the traces, if present.
- Roles must leave a trace for the coordinator (the real ones do): the loop reads it to know that a call happened and what tasks it set. When the protocol has no goal, the reviewer is shown "beat the baseline" as a stand-in goal.
- The defaults, the prompts and the sizes of the summaries are provisional and not measured with a real model. The tests run on fakes and were not run on macOS (they avoid platform-specific paths by design).

## Final report and pipeline (`kaggle_agent.pipeline`)

Step 11 of the design, level 1. After a cycle stops the `pipeline` package finishes the work. `run_pipeline(ctx, pctx)` runs the unchanged `CycleMachine` and installs one hook, `Hooks.report`, unless the caller already set one (then the caller's hook is used and nothing else changes). `start_cycle(registry, id, plan, limits, project_id=None, from_suggestion=None)` creates the cycle in `PLANNING`.

**Flow at `REPORTING`**, each step idempotent and driven by the registry and the output folder, so a crash resumes without a second holdout read or duplicate records:

1. choose the bundle: `CycleView.reference()`, the best so far (none if every candidate failed);
2. persist it: fit on the development data only (never the holdout) with the seed the engine uses for the holdout, and write `<out>/persisted/<bundle_id>/` with `code/` (a copy of the bundle), `model/` (what `fit` saved under `BUNDLE_MODEL_DIR`) and `persisted.json` (ids, hashes of the code and of the model, seed, protocol, protocol hash, data hash, metric, validation mean and dispersion); `load_persisted` refuses a changed code or model file and names it; a valid copy is reused;
3. link it to the chosen attempt: the artifact `pipeline:<persisted dir>` is appended to the last execution of the attempt with `Registry.add_execution_artifact` (the convention `kaggle_agent.web.views` already reads, so the attempt page shows the reference instead of "no pipeline yet"); it is written as soon as the persisted bundle is valid, also if the holdout later fails;
4. evaluate the holdout once: the engine opens the one session, reads the holdout and scores `PersistedModelCandidate` (predict from the stored model, features only). Whatever happens a `HoldoutOutcome` is recorded: `succeeded` (score, rows), `failed` (a failure kind and a one-line reason, never raw stderr: a crash, malformed predictions, a changed protocol hash, a session error) or `skipped` (`no usable attempt`, no session is opened). A crash after the session opened but before the outcome was written is recorded as `failed` with `interrupted before an outcome was recorded` and the holdout is never read again;
5. derive and store the suggestions, build and store the report (`Registry.add_final_report` and `<out>/reports/<cycle>.md`), once.

**Report fields** (`FinalReport`, read back with `get_final_report` and `FinalReport.from_dict`): cycle, stop reason, metric and direction, protocol hash, the chosen bundle (attempt, bundle, hypothesis, frozen-protocol validation mean and dispersion, executions), the baseline score, the holdout outcome, every round (tasks, reviewer text, candidates with outcome, score and failure kind, comparison, any guidance), what worked, failure counts by kind, the suggestions (id and text), the persisted folder and the reproduce command. `render_text` gives a Markdown form.

**Suggestions** are derived by fixed code, with no model call, ids `<cycle>-sg-<kind>`: `stop` (a different text for stagnation, rounds used up, target reached, failure cap), `failures` (the most frequent failure kind, if it occurred at least twice), `holdout` (a failed holdout run with its reason, or a holdout worse than validation by more than the validation dispersion), `guidance-r<n>` (a lead guidance after round n that the next round did not turn into a real improvement, quoted). The guidance itself is already stored as a suggestion by `add_guidance` and is listed, not duplicated. The thresholds are provisional.

**Commands.**

```bash
python -m kaggle_agent.pipeline reproduce --persisted DIR --db registry.db --data data.csv
python -m kaggle_agent.pipeline predict --persisted DIR --input x.csv --output predictions.json
```

- `reproduce` verifies the hashes of the persisted bundle (the only place where the stored `model/` is checked, by hash), opens the registry read-only, rebuilds the protocol, checks it against the frozen hash and re-runs the frozen-protocol validation with a real `BundleRunner`, then compares the mean with the metric in the report: exit 0 and both values printed if they are equal within 1e-9, non-zero with a reason otherwise. **What it proves:** the metric is retrained from the persisted bundle *code*, partition by partition (fit and predict in separate processes, a fresh temporary model each time). It does not use the stored `model/`, which was fitted on the whole development set and belongs to no partition. So it shows that the stored code, protocol and data give the reported metric, not that the stored trained model does; a changed `model/` makes it fail only through the hash check. A different CSV shows up as a different metric (the `data_hash` in `persisted.json` is shown, not verified).
- `predict` is the command that uses the stored model: it runs only the predict stage on `model/` in a fresh process and writes a JSON list, never pickle.
- The holdout is read once per cycle, so neither command re-reads it: the holdout score is read from the registry, not recomputed.

```python
from kaggle_agent.pipeline import PipelineContext, run_pipeline, start_cycle
from kaggle_agent.pipeline.fake import FakeModelStore

# with the loop fakes (tests/loop_helpers.py `make_setup`) and a fake model store:
#     store = FakeModelStore()
#     pctx = PipelineContext(out_dir, lambda loaded: store)          # BundleRunner in production
#     result = run_pipeline(ctx, pctx)       # PipelineResult(outcome=stopped, report=FinalReport)
#     result.report.holdout, result.report.suggestions, result.report.reproduce_command
```

Known limits of the pipeline: a suggestion links a new cycle to its origin (`start_cycle(..., from_suggestion=id)` calls `link_cycle`) but does not carry over the best model or the history; spawning the next cycle by itself is the orchestrator of step 12. Level 1 does not generate the end-to-end pipelines with data loading and submission: the persisted bundle with the two commands is the deliverable. Fit and predict in two processes roughly double the evaluation time (each imports pandas). The persisted model files are bundle-defined: only bundle code in the child reads them, the parent checks a hash, and a bundle may still load untrusted bytes (the level 1 risk of the design). A resume on another machine without the output folder fits the model again. A crash between the holdout read and the write of its outcome uses up the single holdout session of that cycle. Nothing was run on macOS here: run the real-subprocess tests there (`tests/test_pipeline_command.py`, `tests/test_bundle_persistence.py`).

## Bundle writer (`kaggle_agent.writer`)

Step 12 of the design, part 1, level 1. `ClaudeBundleWriter` implements the open `loop.BundleWriter` interface with Claude: `write(BundleRequest)` returns the folder of a verified bundle. The orchestrator, the command line, the real lead input and the real Kaggle submission are the next ticket and are not built.

- **Prompt.** Built by a pure function from the role outputs and the attempt (hypothesis, rationale, what changed, measure, data findings), the schema of the development data (`DataSummary`: columns, dtypes, missing share, distinct values; never rows or labels), the target, the metric, its direction and the prediction kind, and the bundle contract: `fit(train)` and `predict(bundle_id, X)`, the model saved under `BUNDLE_MODEL_DIR` and loaded in another process, JSON-only predictions (1-D or a probability matrix), `bundle.json` with the requested `bundle_id`, the allowed imports. The writer gets no DataFrame, engine or holdout accessor.
- **Tools.** Claude gets exactly `Read`, `Write`, `Edit` and `Glob` (`WRITER_TOOLS`), no Bash and no network tool, in a fresh staging directory under `workspace_root`. No tool runs code, so the bundle is never executed by the writer.
- **Verification and repairs.** After every turn the writer checks the staged bundle with `load_bundle` (manifest, entry points, links, undeclared imports) plus: the manifest `bundle_id` equals `request.bundle_id`, every dependency is allowed (default numpy, pandas, scikit-learn), no `model/` folder, caps on files and bytes. On a failed check it makes a repair turn that carries the exact check error, at most `max_repairs` (default 2, so at most 3 Claude calls); then `InvalidBundleError`. Only a verified bundle is copied to `request.folder`. The writer never runs the evaluation and never touches the holdout.
- **Fix requests.** With `request.fix` the prompt adds the previous bundle code (`bundle.json` and `.py` files, capped) and the failure reason, and asks for a corrected bundle under the new id. The previous bundle is looked up in the registry first (the `reference` of its `BundleRecord`) and, only if there is no usable record, in the sibling folder `request.folder.parent / previous_bundle_id`; `PreviousBundleError` if neither is readable. The old folder is only read, never modified.
- **Errors and the loop.** Every writer error is a `bundle.BundleError`, so the loop maps it as before: a rejection in the propose phase, bounded fix retries then a non-retryable failed execution (`invalid_fix_bundle`) in the fix phase. `loop/` is not edited. A Claude error (`LlmError`: rate limit, quota, CLI failure) propagates unchanged, so the cycle is suspended and can be resumed. A write outside the staging directory or to a protected path is a `WorkspaceViolationError`, detected at once and not repaired; it is treated by the loop as a rejection or a fix failure, not as a stop.
- **Isolation limits.** Confinement relies on the CLI flags and the working directory. After each turn the writer compares hashes of the protected paths (the `bundle` and `evaluation` packages), of everything under `workspace_root` outside the staging directory and of `request.folder`. This is best effort: a write elsewhere on disk, or a read outside the workspace, is not detected without an OS sandbox. How the real CLI behaves with `Write` outside its directory is not verified: the manual test below should confirm it.
- **Recording.** Every Claude call of the writer becomes an `llm_calls` row (tokens, model, outcome), also when it fails. The purpose reads `bundle writer: <bundle_id> turn <n> prompt <hash12>`, where the hash is the first 12 hex characters of the SHA-256 of the prompt; the writer also keeps the full hashes in its `calls` list. The client must not also record to the same registry (`WriterConfigurationError`). The dollar cost of the writer calls is NOT recorded (`CallRecord` and the `llm_calls` table have no cost field); tokens, model and prompt hash are recorded. Adding cost needs a follow-up decision.
- **Tests.** All run without network with a scripted fake Claude (`tests/writer_helpers.py`); a valid written bundle is scored by the real `BundleRunner` in a subprocess under a frozen protocol. One MANUAL test, `tests/test_writer_manual_claude.py`, runs the real Claude on a toy dataset and is skipped unless the environment variable `KAGGLE_AGENT_RUN_CLAUDE=1` is set (it spends subscription quota).

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
- `src/kaggle_agent/`: package with the subpackages `proposals` (data model), `registry` (attempt registry), `evaluation` (protocol, split, holdout, threshold), `loop` (cycle state machine and round pipeline), `pipeline` (final report and end-to-end flow), `writer` (Claude-driven bundle writer), `kaggle`, `llm`, `agents`, `sandbox`, `bundle`, `review` (lead interface), `budget` (limits and stop rules), `web`; `evaluation` holds the evaluation engine, `kaggle` the Kaggle client (interface, fake, credentials, submission rules, recording, real adapter) and `llm` the Claude adapter (interface, fake, structured output, environment, recording, real adapter); `agents` holds the five roles; `sandbox` holds the light sandbox (interface, subprocess implementation, protected files, environment); `bundle` holds the bundle contract, the loader, the fixed-code runner and the scoring helpers; `loop`, `review` and `budget` hold the cycle, the lead interface and the limits; `pipeline` holds the end of the cycle (holdout outcome, persisted bundle, final report, suggestions, flow and the `reproduce` and `predict` commands); `web` holds the read-only web interface.
- `tests/`: check the 16 sections of the design, the data model and its examples, the language convention, the test map, that the package imports, and the registry behavior (deduplication, fix retries, queries, persistence), the evaluation engine (protocol, schemes, split, holdout session, threshold, hash), the Kaggle client on the fake and on a stub of the library, the Claude adapter on the fake and on a fake process runner, and that third-party libraries are imported only by `evaluation` and by `kaggle/real.py`.

Result of a manual run on macOS (Claude Code with the subscription, Python 3.14; the CLI version on that machine was not recorded): a fake `ANTHROPIC_API_KEY` was removed with a WARNING that names only the variable; the call used the subscription; the success envelope was read correctly (model `claude-sonnet-4-6`, usage reported); the result was a valid `ShortPlan`. So the format of the success envelope assumed in the design is confirmed. The first answer, however, was rejected (`invalid_output`, 7113 input and 545 output tokens) and only the retry succeeded (`ok`, 7139 and 271): about 7k input tokens per call, so a first-try failure doubles the cost. This motivated the rich contract and `rejection_reason` above; the benefit is not yet measured.

## Known limits

- The placeholder import in `load_api_class()` depends on the internals of the pinned `kaggle` version (it skips the package `__init__` and supplies only `__version__`). Recheck it, together with the source-scan test in `tests/test_kaggle_real.py`, whenever the pin in `pyproject.toml` changes.
- Adding the goal and the new modules changes the hash of the `evaluation` sources, so a protocol frozen earlier in an open cycle no longer passes `verify`.
- `Kaggle` needed no change; the data hash and the per-file SHA-256 are already in `DownloadResult`, and there is no separate snapshot accessor (the data role of a later ticket will read these).
- There is still no runnable command of the agent: only the data model, the registry, the evaluation engine, the Kaggle client, the Claude adapter and the light sandbox exist.
- The real Kaggle adapter is checked only by the manual smoke test: its mapping of responses is tested against a stub built from the source of the official package, and the exact error for unaccepted rules is to be confirmed by hand. The policy of when a submission is worthwhile and the reserve for the last hours are not implemented (loop and review tickets); the key-like-string check of the artifacts belongs to the tickets that save artifacts. The official package has several transitive dependencies that are not pinned individually.
- The holdout guarantee is one of interface and log (one session per cycle, every read logged in the registry). The light sandbox (ticket 6) only partly isolates generated code, see its section.
- The hash covers the bytes of the `evaluation` sources: any edit of the package invalidates the frozen protocols of open cycles. Library versions are outside the hash and recorded in `resources`. Raising the threshold with the number of comparisons is not implemented; the count is only recorded.
- The registry is meant for a single process; multi-process access is not tested. The cycle `state` is free text; the loop writes the states of section 2 of the design, and an iteration recorded `inconclusive` after repeated rejected proposals is a placeholder (section 7 of the design). The semantic level of deduplication (section 7) is left to the loop ticket.
- The tests on the design check structure and keywords, not the quality of the content. The language test is a heuristic (Italian function words and identifiers), not a proof that every sentence is English.
- The word count of the plan (split on whitespace) is a simple proxy for "one page". `id` and `phase` have no format constraints yet: they belong to the registry and loop tickets.
- The web interface is read-only and meant for localhost (default `127.0.0.1`, no login). A registry file created by an older version is not migrated in read-only mode and some views may fail on it: open it once with the current version first.
- The web interface has no pagination and finds the metric partitions with one registry read per attempt: fine for tens of candidates, not for thousands. It has no login, so do not expose it on a network.
- The guard on third-party dependencies is relaxed one package at a time: `evaluation` may import pandas, numpy and scikit-learn, and only `kaggle/real.py` may import the official `kaggle` package.
- The Claude adapter was designed against the CLI help and an error envelope only (the CLI was not logged in when it was written). The envelope of a successful call and the reported model are now verified by a manual run (`claude-sonnet-4-6`). The reporting of rate limit and quota and the effect of `CLAUDE_CODE_OAUTH_TOKEN` (it is unclear whether the run used the token or the login) are *not verified*; the tests use canned envelopes built from those assumptions, so only a manual run can confirm them. The contract may still not prevent every first-try rejection: `rejection_reason` tells which constraint is missing. Session resume is documented but not used (the validation retry is stateless). The backoff defaults are provisional until the budget ticket, and the WARNING for a stripped variable appears at every call. Tool isolation relies on CLI flags and the working directory until the sandbox exists.
- The bundle child needs pandas (and numpy) to be importable by `sys.executable` without the user site and without `PYTHONPATH`, because the child runs with `python -I`: a virtual environment or a system install works, a user-site install does not. Data crosses the process boundary as CSV, so exotic dtypes (categoricals, datetimes, mixed types) and the row index can be lost; predictions are matched by position. Fit and predict of one partition run in two child processes, each with the policy timeout, and the model goes through a temporary folder between them. Multiclass `roc_auc` and `average_precision` are not supported (see the bundle section). Isolation is the partial isolation of the light sandbox; the undeclared-import check is a static heuristic. The hashes taken when the runner is built are the baseline, so a protected file edited legitimately afterwards blocks every later run until a new runner is built.
