# Cockpit JSON API

The cockpit web server (`python -m kaggle_agent.web`) serves a JSON API under `/api`. The HTML
pages call the same functions (`kaggle_agent/web/data.py`), so a different front end can replace
the pages and keep this API. Version 1.

- **GET, except the write endpoints.** Every endpoint is GET only, EXCEPT the POST endpoints listed under
  "Write endpoints" below; any other method answers 405 with the error envelope. The web never writes
  the registry: writes go to files in `<project>/cockpit/` and the orchestrator applies them.
- **Local only.** Requests whose `Host` header is not `localhost` or `127.0.0.1` get 403.
- **Envelope.** Every answer is `{"api_version": 1, "data": ..., "warnings": [...]}`.
  `warnings` lists `{"section", "reason"}` for each part of the registry file that is not
  available (an old file with fewer tables or columns); the matching `data` part is empty.
- **Errors.** `{"api_version": 1, "error": {"status", "title", "message"}}` with status 400
  (invalid filter or body), 403 (Host, CSRF token or Origin), 404 (unknown id), 405 (method), 409 (a stale
  question, or a run already active), 429 (too many live streams) or 503 (registry or audit log unreadable).
- **Ids.** Run (cycle), attempt and round ids may contain a slash; they are written as they are
  in the path.
- **No code.** There is no endpoint for bundle files, code or diffs: the web never reads a bundle
  folder. `bundles` returns the records of the registry only; `reference` is plain text.
- **Numbers.** `NaN` and infinity are never sent (they become `null`).
- **No holdout number is computed.** `holdout` shows what the registry recorded; before a read
  it is `{"status": "sealed"}` with empty fields.

Types in the shapes below: `string`, `integer`, `number`, `boolean`, `null`, `object`, `array`;
`a|b` means either. A shape lists the exact top-level keys of `data` (or of each item when `data`
is an array); nested values are typed as `object` or `array`. A test checks every endpoint
against these shapes.

## Endpoints

### GET /api/meta

Server information. `registry` holds `mode` (`read-only`), `journal_mode` (for example `wal`, or
`null` when unknown) and `wal` (boolean or `null`).

```shape
data: object
api_version: integer
version: string|null
registry: object
project_file: boolean
```

### GET /api/runs

Every cycle of the registry (a project may have several), as run rows. `status` is one of
`running`, `waiting`, `succeeded`, `failed`, `unknown` (derived from `state` and the stop reason;
the registry has no suspended state). `best` and `chosen` are `null` until a scored attempt
exists. `chosen.model` is the bundle id (the registry stores no model name). `holdout` is
`{"status": "sealed", ...}` unless it was read. `submission` is the latest one.

```shape
data: array<object>
id: string
project_id: string|null
state: string
status: string
plan: object
rounds_done: integer
rounds_planned: integer
attempts_total: integer
best: object|null
chosen: object|null
holdout: object
submission: object|null
direction: string
protocol_hash: string|null
notes: array
tags: array
best_seen: object|null
improvement_rule: object
limits_effective: object
```

### GET /api/runs/<id>

The run row plus protocol, limits, stop reason, what worked and what failed. Query: `best=max|min`
(used only when the direction is known from neither `project.json` nor the comparisons).
`protocol` has `metric`, `direction`, `direction_source`, `scheme`, `repeats`, `threshold_k`,
`seed`, `hash`, `sources` (where each value came from) and `project_file`.

```shape
data: object
id: string
project_id: string|null
state: string
status: string
plan: object
rounds_done: integer
rounds_planned: integer
attempts_total: integer
best: object|null
chosen: object|null
holdout: object
submission: object|null
direction: string
protocol_hash: string|null
created_at: string|null
stop_reason: string|null
baseline: object|null
worked: array
failed: array
counts: object
protocol: object
limits: object
has_report: boolean
notes: array
tags: array
best_seen: object|null
improvement_rule: object
limits_effective: object
limits_view: object
```

### GET /api/runs/<id>/chart

Score per attempt for the overview chart: mean and dispersion of the validation partition, the
baseline, the improvement bar (baseline plus the comparison threshold in the metric direction)
and the chosen attempt. Query: `best=max|min` as above. Failed attempts have no point.

```shape
data: object
run_id: string
partition: string
direction: string
direction_source: string
metric: string|null
baseline: object|null
threshold: number|null
bar: number|null
chosen_attempt_id: string|null
points: array
```

### GET /api/runs/<id>/rounds

```shape
data: array<object>
id: string
position: integer
tasks: array
is_setup: boolean
candidates: array
complete: boolean
```

### GET /api/runs/<id>/rounds/<round_id>

One round with its candidates (each with the comparison verdict and the difference against the
bar: `vs_bar` is the difference minus the threshold), the reviewer reports (Markdown text), the
guidance given and the role outputs grouped by role (`data`, `feature_engineering`, `modeling`,
`coordinator`, `reviewer`).

```shape
data: object
id: string
run_id: string
position: integer
tasks: array
is_setup: boolean
candidates: array
reviewer_reports: array
guidance: array
role_outputs: object
direction: string
```

### GET /api/runs/<id>/attempts

Attempts of a run. Filters (all optional, combined with AND): `outcome` (`succeeded`, `failed`,
`inconclusive` or `open`), `round` (round id), `min_score`, `max_score` (numbers, on the mean
validation score; an attempt without a score never matches a score filter) and `q` (text, case
insensitive, in id, hypothesis, rationale, what changed, reason and evidence). A malformed number
or an unknown outcome answers 400.

```shape
data: array<object>
id: string
run_id: string
iteration: integer
round_id: string|null
parent_id: string|null
phase: string
outcome: string|null
failure_kind: string|null
reason: string|null
hypothesis: string
what_changed: string
mean: number|null
dispersion: number|null
runs: integer
closed: boolean
bundle_id: string|null
chosen: boolean
notes: array
tags: array
```

### GET /api/runs/<id>/executions

Every execution of every attempt of the run, in attempt order.

```shape
data: array<object>
id: string
attempt_id: string
sequence: integer
outcome: string
failure_kind: string|null
reason: string|null
seed: integer
code_hash: string
data_hash: string
protocol_hash: string
duration_seconds: number
metrics: array
config: object
resources: object
artifacts: array
library_versions: object
```

### GET /api/runs/<id>/comparisons

```shape
data: array<object>
candidate_attempt_id: string
reference_attempt_id: string
difference: number
threshold: number
is_real: boolean
vs_bar: number
verdict: string
```

### GET /api/runs/<id>/reviewer-reports

```shape
data: array<object>
id: string
round_id: string
text: string
created_at: string
```

### GET /api/runs/<id>/guidance

```shape
data: array<object>
id: string
round_id: string
text: string
created_at: string
suggestion_id: string
```

### GET /api/runs/<id>/submissions

```shape
data: array<object>
cycle_id: string
competition: string
submission_id: string
file_sha256: string
message: string
submitted_at: string
status: string
public_score: number|null
completed_at: string|null
```

### GET /api/runs/<id>/holdout

`status` is `sealed` (nothing recorded: no number), or the status of the recorded outcome
(`succeeded`, `failed`, `skipped`), or `read` when only a read was logged. `score`, `rows`,
`reason` and `metric` come from the stored outcome; `reads` are the logged reads.

```shape
data: object
status: string
score: number|null
rows: integer|null
attempt_id: string|null
reason: string|null
reads: array
metric: string|null
recorded_at: string|null
```

### GET /api/runs/<id>/final-report

The stored report. `content` is the object the pipeline stored (its keys belong to the pipeline).

```shape
data: object
available: boolean
created_at: string|null
content: object|null
```

### GET /api/runs/<id>/bundles

The bundle records of the registry. Nothing is read from the folder or address in `reference`.

```shape
data: array<object>
id: string
run_id: string
folder_hash: string
reference: string
schema: object
```

### GET /api/runs/<id>/llm-calls

Calls of the language model, in insertion order. Filters: `role` (a known role, else 400),
`outcome`, `q` (text in purpose, model and rejection reason). The registry stores no prompt:
`purpose_hash` is a short hash of the purpose text. `index` counts from 1 in the run.

```shape
data: array<object>
index: integer
role: string|null
purpose: string
purpose_hash: string
outcome: string
started_at: string
duration_seconds: number
input_tokens: integer|null
output_tokens: integer|null
attempt: integer
model: string|null
rejection_reason: string|null
```

### GET /api/runs/<id>/role-traces

Stored role traces without their input and output. Filters: `role`, `round` (round id, 404 if
unknown) and `candidate` (attempt id). `n` counts from 1 in the run and is the number used by the
next endpoint.

```shape
data: array<object>
n: integer
role: string
round_id: string
candidate_id: string|null
input_type: string
output_type: string|null
outcome: string
llm_call_id: integer|null
created_at: string
```

### GET /api/runs/<id>/role-traces/<n>

One role trace with its typed input and output (`output` is `null` when the call produced none).

```shape
data: object
n: integer
role: string
round_id: string
candidate_id: string|null
input_type: string
output_type: string|null
outcome: string
llm_call_id: integer|null
created_at: string
input: object
output: object|null
```

### GET /api/attempts/<id>

An attempt: the proposal, the result, the executions (retries) and the hashes. `library_versions`
come from the execution resources. `bundle` is the registry record of the bundle (text only).
`pipeline` is `{"ref", "url"}` for a `pipeline:` artifact (`url` only for http and https).

```shape
data: object
id: string
run_id: string
round_id: string|null
iteration: integer
phase: string
parent_id: string|null
closed: boolean
conclusion: string|null
fingerprint: string
hypothesis: string
rationale: string
evidence: array
what_changed: string
expected_effect: string|null
measure: string
risks: array
exploratory: boolean
estimated_cost: string|null
artifacts: array
outcome: string|null
failure_kind: string|null
reason: string|null
mean: number|null
dispersion: number|null
chosen: boolean
comparison: object|null
retries: integer
duration_seconds: number
seed: integer|null
code_hash: string|null
data_hash: string|null
protocol_hash: string|null
library_versions: object
executions: array
bundle_id: string|null
bundle: object|null
pipeline: object|null
role_trace_count: integer
direction: string
notes: array
tags: array
```

### GET /api/attempts/<id>/executions

```shape
data: array<object>
id: string
attempt_id: string
sequence: integer
outcome: string
failure_kind: string|null
reason: string|null
seed: integer
code_hash: string
data_hash: string
protocol_hash: string
duration_seconds: number
metrics: array
config: object
resources: object
artifacts: array
library_versions: object
```

### GET /api/suggestions

```shape
data: array<object>
id: string
text: string
origin_cycle: string
spawned_cycles: array
```

## Live and cockpit endpoints (GET)

### GET /api/events

The event log of the run, polling fallback of the stream. Query: `after` (event id, default 0) and
`limit` (default 200, at most 1000). Events come in id order; `last_id` is the id to ask next. Each event is
`{"id", "at" (UTC), "type", "cycle", "data"}`; `data` is a small payload (role names, counts, tokens,
durations, outcome codes, fold k of N, partial scores), never a prompt or a secret.

```shape
data: object
events: array
last_id: integer
```

### SSE /api/events/stream

Server-sent events from the same log. Same origin only. It honours the `Last-Event-ID` header (or `after`),
sends an `id:` per event, a heartbeat comment (`: heartbeat`) every 15 s and closes after at most 60 s, so
the browser reconnects and resumes with no loss and no duplicate. It closes at once when the server stops. At
most 8 streams are open at once: the next answers 429 with `Retry-After` and the page polls instead. A
stream holds no registry connection.

### GET /api/console

The last bytes of the captured output of the orchestrator process (`tail`, bytes, default 8192, at most
65536), with token-like text replaced by `[redacted]`.

```shape
data: object
text: string
bytes: integer
```

### GET /api/run

The run controller state (`idle`, `running` with its stage, `waiting` with what for, `suspended` (quota or
login), `finished`, `failed`, `paused`, `stopped`, `stopping`; `not responding` when the heartbeat is old), the
limits of the cycle (`base`, `changes`, `effective`), the history and the settings.

```shape
data: object
run: object
limits: object|null
history: array
settings: object
project_started: boolean
```

### GET /api/decisions/pending

The question the cycle waits for (`plan`, `guidance` or `review`, with its address and what the page needs to
show), every saved web decision with its status (`pending`, `applied`, `stale`), the conflicts with
`lead.toml` (names and numbers only), the lead file summary and the notices of the run.

```shape
data: object
question: object|null
decisions: array
conflicts: array
lead_file: object|null
notices: array
```

### GET /api/audit

The audit log: every action of the DS (`who`, `at`, `what`, `why`, `payload`) merged chronologically with the
relevant events of the run. Query: `limit`.

```shape
data: object
entries: array
```

### GET /api/notes

Notes and tags per run or attempt (the latest record wins; a deleted note is gone from the view).

```shape
data: object
targets: array
```

### GET /api/environment

The environment cards. The Claude card is `checking` until its background check ends. Query `refresh=1`
starts a new check. Every card degrades to `unknown`.

```shape
data: object
kaggle: object
claude: object
libraries: object
openmp: object
python: object
disk: object
checked_at: number
```

### GET /api/jobs/<id>

A background job (`validate-config`, `preview`, `profile`): `status` is `running`, `done`, `failed` or `timeout`;
`result` is the JSON the job wrote.

```shape
data: object
id: string
kind: string
status: string
started_at: string
finished_at: string|null
error: string|null
result: object|null
```

### GET /api/config

The current `config.toml` as parsed, whether the project is started, which keys are frozen and the allowed
libraries (display only). There is no secret in it.

```shape
data: object
exists: boolean
config: object|null
started: boolean
frozen: array
allowed_libraries: array
```

## Write endpoints (POST)

Every one needs the CSRF token (header `X-CSRF-Token` or the form field `csrf_token`) and an `Origin` that
matches the `Host`; otherwise 403 and nothing is written. The body is JSON. An action is first recorded in
`cockpit/audit.jsonl` (who, when, what, why); if that fails the action is refused with 503. `request_id`
makes a decision or a limit change idempotent: the same id again changes nothing. `confirm: true` is
required where noted. Answers are `{"api_version": 1, "data": ..., "warnings": []}`.

### POST /api/run/start

Starts the orchestrator as a detached process (the project must not be started yet and `config.toml` must
exist). 202 with `pid`; 409 if a run is active.

### POST /api/run/resume

Resumes a started project. 409 if a run is already active (a second run is never started).

### POST /api/run/pause

Asks the run to pause after the current call or candidate, at a safe checkpoint. 409 if nothing runs.

### POST /api/run/stop

Body `reason` (required) and `confirm: true`. SIGTERM to the process group, SIGKILL after the grace period.

### POST /api/config/validate

Body `{"config": {"competition": {...}, "limits": {...}, "protocol": {...}, "models": {...}}}`. Returns
`{"ok", "errors": [{"key", "message"}], "resolved"}` from the strict validation of the orchestrator.

### POST /api/config/save

Same body. On `ok` the previous `config.toml` is kept in `cockpit/config-history/` and the new one is
written atomically. Frozen settings of a started project are refused with the reason that a change means a
new plan version or a new run.

### POST /api/jobs/preview

Starts the protocol preview as a background job; returns the job record (read `/api/jobs/<id>`).

### POST /api/jobs/profile

Starts the data profile (development part only) as a background job.

### POST /api/decisions/plan

Body `action` (`approve`, `revise` with `text`, or `own` with `plan` = {metric, validation,
feature_engineering, models, baseline, tasks}, `reason` and `confirm: true`), `address` (the plan number
asked). A stale address answers 409. The own plan is recorded as plan version n approved by the DS.

### POST /api/decisions/guidance

Body `action` (`guidance` with `text`, or `continue`) and `address` (the round number asked).

### POST /api/decisions/review

Body `confirm: true`. Accepts the final report.

### POST /api/limits

Body `values` (`rounds`, `candidates`, `stagnation`: integers of at least 1), `reason`, `confirm: true`. The
change is appended to `cockpit/limit-changes.jsonl`; the orchestrator overlays it on the stored limits (never
rewritten), also after a resume. A value below what is already done is accepted with a warning.

### POST /api/notes

Body `target` = `{"type": "run" | "attempt", "id"}`, `op` (`add`, `edit`, `delete`), `text`, `note_id`.

### POST /api/tags

Body `target` and `tags` (a list of at most 20 short texts; the latest list replaces the earlier one).

### POST /api/settings

Body `auto_resume` (boolean): resume the run by itself after a decision is saved.

## Registry file notes

The orchestrator opens the registry file for writing and turns on WAL (write-ahead logging), so a
reader never blocks it. The server opens one read-only connection per request. A read-only reader
of a WAL file needs a writable folder for the `-shm` file. An old file stays in the old journal
mode until the orchestrator opens it for writing. A request may mix two moments while the
orchestrator writes (each registry read is its own statement).
