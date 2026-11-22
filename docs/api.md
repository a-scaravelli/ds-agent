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
- **No code.** There is no endpoint for code or diffs. `bundles` returns the records of the registry only;
  `reference` is plain text. The only bundle files served are the allowlisted downloads of ticket 30 (`persisted.json`,
  model files, the report); the `code/` folder is never served.
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

The holdout panel. `state` is `sealed` (the default: no number, no row count), `ready` (the cycle is at its review and
a read has not happened yet) or `unsealed` (a read of the DS is logged in `cockpit/holdout-reads.jsonl`). `status` is
`sealed` or the status of the recorded outcome (`succeeded`, `failed`, `skipped`) once unsealed. `score`, `rows`,
`reason`, `metric` and `attempt_id` come from the stored outcome and are `null` while sealed. `reads` are the logged
reads of the registry, `ds_reads` those of the DS (`kind` first, early or second, with the reason). `next_kind` is the
kind of the next read, `reason_required` says whether it needs a reason, `warning` is the text shown before an early or a
second read, `text` the plain explanation of the state.

```shape
data: object
state: string
status: string
score: number|null
rows: integer|null
attempt_id: string|null
reason: string|null
reads: array
ds_reads: array
metric: string|null
recorded_at: string|null
next_kind: string
reason_required: boolean
warning: string|null
text: string
```

### GET /api/runs/<id>/final-report

The stored report. `content` is the object the pipeline stored (its keys belong to the pipeline); while the holdout is sealed
its `holdout` entry carries no score and no row count. `holdout_state` is `sealed`, `ready` or `unsealed`.

```shape
data: object
available: boolean
created_at: string|null
content: object|null
holdout_state: string
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
slot is held only while the connection is open: it is freed when the client closes (detected by a socket probe
at every poll, without needing a heartbeat write), when the stream ends, or at the latest when its lease
expires; 429 is returned only while 8 live connections exist. A stream holds no registry connection.

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

A background job (`validate-config`, `preview`, `profile`, `importance`): `status` is `running`, `done`, `failed` or `timeout`;
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

## Performance analytics (ticket 29)

Everything here is GET only, uses the DEVELOPMENT (validation) data only and never touches the holdout. The data comes
from the registry (read-only) and from the files the orchestrator records in `<project>/analytics/` (see
`docs/design.md`, "Performance analytics"). Ids are validated (`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`): a malformed or unknown
id answers 404; a bad parameter answers 400. A section whose input was not recorded answers 200 with the
**not_recorded object** instead of its data:

```json
{"not_recorded": true, "code": "older_project", "reason": "this project was run before analytics were recorded ..."}
```

`code` is one of `run_failed`, `no_bundle`, `older_project`, `not_in_index`, `too_large`, `budget_exhausted`,
`recording_error`, `not_readable`, `describe_missing`, `describe_invalid`, `label_only`, `no_snapshot`,
`not_comparable`, `no_scores`, `one_class`, `too_many_classes`, `no_importance`. The shapes below are those of the
answer WITH data; a nested section (`describe`, `permutation`, `correlation`, `comparison`) may be a not_recorded object
on its own. Every file carries a schema version `v` = 1 in `analytics/index.jsonl`.

### GET /api/attempts/<id>/analytics

What is recorded for the attempt (through its last succeeded execution and `config.bundle_id`): the out-of-fold file,
the describe.json copy, the development features and the permutation importance, each with status, hash, size.

```shape
data: object
attempt_id: string
bundle_id: string
development_only: boolean
recorded: object
run_id: string
skipped_index_lines: integer
task: object
```

### GET /api/attempts/<id>/folds

Every partition score (from the registry metrics `validation/<partition>`), mean, sample spread, worst and best fold,
the stability by repetition with the "within noise" warning, and the check against the recorded copy.

```shape
data: object
best: object
count: integer
direction: string
mean: number
points: array
recorded_check: object|null
source: string
spread: number|null
stability: object
worst: object
```

### GET /api/attempts/<id>/panels

Metric panels of one repetition (`?rep=0`): confusion matrix, ROC, precision-recall, calibration, log loss and
threshold sweep for classification; residuals, error by quantile of the truth and bias for regression. A panel that
cannot be shown is `{"available": false, "reason": "not available because ..."}`. The shape is the classification one.

```shape
data: object
available: boolean
calibration: object
classes: array
confusion: object
input_kind: string
log_loss: object
positive_class: string|null
pr: object
repetition: integer
roc: object
rows: integer
sweep: object
task: string
```

### GET /api/attempts/<id>/segments

Error by segment of one feature and of two features together. Query: `min_size` (1 to 10000000, default 1), `interactions`
(0 or 1), `top` (1 to 100, default 20). `min_size` only filters the ranking, which is computed once.

```shape
data: object
caps: object
interactions: array
min_size: integer
overall_error: number
rows: integer
segments: array
task: string
```

### GET /api/attempts/<id>/hardest

The worst out-of-fold rows with their features (at most 30 columns). Query: `sort` (a returned column, default `error`),
`order` (`asc` or `desc`), `limit` (1 to 200, default 50).

```shape
data: object
columns: array
explanation: string
limit: integer
order: string
ranked_by: string
rows: array
sort: string
```

### GET /api/attempts/<id>/flags

Leakage and sanity flags (heuristics, not proof): ID-like columns, features almost identical to the target, suspicious
score jumps, a perfect out-of-fold result.

```shape
data: object
checked: array
explanation: string
flags: array
```

### GET /api/attempts/<id>/features

The describe.json of the bundle (`describe.status` is `valid`, `missing` or `invalid`), native importance, permutation
importance (from the importance job) and the feature families, with importance grouped by family.

```shape
data: object
attempt_id: string
bundle_id: string
describe: object
families: object
native: object|null
permutation: object
```

### GET /api/runs/<id>/history

Score of every succeeded attempt in order with the best so far, the baseline and the improvement bar; `direction` is
`maximize` or `minimize`.

```shape
data: object
bar: number|null
baseline: object|null
direction: string
metric: string
points: array
run_id: string
```

### GET /api/runs/<id>/compare

All attempts of a run with score, spread, round, family, number of features and time. Query: `sort` (`score`, `spread`,
`round`, `family`, `features`, `time`, `attempt`), `order`, `q`, `round`, `family`. Failed attempts show their status
and no score, and always come last.

```shape
data: object
count: integer
direction: string
families: array
rounds: array
rows: array
run_id: string
```

### GET /api/runs/<id>/paired

Two attempts (`?a=ID&b=ID`) fold by fold: differences, 95% interval, win count, plain-words verdict, the program's own
rule beside it (bar = threshold_k x baseline dispersion), why they may disagree, and the prediction correlation (F12).

```shape
data: object
a: string
b: string
comparison: object
correlation: object
direction: string
run_id: string
threshold_k: number|null
```

### GET /api/runs/<id>/side-by-side

Two or three attempts (`?ids=A,B,C`): hypothesis, features, hyperparameters and per-fold scores; no code, no paths.
One or four ids answer 400.

```shape
data: object
attempts: array
run_id: string
```

### GET /api/runs/<id>/ablation

Display only: score and spread of the attempts with and without each feature family, from the describe.json of the
bundles; attempts without one are listed in `not_provided`. No endpoint starts an ablation run.

```shape
data: object
families: object
not_provided: array
note: string
rows: array
run_id: string
```

## Final steps, memory across runs and overrides (ticket 30)

All of these are read only. The holdout number is never in any of them until the DS unsealed it. `<id>` is a cycle id.

### GET /api/runs/<id>/labels

The labels of the run, derived from the cockpit files and the registry facts (`chosen by the DS`,
`bar overridden by the DS`, `holdout re-read`, `new protocol (set by the DS)`, `steered by the DS`,
`re-run with another seed`), each with its reason; `choice` holds the program's own pick beside the DS pick.

```shape
data: object
labels: array
choice: object
```

### GET /api/runs/<id>/origin

Where the run started from: the entry of `cycles.json` and the carry-over snapshot (both `null` for the first cycle).

```shape
data: object
run_id: string
entry: object|null
carry: object|null
```

### GET /api/runs/<id>/bundle

The persisted bundle of the effective final model: folder, model and file hashes, protocol and data hash,
`library_versions` (`not recorded` where the execution has none), `files` the DS may download (never the `code/`
folder; `code` says so), `predict` the columns the stored model expects and the limits of an upload, and `reproduce`
the last reproduce job (`null` before the first).

```shape
data: object
available: boolean
run_id: string
attempt_id: string|null
bundle_id: string|null
chosen_by: string|null
metric: string|null
validation_mean: number|null
hashes: object
library_versions: object
files: array
code: object
predict: object
reproduce: object|null
```

### FILE /api/runs/<id>/bundle/files/<name>

A download (not JSON) of one allowed file: `persisted.json`, a file of `model/`, or the report. The `code/` folder,
a path with `..` and any unknown name answer 404. The hash of the file is in the `X-Content-SHA256` header.

### FILE /api/runs/<id>/report.md

The final report as Markdown, `Content-Disposition: attachment`. The holdout is masked while sealed; the labels, the
reasons and the program's own values are in the section `ds-decisions`. The print view is the page
`/cycles/<id>/report/print` (print CSS, no external asset).

### FILE /api/jobs/<id>/predictions.csv

The predictions of a finished predict job as CSV (one row per input row, the id column if the input had one).

### GET /api/runs/<id>/submissions/preview

What the submission page shows: the last dry-run (`preview`: file, sha256, rows, columns, message, daily limit left
and its reset, chosen bundle, CV, holdout status), the `history` of submissions with CV and holdout columns (the holdout
reads `sealed` until the DS unsealed it), the unrecorded-send `marker` with its recovery options (`null` when none) and
the last job.

```shape
data: object
run_id: string
preview: object|null
preview_code: integer|null
preview_lines: array|null
preview_job: object|null
history: array
marker: object|null
last_job: object|null
```

### GET /api/runs/<id>/gap

Public score versus CV versus holdout per submission, with the gaps and a plain explanation of the typical gap (small
public test sets, overfitting to the public board, optimism of the validation). Empty and pending states are explained
in `state_text`. A sealed holdout is not leaked.

```shape
data: object
run_id: string
cycle_state: string
direction: string
rows: array
explanation: string
state_text: string
```

### GET /api/runs/<id>/bar-override

The numbers behind the override panel of the last complete round: the `bar`, the paired `differences`, `wins`, `ties`,
`losses` and the interval from the analytics. `available` is false when nothing can be overridden now;
`stale_warning` says why an old override would be ignored.

```shape
data: object
available: boolean
run_id: string
round_id: string|null
round: integer|null
rounds_started: integer
candidate: string|null
reference: string|null
bar: number|null
difference: number|null
differences: object|null
interval: array|null
folds: integer|null
wins: integer|null
ties: integer|null
losses: integer|null
is_real: boolean|null
stale_warning: string|null
text: string
```

### GET /api/history

The run history of the competition: per run what it tried (hypotheses, outcomes, scores) and learned (conclusions,
reviewer texts, guidance, suggestions, DS notes), its origin, `protocol_label` (`same` or `changed`) and labels.

```shape
data: array<object>
id: string
state: string
origin: object|null
protocol_label: string
protocol_changes: object
tried: array
learned: object
labels: array
```

### GET /api/learnings

The learnings notebook: DS notes (run, attempt and competition level, with tags) and agent suggestions, searched
case-insensitively by `?q=` over text, tags and suggestions; paged by `limit` and `offset`. Every item names its source.

```shape
data: object
q: string
total: integer
limit: integer
offset: integer
items: array
```

### GET /api/overrides/pending

The overrides of the DS that wait for the orchestrator, with their state (`pending`, `applied`, `stale`, `not used`).

```shape
data: object
overrides: array
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

### POST /api/jobs/importance

Body `{"attempt": ID}`. Starts permutation importance of one attempt as a heavy background job (validation folds of
repetition 0, at most 50 features, 3 shuffles each, timeout 3600 s). 400 for a malformed id or an attempt without a
recorded bundle, 404 for an unknown attempt. 202 with the job record (`attempt` is added); poll `GET /api/jobs/<id>`, which
carries `progress` (`last` event, `events`, `percent`) for this kind. The job writes
`analytics/importance/<bundle>.json` and its events (`job_start`, `job_progress`, `job_end`) to
`cockpit/jobs/<id>.progress.jsonl`; it never writes the event log of a run or the registry.

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

## Write endpoints of ticket 30 (POST)

The same guard as above (CSRF, Origin, validation 400, preconditions 409, audit, action). Nothing here writes
`registry.db`: a decision is a record in `cockpit/decisions.jsonl` applied once by the orchestrator, a job is an
orchestrator command started through the closed allowlist of the controller (409 while a run is active). Where a
reason is required (marked *risky*) the answer without it is 400 naming `reason`; `confirm: true` is required where noted.

### POST /api/holdout/unseal

Body `kind` (`first` at the review, `early` while the run iterates, `second` after a read), `reason` (*risky*: required
for `early` and `second`; the 400 carries the warning text), `confirm: true`, `request_id`. Logs the read in
`cockpit/holdout-reads.jsonl`; an early read also records a finish-now limit change. The run is labelled
`holdout re-read` for an early or second read. The registry is not written: the pipeline took one holdout session per
cycle and this is a display gate over it.

### POST /api/decisions/choose-model

Body `attempt`, `reason` (*risky*), `confirm: true`, `request_id`. 400 for a failed attempt or one of another run; a pick
worse than the program's is accepted with a warning. Before the report the pick is the chosen bundle; after it, run
`POST /api/jobs/finalize`. The report says `chosen by the DS` and shows the program's own pick.

### POST /api/decisions/bar-override

Body `scope` (`round` or `cycle`), `round` (the last complete round: a stale address is 409; for a cycle rule the round
from which it applies), `attempt` (the candidate to accept, for `round`), `reason` (*risky*), `confirm: true`. Accepts the candidate the strict bar rejected; the stored comparison is not
changed. Label `bar overridden by the DS`. `threshold_k` is never changed here (frozen): use the fork.

### POST /api/decisions/steering

Body `steering` = {`focus`, `pin_features`, `avoid_features`, `try_families`, `avoid_families`,
`allowed_model_families`, `notes`} or `clear: true`. A contradiction or an unknown focus is 400; a family limit that excludes every
allowed library is saved with a warning. Passed to the roles as typed guidance; nothing enforces it.

### POST /api/decisions/try-this

Body `request` = {`hypothesis`, optional `constraints`}, optional `from_round` (default: the next round). Bound once to the first proposal of the next round and
evaluated under the frozen protocol. A stopped cycle is 409; a request never used is shown as `not used`.

### POST /api/jobs/rerun

Body `attempt`, `seed_offset`. Starts a new execution of a closed attempt on the same bundle (a new attempt with
`round_id` null); the original is untouched. A repeat is refused as a duplicate and names the existing attempt.

### POST /api/jobs/finalize

Persists the bundle the DS chose after the report and makes it the model of reproduce, predict and submission. It reads
no holdout.

### POST /api/jobs/reproduce

Runs the reproduce command as a background job; the result is `OK` with the reproduced and reported values or the
exact mismatch lines (`the metrics differ by <x>`, `the persisted code changed: code/<file>`).

### POST /api/jobs/predict

Multipart body, field `file` (a `.csv`). 413 over the size cap, 400 for a non-CSV file, missing columns (named) or too
many rows; extra columns give a warning. The upload is stored under `cockpit/uploads/` with a generated name and
removed after the job. 202 with the job; the predictions are at `GET /api/jobs/<id>/predictions.csv`.

### POST /api/run/fork

Body `from_run`, `from_attempt` (optional), `request_id`, `threshold_k` (optional: a different value makes a new,
labelled protocol, with a *risky* `reason` and `confirm: true`). 404 for an unknown run or attempt, 409 while a run is
active or the current cycle is not stopped. The new cycle `<slug>-n` starts with the carry-over (descriptions and
numbers, never code) as typed input of the roles. The same `request_id` never makes a second cycle.

### POST /api/submission/preview

Starts the dry run (the cycle must be STOPPED): file, hash, rows, columns, message and daily limit left. Nothing is sent.

### POST /api/submission/confirm

Body `sha256` (of the previewed file), `confirm: true`, `typed` = `SEND`. The orchestrator refuses (exit 21) when the file
changed, 22 at the daily limit, 23 while a score is pending, 24 with an unrecorded send marker. Sends exactly once.

### POST /api/submission/recover

Body `action`: `not_sent` with a `reason` (*risky*) reruns the send, or `adopt` with `submission_id` of the Kaggle page
records the existing submission and sends nothing; `confirm: true`.

### POST /api/notes (competition level)

`target` may also be `{"type": "competition"}`; a note may carry `tags`.

