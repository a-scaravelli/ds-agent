# Cockpit JSON API (read only)

The cockpit web server (`python -m kaggle_agent.web`) serves a JSON API under `/api`. The HTML
pages call the same functions (`kaggle_agent/web/data.py`), so a different front end can replace
the pages and keep this API. Version 1.

- **GET only.** Any other method answers 405 with the error envelope. Nothing is ever written.
- **Local only.** Requests whose `Host` header is not `localhost` or `127.0.0.1` get 403.
- **Envelope.** Every answer is `{"api_version": 1, "data": ..., "warnings": [...]}`.
  `warnings` lists `{"section", "reason"}` for each part of the registry file that is not
  available (an old file with fewer tables or columns); the matching `data` part is empty.
- **Errors.** `{"api_version": 1, "error": {"status", "title", "message"}}` with status 400
  (invalid filter), 403 (Host), 404 (unknown id), 405 (method) or 503 (registry unreadable).
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

## Registry file notes

The orchestrator opens the registry file for writing and turns on WAL (write-ahead logging), so a
reader never blocks it. The server opens one read-only connection per request. A read-only reader
of a WAL file needs a writable folder for the `-shm` file. An old file stays in the old journal
mode until the orchestrator opens it for writing. A request may mix two moments while the
orchestrator writes (each registry read is its own statement).
