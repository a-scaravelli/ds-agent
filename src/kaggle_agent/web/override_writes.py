"""Write endpoints of the final steps, the memory across runs and the DS power overrides (ticket 30).

All of them are POST, registered with ``writes.post`` (so they are in ``WRITE_ENDPOINTS`` and wrapped in
``security.write_guard``: CSRF, Origin, then validation 400, then preconditions 409, then the AUDIT entry, then the
action). The principles:

- the DS is in control: the system warns, asks for a reason where it is risky (``overrides.REASON_REQUIRED``), records
  the action in the audit log and LABELS the result; it never blocks without a technical reason;
- frozen things (metric, validation scheme, seed, threshold, holdout) are never edited in place: a change is a new
  plan version or a new run (a fork), and ``/api/config/save`` keeps answering with the frozen note;
- single writer: nothing here writes ``registry.db``. A decision is a record in ``cockpit/decisions.jsonl`` that the
  orchestrator applies exactly once; a job (rerun, finalize, send, fork) is an orchestrator command started through the
  closed allowlist of the controller, and refused (409) while a run is active.
"""

from flask import abort, current_app, request

from kaggle_agent.overrides import classify_read, normalize_payload
from kaggle_agent.overrides.records import REASON_REQUIRED
from kaggle_agent.proposals import ALLOWED_LIBRARIES, MODEL_FAMILIES, Steering, ValidationError
from kaggle_agent.registry import NotFoundError

from . import audit, bundle_data, cycles_data, data, decisions, labels_data, store
from .controller import Conflict
from .writes import (
    FROZEN_NOTE,
    _body,
    _confirm,
    _controller,
    _ok,
    _project,
    _request_id,
    _save_decision,
    _source,
    append_limit_change,
    post,
)

PREDICT_MAX_ROWS = 100_000
SEND_WORD = "SEND"
FINISH_STATES = ("AWAITING_REVIEW", "STOPPED")


def _active(src):
    cycle_id = cycles_data.active_cycle_id(_project())
    if cycle_id is None:
        raise Conflict("the project is not started yet")
    return cycle_id, src.registry.get_cycle(cycle_id)  # NotFoundError: 404


def _reads_by_request(project, request_id: str) -> dict | None:
    records, _ = store.read_jsonl(store.cockpit_path(project, "holdout-reads.jsonl"))
    return next((r for r in records if r.get("request_id") == request_id), None)


# -- A1: the holdout panel ---------------------------------------------------------------------------


@post("/holdout/unseal")
def holdout_unseal():
    """Unseal and read the holdout: a display gate over the outcome the pipeline recorded (one session per cycle).

    ``kind`` is first (at the review), early (the run is still iterating) or second (a read exists). An early or a
    second read needs a reason and shows a warning; every read is logged, the run is labelled 'holdout re-read' and
    nothing is blocked. The registry is never written.
    """
    project, body = _project(), _body()
    request_id = _request_id(body)
    if (done := _reads_by_request(project, request_id)) is not None:
        return _ok({"read": done["id"], "kind": done["kind"], "duplicate": True})
    src = _source()
    cycle_id, cycle = _active(src)
    reads = cycles_data.reads_of(project, cycle_id)
    kind = classify_read(cycle.state, reads)
    asked = body.get("kind")
    if asked not in ("first", "early", "second"):
        raise ValidationError("kind", "must be first, early or second")
    view = data.holdout(src, cycle_id)
    if asked != kind:
        raise Conflict(f"the situation changed: this would now be a {kind} read, not a {asked} one; reload the page")
    reason = None
    if kind != "first":
        text = body.get("reason")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("reason", f"is required for this read. {view['warning']}")
        reason = decisions.text_of(body, "reason")
    _confirm(body)
    audit.record(project, f"holdout.{kind}", reason, {"cycle": cycle_id})
    record = {"id": store.new_id(), "request_id": request_id, "cycle": cycle_id, "kind": kind,
              "reason": reason or "", "who": store.os_user(), "at": store.utc_text()}
    store.append_jsonl(store.cockpit_path(project, "holdout-reads.jsonl"), record, sync=True)
    finish = None
    if kind == "early":  # ask the run to finish now: the existing limits path (rounds = rounds done)
        done_rounds = max(1, decisions.round_position(src, cycle_id))
        finish = append_limit_change(project, {"rounds": done_rounds}, reason, f"finish-{request_id}"[:64],
                                     what="limits.finish_now")["id"]
    return _ok({"read": record["id"], "kind": kind, "duplicate": False, "finish_now": finish,
                "label": None if kind == "first" else "holdout re-read"})


# -- D: decisions that override the program -----------------------------------------------------------


def _scores(src, cycle_id: str) -> dict[str, float]:
    return {r["id"]: r["mean"] for r in data.attempts(src, cycle_id) if r["mean"] is not None
            and r["outcome"] == "succeeded"}


def _check_choose(src, cycle_id, cycle, extra) -> list[str]:
    attempt_id = extra["attempt"]
    detail = src.registry.get_attempt(attempt_id)  # NotFoundError: 404
    if detail.cycle_id != cycle_id:
        raise ValidationError("attempt", f"belongs to another run ({detail.cycle_id}): it cannot be the model of this one")
    last = detail.executions[-1] if detail.executions else None
    score = next((m.mean for m in last.metrics if m.partition == "validation"), None) if last else None
    if not detail.closed or detail.outcome != "succeeded" or score is None:
        raise ValidationError("attempt", "failed or has no validation score: it cannot be the final model")
    if not (last.config.get("bundle_id") or detail.attempt.bundle_id):
        raise ValidationError("attempt", "has no bundle to persist")
    warnings = []
    scores = _scores(src, cycle_id)
    direction = data._core(src, cycle_id)["protocol"]["direction"]
    best_id = (max if direction == "max" else min)(scores, key=scores.get) if scores else None
    if best_id and best_id != attempt_id and (scores[attempt_id] < scores[best_id] if direction == "max"
                                              else scores[attempt_id] > scores[best_id]):
        warnings.append(f"this attempt scores {scores[attempt_id]:.5g} on validation; attempt {best_id} scores "
                        f"{scores[best_id]:.5g}. Your pick is kept, labelled 'chosen by the DS', and the program's "
                        "own choice stays visible.")
    if cycle.state in FINISH_STATES:
        warnings.append("the report is already written: run Finalize to persist your pick; the holdout figure "
                        "shown stays the program pick's (one holdout session per cycle).")
    return warnings


def _check_bar(src, cycle_id, extra) -> list[str]:
    state = labels_data.round_state(src, cycle_id)
    last, started = state["last_complete"], state["rounds_started"]
    number = extra["round"]
    if extra["scope"] == "cycle":
        if number < last:
            raise Conflict(f"round {number} is behind round {last}, the last complete one: this answer is stale")
        return ["the rule applies from round %d to the end of this cycle; rounds before it are untouched." % number]
    if state["comparison"] is None or number != last or last != started:
        raise Conflict(f"this answer is for round {number}, but the last complete round is {last} "
                       f"(rounds opened: {started}): it is stale and was not saved")
    if extra["attempt"] != state["candidate"]:
        raise ValidationError("attempt", f"must be the candidate of round {number} that was compared "
                                         f"({state['candidate']})")
    comparison = state["comparison"]
    if comparison.is_real:
        raise ValidationError("attempt", "already passed the bar: there is nothing to override")
    warnings = []
    if comparison.difference <= 0:
        warnings.append("this candidate is not better than the reference at all (difference "
                        f"{comparison.difference:.5g}); you are accepting a worse model on purpose.")
    return warnings


def _check_steering(extra) -> list[str]:
    steering = Steering.from_dict(extra["steering"]) if not extra.get("clear") else None
    if steering is None:
        return []
    warnings = ["the roles are TOLD to follow this steering; nothing checks the code, the reviewer report is the check."]
    if steering.allowed_model_families:
        usable = [f for f in steering.allowed_model_families
                  if f in MODEL_FAMILIES and (f not in ("xgboost", "lightgbm", "catboost") or f in _libraries())]
        if not usable:
            warnings.append("none of the model families you allow is available with the libraries of this project: "
                            "the roles will have no model to propose. It is saved anyway.")
    return warnings


def _libraries() -> tuple[str, ...]:
    controller = current_app.config.get("CONTROLLER")
    raw = store.read_toml(controller.config_path) if controller else None
    listed = ((raw or {}).get("runtime") or {}).get("allowed_dependencies")
    return tuple(listed) if isinstance(listed, list) else ALLOWED_LIBRARIES


def _override(kind: str):
    project, body = _project(), _body()
    request_id = _request_id(body)
    if (existing := decisions.find_by_request(project, request_id)) is not None:
        return _ok({"decision": existing["id"], "duplicate": True, "resumed": False, "warnings": []})
    src = _source()
    cycle_id, cycle = _active(src)
    if kind == "try_this" and "from_round" not in body:
        body = {**body, "from_round": labels_data.round_state(src, cycle_id)["rounds_started"] + 1}
    extra, address, reason = normalize_payload(kind, body)
    if kind in ("choose_model", "bar_override"):
        _confirm(body)
    if kind == "choose_model":
        warnings = _check_choose(src, cycle_id, cycle, extra)
    elif kind == "bar_override":
        warnings = _check_bar(src, cycle_id, extra)
    elif kind == "steering":
        warnings = _check_steering(extra)
    else:
        if cycle.state == "STOPPED":
            raise Conflict("the cycle is stopped: nothing would run this request. Start a new run from it instead.")
        warnings = []
    saved = _save_decision(project, body, kind, address, extra, reason or None, request_id)
    return _ok({**saved, "warnings": warnings, "needs_finalize": kind == "choose_model" and cycle.state in FINISH_STATES})


@post("/decisions/choose-model")
def decide_choose_model():
    return _override("choose_model")


@post("/decisions/bar-override")
def decide_bar_override():
    return _override("bar_override")


@post("/decisions/steering")
def decide_steering():
    return _override("steering")


@post("/decisions/try-this")
def decide_try_this():
    return _override("try_this")


# -- D13: re-run an attempt, finalize a pick made after the report ---------------------------------------------


@post("/jobs/rerun")
def job_rerun():
    project, controller, body = _project(), _controller(), _body()
    request_id = _request_id(body)
    extra, _address, _reason = normalize_payload("rerun", body)
    src = _source()
    cycle_id, _cycle = _active(src)
    detail = src.registry.get_attempt(extra["attempt"])
    if detail.cycle_id != cycle_id:
        raise ValidationError("attempt", f"belongs to another run ({detail.cycle_id})")
    if not detail.closed:
        raise Conflict("the attempt is not finished yet")
    if detail.outcome == "succeeded" and extra["seed_offset"] == 0:
        raise ValidationError("seed_offset", "a succeeded attempt is re-run with another seed: use 1 or more")
    if controller.active():
        raise Conflict("a run is active in this project: wait until it has stopped")
    saved = _save_decision(project, body, "rerun", None, extra, None, request_id, resume=False)
    if saved["duplicate"]:
        return _ok({**saved, "job": None})
    job = controller.start_job("rerun", attempt=extra["attempt"], seed_offset=extra["seed_offset"],
                               decision=saved["decision"])
    return _ok({**saved, "job": job}, 202)


@post("/jobs/finalize")
def job_finalize():
    project, controller, body = _project(), _controller(), _body()
    src = _source()
    cycle_id, cycle = _active(src)
    if cycle.state not in FINISH_STATES:
        raise Conflict("the cycle has not reported yet: a pick made now is applied at the report")
    picks = [d for d in cycles_data.decisions_of(project, cycle_id, ("choose_model",))
             if (cycles_data.outcomes(project).get(d["id"]) or {}).get("outcome") != "stale"]
    if not picks:
        raise Conflict("there is no pick of the DS to finalize: choose a model first")
    if controller.active():
        raise Conflict("a run is active in this project: wait until it has stopped")
    audit.record(project, "job.finalize", None, {"decision": picks[-1]["id"]})
    return _ok({"job": controller.start_job("finalize")}, 202)


# -- A3 and A4: reproduce, predict ---------------------------------------------------------------------------------


@post("/jobs/reproduce")
def job_reproduce():
    project, controller, _ = _project(), _controller(), _body()
    src = _source()
    cycle_id, _cycle = _active(src)
    if bundle_data.effective_model(src, cycle_id) is None:
        raise Conflict("there is no persisted bundle to reproduce yet")
    audit.record(project, "job.reproduce", None, {"cycle": cycle_id})
    return _ok({"job": controller.start_job("reproduce")}, 202)


@post("/jobs/predict")
def job_predict():
    """Predict an uploaded CSV with the stored model (multipart field ``file``; size, columns and rows are checked)."""
    project, controller = _project(), _controller()
    request.max_content_length = store.MAX_UPLOAD + 256 * 1024  # only this route takes a large body
    upload = request.files.get("file")
    if upload is None:
        raise ValidationError("file", "is required: a CSV file")
    blob = upload.read(store.MAX_UPLOAD + 1)
    if len(blob) > store.MAX_UPLOAD:
        abort(413)
    name = (upload.filename or "").lower()
    header = store.csv_header(blob) if b"\0" not in blob[:4096] else None
    if not name.endswith(".csv") or not header:
        raise ValidationError("file", "is not a CSV file (a .csv text file with a header line is expected)")
    src = _source()
    cycle_id, _cycle = _active(src)
    if bundle_data.effective_model(src, cycle_id) is None:
        raise Conflict("there is no persisted model to predict with yet")
    needed = bundle_data.training_columns(project) or []
    missing = [c for c in needed if c not in header]
    if missing:
        raise ValidationError("file", "missing columns: " + ", ".join(missing))
    rows = store.count_csv_rows(blob)
    if rows > PREDICT_MAX_ROWS:
        raise ValidationError("file", f"has {rows} rows: at most {PREDICT_MAX_ROWS} are predicted at once")
    warnings = [f"extra column ignored: {c}" for c in header if c not in needed]
    upload_id = store.new_id()[:16]
    store.write_upload(project, upload_id, blob)
    try:
        audit.record(project, "job.predict", None, {"rows": rows})
        job = controller.start_job("predict", upload=upload_id)
    except BaseException:
        store.remove_upload(project, upload_id)
        raise
    return _ok({"job": job, "rows": rows, "warnings": warnings}, 202)


# -- C8: fork --------------------------------------------------------------------------------------------------------


def _origin_threshold(project, run_id: str):
    entry = cycles_data.entry_of(project, run_id)
    protocol = (entry or {}).get("protocol") if entry else (
        store.read_json(store.project_path(project, "project.json")) or {}).get("protocol")
    return (protocol or {}).get("threshold_k")


@post("/run/fork")
def run_fork():
    """Start a NEW cycle of the project from run n / attempt m: its carry-over reaches the roles (see ``cycles``)."""
    project, controller, body = _project(), _controller(), _body()
    request_id = _request_id(body)
    origin = body.get("from_run")
    if not isinstance(origin, str) or not origin:
        raise ValidationError("from_run", "is required: the run to start from")
    attempt = body.get("from_attempt") or None
    if attempt is not None and not isinstance(attempt, str):
        raise ValidationError("from_attempt", "must be the id of an attempt")
    threshold = body.get("threshold_k")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not 0 <= threshold < 1e6):
        raise ValidationError("threshold_k", "must be a number from 0 up")
    src = _source()
    src.registry.get_cycle(origin)  # NotFoundError: 404
    if attempt is not None:
        detail = src.registry.get_attempt(attempt)
        if detail.cycle_id != origin:
            raise NotFoundError(f"attempt {attempt!r} is not in run {origin!r}")
    changed = threshold is not None and threshold != _origin_threshold(project, origin)
    reason = None
    if changed:
        reason = decisions.text_of(body, "reason")
        _confirm(body)
    existing = next((e for e in cycles_data.entries(project) if e.get("request_id") == request_id), None)
    if existing is not None:
        return _ok({"cycle": existing["cycle_id"], "duplicate": True})
    cycle_id, cycle = _active(src)
    if controller.active():
        raise Conflict("a run is active in this project: a fork starts only after it has stopped")
    if cycle.state != "STOPPED":
        raise Conflict(f"the current cycle {cycle_id} is not stopped yet: a fork starts only after its review was accepted")
    audit.record(project, "run.fork", reason, {"from_run": origin, "from_attempt": attempt,
                                               "threshold_k": threshold if changed else None})
    launched = controller.launch("fork", fork={"from_run": origin, "from_attempt": attempt, "request_id": request_id,
                                               "threshold_k": threshold if changed else None, "reason": reason})
    return _ok({**launched, "duplicate": False, "protocol": "changed" if changed else "same"}, 202)


# -- B5: the submission cockpit -----------------------------------------------------------------------------------------


def _marker(src, cycle_id):
    from . import submission_data

    return submission_data.marker_view(src, cycle_id)


@post("/submission/preview")
def submission_preview():
    project, controller, _ = _project(), _controller(), _body()
    src = _source()
    cycle_id, cycle = _active(src)
    if cycle.state != "STOPPED":
        raise Conflict(f"the cycle is in state {cycle.state}: accept the review before submitting")
    audit.record(project, "submission.preview", None, {"cycle": cycle_id})
    return _ok({"job": controller.start_job("submit-preview")}, 202)


@post("/submission/confirm")
def submission_confirm():
    """Send the previewed file exactly once, through the existing submit logic (the orchestrator, never the web)."""
    project, controller, body = _project(), _controller(), _body()
    sha = body.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValidationError("sha256", "is required: the hash of the previewed file")
    _confirm(body)
    if body.get("typed") != SEND_WORD:
        raise ValidationError("typed", f"must be the word {SEND_WORD}: type it to confirm the send")
    src = _source()
    cycle_id, cycle = _active(src)
    if cycle.state != "STOPPED":
        raise Conflict(f"the cycle is in state {cycle.state}: accept the review before submitting")
    if controller.active():
        raise Conflict("a run is active in this project: nothing is sent while it runs")
    audit.record(project, "submission.confirm", None, {"cycle": cycle_id, "sha256": sha})
    return _ok({"job": controller.start_job("submit-confirm", sha256=sha)}, 202)


@post("/submission/recover")
def submission_recover():
    """After a send that was never recorded (exit 24): 'it was NOT sent' (with a reason) or 'it WAS sent' (an id)."""
    project, controller, body = _project(), _controller(), _body()
    action = body.get("action")
    if action not in ("not_sent", "adopt"):
        raise ValidationError("action", "must be not_sent or adopt")
    reason = None
    adopt = None
    if action == "not_sent":
        reason = decisions.text_of(body, "reason")
    else:
        adopt = body.get("submission_id")
        if not isinstance(adopt, str) or not adopt.strip() or len(adopt) > 64 or not all(
                c.isalnum() or c in "._-" for c in adopt):
            raise ValidationError("submission_id", "is required: the id of the submission on the Kaggle page")
    _confirm(body)
    src = _source()
    cycle_id, _cycle = _active(src)
    marker = _marker(src, cycle_id)
    if marker is None or not marker["recovery_needed"]:
        raise Conflict("there is no unrecorded send to recover")
    if controller.active():
        raise Conflict("a run is active in this project: wait until it has stopped")
    audit.record(project, f"submission.recover.{action}", reason,
                 {"cycle": cycle_id, "sha256": marker["sha256"], "submission_id": adopt})
    job = controller.start_job("submit-confirm", sha256=marker["sha256"], assume_not_sent=(action == "not_sent"),
                               adopt_id=adopt)
    return _ok({"job": job}, 202)
