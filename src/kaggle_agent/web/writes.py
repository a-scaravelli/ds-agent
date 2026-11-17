"""Write endpoints of the cockpit: every one is a POST wrapped in ``security.write_guard``.

CSRF token and Origin check first (403), then strict validation (400), then the preconditions (409 for a
stale question or a run that is already active), then the AUDIT entry (refused when it cannot be written),
then the action. A reason is required for stop, a limit extension and an own plan; stop, limits, accept and
an own plan need an explicit ``confirm: true``. Decisions and limit changes carry a ``request_id``: the same
id again changes nothing and returns the first answer. The web never writes the registry: decisions, limit
changes, notes and settings go to ``cockpit/`` files (see ``store``), the orchestrator applies them.
"""

import json

from flask import Blueprint, current_app, jsonify, request
from kaggle_agent.proposals import ValidationError

from . import audit, data, decisions, store
from .controller import Conflict
from .security import write_guard

API_VERSION = 1
blueprint = Blueprint("writes", __name__, url_prefix="/api")
WRITE_ENDPOINTS: list[str] = []
CONFIG_SECTIONS = ("competition", "limits", "protocol", "models")
FROZEN_NOTE = ("A frozen setting (metric, validation scheme, seed, threshold, holdout) is never edited in place: "
               "a change means a new plan version or a new run.")
LIMIT_FIELDS = ("rounds", "candidates", "stagnation")


def post(rule: str):
    """Register a POST route under /api, guarded; the list WRITE_ENDPOINTS is what the tests iterate."""
    def decorator(view):
        WRITE_ENDPOINTS.append("/api" + rule)
        guarded = write_guard(view)
        return blueprint.route(rule, methods=["POST"], provide_automatic_options=False)(guarded)
    return decorator


# -- helpers --------------------------------------------------------------------------------------


def _ok(payload, status: int = 200):
    response = jsonify({"api_version": API_VERSION, "data": payload, "warnings": []})
    response.status_code = status
    return response


def _project():
    project = current_app.config["PROJECT_DIR"]
    if project is None:
        raise Conflict("the cockpit was started without a project folder")
    return project


def _controller():
    controller = current_app.config.get("CONTROLLER")
    if controller is None:
        raise Conflict("the cockpit has no run controller")
    return controller


def _body() -> dict:
    body = request.get_json(silent=True)
    if body is None and not request.data:
        return {}
    if not isinstance(body, dict):
        raise ValidationError("body", "must be a JSON object")
    return body


def _confirm(body: dict) -> None:
    if body.get("confirm") is not True:
        raise ValidationError("confirm", "must be true: confirm the action in the dialog first")


def _reason(body: dict, required: bool = True) -> str | None:
    return decisions.text_of(body, "reason", required)


def _request_id(body: dict) -> str:
    value = body.get("request_id")
    if value is None:
        return store.new_id()
    if not isinstance(value, str) or not (8 <= len(value) <= 64):
        raise ValidationError("request_id", "must be a text of 8 to 64 characters")
    return value


def _auto_resume(project) -> bool:
    settings = store.read_json(store.cockpit_path(project, "settings.json"), {}) or {}
    return bool(settings.get("auto_resume"))


def _started(project) -> bool:
    return store.exists(store.project_path(project, "project.json"))


def _source():
    return current_app.config["CURRENT_SOURCE"]()


# -- run control ----------------------------------------------------------------------------------


@post("/run/start")
def run_start():
    project, controller = _project(), _controller()
    body = _body()
    if _started(project):
        raise Conflict("the project is already started: use Resume to continue it")
    if not store.exists(controller.config_path):
        raise Conflict("there is no config.toml yet: save the settings first")
    audit.record(project, "run.start", None, {})
    return _ok(controller.launch("start"), 202)


@post("/run/resume")
def run_resume():
    project, controller = _project(), _controller()
    _body()
    if not _started(project):
        raise Conflict("the project is not started yet: use Start")
    if controller.active():
        raise Conflict("a run is already active in this project; no second run is started")
    audit.record(project, "run.resume", None, {})
    return _ok(controller.launch("resume"), 202)


@post("/run/pause")
def run_pause():
    project, controller = _project(), _controller()
    _body()
    if not controller.active():
        raise Conflict("nothing is running, so there is nothing to pause")
    audit.record(project, "run.pause", None, {})
    controller.request_pause()
    return _ok({"pause_requested": True})


@post("/run/stop")
def run_stop():
    project, controller = _project(), _controller()
    body = _body()
    reason = _reason(body)
    _confirm(body)
    if not controller.active():
        raise Conflict("nothing is running, so there is nothing to stop")
    audit.record(project, "run.stop", reason, {})
    return _ok(controller.stop())


# -- configuration and jobs --------------------------------------------------------------------------


def _candidate(project, controller, body: dict) -> dict:
    """The merged configuration: the existing file with the form's sections replacing their keys."""
    form = body.get("config")
    if not isinstance(form, dict):
        raise ValidationError("config", "is required: the form values")
    merged = store.read_toml(controller.config_path) or {"project": {"folder": "."}}
    for section, values in form.items():
        if section not in CONFIG_SECTIONS:
            raise ValidationError(f"config.{section}", f"unknown section (editable: {', '.join(CONFIG_SECTIONS)})")
        if not isinstance(values, dict):
            raise ValidationError(f"config.{section}", "must be an object")
        target = merged.setdefault(section, {})
        for key, value in values.items():
            if value is None or value == "":
                target.pop(key, None)
            else:
                target[key] = value
        if not target:
            merged.pop(section, None)
    return merged


def _validate(project, controller, body: dict) -> tuple[dict, dict]:
    merged = _candidate(project, controller, body)
    try:
        text = store.dumps_toml(merged)
    except ValueError as exc:
        raise ValidationError("config", str(exc)) from None
    candidate = controller.config_path.with_name("config.candidate.toml")
    store.write_atomic(candidate, text)
    job = controller.start_job("validate-config", candidate)
    result = controller.wait_job(job["id"])
    outcome = (result or {}).get("result") or {}
    if result is None or result["status"] != "done" or "ok" not in outcome:
        reason = (result or {}).get("error") or "the validation did not finish"
        return {"ok": False, "errors": [{"key": "", "message": f"The check could not run: {reason}"}],
                "resolved": None, "unknown": True}, merged
    if not outcome["ok"] and any("started project cannot change" in e["message"] for e in outcome["errors"]):
        outcome["errors"] = [{**e, "message": f"{e['message']}. {FROZEN_NOTE}"} for e in outcome["errors"]]
        outcome["frozen"] = True
    return outcome, merged


@post("/config/validate")
def config_validate():
    project, controller = _project(), _controller()
    outcome, _ = _validate(project, controller, _body())
    return _ok(outcome)


@post("/config/save")
def config_save():
    project, controller = _project(), _controller()
    body = _body()
    outcome, merged = _validate(project, controller, body)
    if not outcome.get("ok"):
        return _ok({**outcome, "saved": False})
    audit.record(project, "config.save", None, {"sections": sorted(body["config"])})
    path = controller.config_path
    candidate = path.with_name("config.candidate.toml")
    if store.exists(path):
        store.copy_file(path, store.history_path(project, store.utc_text()))
    store.write_atomic(path, candidate.read_text(encoding="utf-8"))
    store.remove(candidate)
    return _ok({**outcome, "saved": True})


def _job(kind: str):
    project, controller = _project(), _controller()
    _body()
    if not store.exists(controller.config_path):
        raise Conflict("there is no config.toml yet: save the settings first")
    audit.record(project, f"job.{kind}", None, {})
    return _ok(controller.start_job(kind), 202)


@post("/jobs/preview")
def job_preview():
    return _job("preview")


@post("/jobs/profile")
def job_profile():
    return _job("profile")


# -- decisions ----------------------------------------------------------------------------------------


def _save_decision(project, body: dict, kind: str, address, extra: dict, why: str | None, request_id: str):
    """Audit first, then the decision record; the same ``request_id`` returns the first answer."""
    existing = decisions.find_by_request(project, request_id)
    if existing is not None:
        return {"decision": existing["id"], "duplicate": True, "resumed": False}
    record = {"id": store.new_id(), "request_id": request_id, "kind": kind, "address": address,
              "who": store.os_user(), "at": store.utc_text(), **extra}
    audit.record(project, f"decision.{kind}", why, {"decision": record["id"], "address": address})
    store.append_jsonl(store.cockpit_path(project, "decisions.jsonl"), record, sync=True)
    resumed = False
    if _auto_resume(project):
        controller = current_app.config.get("CONTROLLER")
        resumed = bool(controller and controller.resume_if_idle())
    return {"decision": record["id"], "duplicate": False, "resumed": resumed}


@post("/decisions/plan")
def decide_plan():
    project, body = _project(), _body()
    request_id = _request_id(body)
    if (existing := decisions.find_by_request(project, request_id)) is not None:
        return _ok({"decision": existing["id"], "duplicate": True, "resumed": False})
    action = body.get("action")
    if action not in decisions.KINDS["plan"]:
        raise ValidationError("action", "must be approve, revise or own")
    address = body.get("address")
    if isinstance(address, bool) or not isinstance(address, int) or address < 1:
        raise ValidationError("address", "must be the plan number shown on the page")
    extra, why = {}, None
    if action == "revise":
        extra = {"text": decisions.text_of(body, "text")}
    elif action == "own":
        why = _reason(body)
        _confirm(body)
        extra = {"plan": decisions.own_plan_fields(body), "reason": why}
    decisions.check_address(decisions.current_question(_source()), "plan", address)
    return _ok(_save_decision(project, body, f"plan_{action}", address, extra, why, request_id))


@post("/decisions/guidance")
def decide_guidance():
    project, body = _project(), _body()
    request_id = _request_id(body)
    if (existing := decisions.find_by_request(project, request_id)) is not None:
        return _ok({"decision": existing["id"], "duplicate": True, "resumed": False})
    action = body.get("action")
    if action not in decisions.KINDS["guidance"]:
        raise ValidationError("action", "must be guidance or continue")
    address = body.get("address")
    if isinstance(address, bool) or not isinstance(address, int) or address < 1:
        raise ValidationError("address", "must be the round number shown on the page")
    extra = {"action": action, "text": decisions.text_of(body, "text") if action == "guidance" else ""}
    decisions.check_address(decisions.current_question(_source()), "guidance", address)
    return _ok(_save_decision(project, body, "guidance", address, extra, extra["text"] or None, request_id))


@post("/decisions/review")
def decide_review():
    project, body = _project(), _body()
    request_id = _request_id(body)
    if (existing := decisions.find_by_request(project, request_id)) is not None:
        return _ok({"decision": existing["id"], "duplicate": True, "resumed": False})
    _confirm(body)
    decisions.check_address(decisions.current_question(_source()), "review", None)
    return _ok(_save_decision(project, body, "review_accept", None, {}, None, request_id))


# -- limits ---------------------------------------------------------------------------------------------


@post("/limits")
def limits():
    project, body = _project(), _body()
    request_id = _request_id(body)
    records, _ = store.read_jsonl(store.cockpit_path(project, "limit-changes.jsonl"))
    for record in records:
        if record.get("request_id") == request_id:
            return _ok({"change": record["id"], "duplicate": True, "warnings": []})
    values = body.get("values")
    if not isinstance(values, dict) or not values:
        raise ValidationError("values", "is required: rounds, candidates and/or stagnation")
    clean = {}
    for key, value in values.items():
        if key not in LIMIT_FIELDS:
            raise ValidationError(f"values.{key}", f"unknown limit (allowed: {', '.join(LIMIT_FIELDS)})")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"values.{key}", "must be an integer")
        if value < 1:
            raise ValidationError(f"values.{key}", "must be at least 1")
        clean[key] = value
    reason = _reason(body)
    _confirm(body)
    src = _source()
    warnings = []
    cycle_id = decisions.cycle_id_of(src)
    if cycle_id is not None:
        try:
            cycle = src.registry.get_cycle(cycle_id)
        except Exception:
            cycle = None
        if cycle is not None:
            done = decisions.round_position(src, cycle_id)
            if "rounds" in clean and clean["rounds"] < done:
                warnings.append(f"{clean['rounds']} rounds is below the {done} rounds already done: "
                                "the cycle will stop after the current round")
            if cycle.state in ("REPORTING", "AWAITING_REVIEW", "STOPPED"):
                warnings.append("the cycle is already past the iteration phase and cannot go back: the change "
                                "does not apply to it (start a new cycle from a suggestion)")
            if cycle.parameters is None:
                warnings.append("the cycle has not stored its limits yet: the change applies once it has")
    record = {"id": store.new_id(), "request_id": request_id, "values": clean, "reason": reason,
              "who": store.os_user(), "at": store.utc_text()}
    audit.record(project, "limits.extend", reason, {"values": clean, "change": record["id"]})
    store.append_jsonl(store.cockpit_path(project, "limit-changes.jsonl"), record, sync=True)
    return _ok({"change": record["id"], "duplicate": False, "warnings": warnings})


# -- notes, tags, settings ----------------------------------------------------------------------------


def _target(body: dict) -> tuple[str, str]:
    target = body.get("target")
    if not isinstance(target, dict) or target.get("type") not in ("run", "attempt") \
            or not isinstance(target.get("id"), str) or not target["id"]:
        raise ValidationError("target", "must be {type: run|attempt, id}")
    kind, ident = target["type"], target["id"]
    registry = _source().registry
    if kind == "run":
        registry.get_cycle(ident)
    else:
        registry.get_attempt(ident)
    return kind, ident


@post("/notes")
def notes():
    project, body = _project(), _body()
    kind, ident = _target(body)
    op = body.get("op", "add")
    if op not in ("add", "edit", "delete"):
        raise ValidationError("op", "must be add, edit or delete")
    record = {"id": store.new_id(), "op": op, "target_type": kind, "target_id": ident,
              "who": store.os_user(), "at": store.utc_text()}
    if op == "add":
        record["note_id"] = record["id"]
        record["text"] = decisions.text_of(body, "text")
    else:
        note_id = body.get("note_id")
        known = [n["id"] for n in _source().notes().get((kind, ident), {"notes": []})["notes"]]
        if note_id not in known:
            raise Conflict("there is no such note on this target")
        record["note_id"] = note_id
        if op == "edit":
            record["text"] = decisions.text_of(body, "text")
    audit.record(project, f"note.{op}", None, {"target": f"{kind}:{ident}", "note": record["note_id"]})
    store.append_jsonl(store.cockpit_path(project, "notes.jsonl"), record, sync=True)
    return _ok({"note": record["note_id"]})


@post("/tags")
def tags():
    project, body = _project(), _body()
    kind, ident = _target(body)
    raw = body.get("tags")
    if not isinstance(raw, list) or len(raw) > 20:
        raise ValidationError("tags", "must be a list of at most 20 tags")
    clean = []
    for tag in raw:
        if not isinstance(tag, str) or not tag.strip() or len(tag.strip()) > 40:
            raise ValidationError("tags", "each tag must be a text of 1 to 40 characters")
        if tag.strip() not in clean:
            clean.append(tag.strip())
    record = {"id": store.new_id(), "op": "tags", "target_type": kind, "target_id": ident, "tags": clean,
              "who": store.os_user(), "at": store.utc_text()}
    audit.record(project, "tags.set", None, {"target": f"{kind}:{ident}", "tags": clean})
    store.append_jsonl(store.cockpit_path(project, "notes.jsonl"), record, sync=True)
    return _ok({"tags": clean})


@post("/settings")
def settings():
    project, body = _project(), _body()
    if not isinstance(body.get("auto_resume"), bool):
        raise ValidationError("auto_resume", "must be true or false")
    audit.record(project, "settings.change", None, {"auto_resume": body["auto_resume"]})
    store.write_json(store.cockpit_path(project, "settings.json"), {"auto_resume": body["auto_resume"]})
    return _ok({"auto_resume": body["auto_resume"]})
