"""The submission cockpit data and the leaderboard gap view (ticket 30).

Everything here is read from the registry and from the files of the orchestrator (``out/submissions/<cycle>.pending.json``,
the job results). The web never calls Kaggle and never reads a credential: the dry run and the send are orchestrator
jobs that use the existing submit logic.
"""

from . import bundle_data, cycles_data, data, store

GAP_TEXT = (
    "The public leaderboard score is computed by Kaggle on a part of the test set. It is normally NOT equal to your "
    "validation (CV) score or to the holdout score, for three plain reasons: (1) the public part of the test set is "
    "small, so its score is noisy; (2) a model chosen because it scored best on validation is slightly optimistic on "
    "validation (CV optimism); (3) if you keep picking what scores best on the public board you overfit to it. A "
    "small gap in either direction is normal; a large one is a reason to look at the data split, not at the model.")
EMPTY_TEXT = "No submission yet: nothing to compare. Send one from the submission page after the review."
PENDING_TEXT = "Sent, the public score is still pending: the gap shows when Kaggle has scored it."
SEALED = "sealed"


def marker_view(src, run_id: str) -> dict | None:
    marker = store.read_json(store.out_file(src.project_dir, "marker", run_id)) if src.project_dir else None
    if not isinstance(marker, dict):
        return None
    entries = src.section("submissions", lambda: src.registry.list_submissions(run_id), [])
    known = any(e.file_sha256 == marker.get("sha256") for e in entries)
    needs = marker.get("state") == "sending" and not known
    return {"state": marker.get("state"), "sha256": marker.get("sha256"), "message": marker.get("message"),
            "submission_id": marker.get("submission_id"), "at": marker.get("at"),
            "recovery_needed": needs,
            "text": ("A send started but was never recorded (exit 24). It may or may not have reached Kaggle: "
                     "check the Kaggle submissions page, then choose what happened.") if needs else None}


def _cv_of(src, run_id: str) -> float | None:
    model = bundle_data.effective_model(src, run_id)
    if model is None or not model.get("attempt_id"):
        return None
    try:
        detail = src.registry.get_attempt(model["attempt_id"])
    except Exception:
        return None
    for e in reversed(detail.executions):
        for m in e.metrics:
            if m.partition == "validation" and e.outcome == "succeeded":
                return m.mean
    return None


def _holdout_for(src, run_id: str):
    view = data.holdout(src, run_id)
    return view, (view["score"] if view["state"] == "unsealed" else None)


def history(src, run_id: str) -> dict:
    src.registry.get_cycle(run_id)
    cv = _cv_of(src, run_id)
    holdout_view, holdout_score = _holdout_for(src, run_id)
    unsealed = holdout_view["state"] == "unsealed"
    rows = []
    for e in data.submissions(src, run_id):
        rows.append({"submission_id": e["submission_id"], "status": e["status"], "public_score": e["public_score"],
                     "submitted_at": e["submitted_at"], "message": e["message"], "file_sha256": e["file_sha256"],
                     "cv": cv, "holdout": holdout_score if unsealed else SEALED})
    return {"run_id": run_id, "submissions": rows, "metric": None, "marker": marker_view(src, run_id)}


def preview_view(src, run_id: str) -> dict:
    job = store.latest_job(src.project_dir, ("submit-preview", "submit-confirm", "submit-recover"))
    preview = store.latest_job(src.project_dir, ("submit-preview",))
    result = (preview or {}).get("result") or {}
    return {"run_id": run_id, "preview_job": preview, "preview": result.get("data"),
            "preview_lines": result.get("lines"), "preview_code": result.get("code"),
            "last_job": job, "marker": marker_view(src, run_id),
            "history": history(src, run_id)["submissions"]}


def gap(src, run_id: str) -> dict:
    cycle = src.registry.get_cycle(run_id)
    base = history(src, run_id)
    direction = data._core(src, run_id)["protocol"]["direction"]
    rows = []
    for s in base["submissions"]:
        public, cv, holdout = s["public_score"], s["cv"], s["holdout"]
        pending = public is None
        row = {"submission_id": s["submission_id"], "status": s["status"], "public": public, "cv": cv,
               "holdout": holdout, "state": "pending" if pending else "scored",
               "gap_cv": None if pending or cv is None else public - cv,
               "gap_holdout": None if pending or holdout in (None, SEALED) else public - holdout}
        row["public_worse_than_cv"] = None if row["gap_cv"] is None else (
            row["gap_cv"] < 0 if direction == "max" else row["gap_cv"] > 0)
        rows.append(row)
    text = EMPTY_TEXT if not rows else (PENDING_TEXT if all(r["state"] == "pending" for r in rows) else None)
    return {"run_id": run_id, "direction": direction, "rows": rows, "explanation": GAP_TEXT, "state_text": text,
            "cycle_state": cycle.state}
