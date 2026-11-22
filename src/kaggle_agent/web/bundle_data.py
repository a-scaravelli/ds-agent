"""The persisted bundle of a run, what the DS may download, and the reproduce and predict jobs (ticket 30).

Reads ``out/persisted/<bundle>/persisted.json`` and the model files it names (paths from validated ids only, see
``store.out_file``). The code of the bundle is NEVER listed or served: the DS never sees or writes code. The
library versions come from the execution resources recorded at run time; an older run shows 'not recorded'.
"""

from pathlib import Path

from kaggle_agent.registry import NotFoundError

from . import cycles_data, data, store

NOT_RECORDED = "not recorded"
REPORT_NAME = "report.md"
META_NAME = "persisted.json"


def effective_model(src, run_id: str) -> dict | None:
    """The final model: ``out/chosen/<cycle>.json`` (a DS pick after the report) or the stored report's."""
    project = src.project_dir
    chosen = store.read_json(store.out_file(project, "chosen", run_id)) if project else None
    if isinstance(chosen, dict) and isinstance(chosen.get("bundle"), str):
        return {"bundle_id": chosen["bundle"], "attempt_id": chosen.get("attempt"), "source": "the DS"}
    stored = src.section("report", lambda: src.registry.get_final_report(run_id), None)
    picked = (stored.content.get("chosen") if stored else None) or {}
    if not picked.get("bundle_id") or not stored.content.get("persisted_path"):
        return None
    return {"bundle_id": picked["bundle_id"], "attempt_id": picked.get("attempt_id"),
            "source": picked.get("chosen_by", "the program")}


def _meta(src, bundle_id: str) -> dict | None:
    meta = store.read_json(store.out_file(src.project_dir, "persisted", bundle_id))
    return meta if isinstance(meta, dict) else None


def library_versions(src, attempt_id) -> dict | str:
    try:
        detail = src.registry.get_attempt(attempt_id)
    except (NotFoundError, TypeError):
        return NOT_RECORDED
    for execution in reversed(detail.executions):
        versions = (execution.resources or {}).get("library_versions")
        if isinstance(versions, dict) and versions:
            return {str(k): str(v) for k, v in versions.items()}
    return NOT_RECORDED


def training_columns(project_dir) -> list[str] | None:
    info = store.read_json(store.project_path(project_dir, "project.json")) or {}
    train = info.get("train_file")
    if not isinstance(train, str) or "/" in train or train.startswith("."):
        return None
    header = store.csv_header(Path(project_dir) / "data" / train)
    return None if header is None else [c for c in header if c != info.get("target")]


def bundle_view(src, run_id: str) -> dict:
    src.registry.get_cycle(run_id)
    model = effective_model(src, run_id)
    if model is None:
        return {"available": False, "reason": "no persisted bundle yet: the final report persists it",
                "run_id": run_id}
    meta = _meta(src, model["bundle_id"])
    if meta is None:
        return {"available": False, "reason": "the persisted bundle file is not readable", "run_id": run_id,
                "bundle_id": model["bundle_id"]}
    model_files = meta.get("model_files") or {}
    files = [{"name": META_NAME, "sha256": None}] + [
        {"name": f"model/{name}", "sha256": sha} for name, sha in sorted(model_files.items())]
    if store.exists(store.out_file(src.project_dir, "report", run_id)):
        files.append({"name": REPORT_NAME, "sha256": None})
    for item in files:
        if item["sha256"] is None:
            blob = _file_bytes(src, run_id, model["bundle_id"], item["name"], meta)
            item["sha256"] = None if blob is None else store.sha256_bytes(blob)
    return {
        "available": True, "run_id": run_id, "bundle_id": model["bundle_id"], "attempt_id": model["attempt_id"],
        "chosen_by": model["source"],
        "hashes": {"folder": meta.get("folder_hash"), "model": meta.get("model_hash"),
                   "protocol": meta.get("protocol_hash"), "data": meta.get("data_hash")},
        "code": {"files": len(meta.get("code_files") or {}), "note": "the code is never shown or served"},
        "library_versions": library_versions(src, model["attempt_id"]),
        "files": files, "metric": meta.get("metric"), "validation_mean": meta.get("validation_mean"),
        "reproduce": store.latest_job(src.project_dir, ("reproduce",)),
        "predict": {"columns": training_columns(src.project_dir)},
    }


def _file_bytes(src, run_id: str, bundle_id: str, name: str, meta: dict) -> bytes | None:
    project = src.project_dir
    if name == META_NAME:
        return store.read_capped(store.out_file(project, "persisted", bundle_id))
    if name == REPORT_NAME:
        return store.read_capped(store.out_file(project, "report", run_id))
    if name.startswith("model/"):
        rel = name[len("model/"):]
        if rel not in (meta.get("model_files") or {}):
            return None
        try:
            return store.read_capped(store.out_file(project, "model", bundle_id, rel))
        except ValueError:
            return None
    return None  # anything else, the code folder included, is not served


def download(src, run_id: str, name: str) -> tuple[bytes, str]:
    """(bytes, sha256) of an allowed file; ``NotFoundError`` for code, unknown names and traversal."""
    model = effective_model(src, run_id)
    meta = _meta(src, model["bundle_id"]) if model else None
    if model is None or meta is None:
        raise NotFoundError("there is no persisted bundle to download from")
    blob = _file_bytes(src, run_id, model["bundle_id"], name, meta)
    if blob is None:
        raise NotFoundError(f"{name!r} is not a file you can download")
    return blob, store.sha256_bytes(blob)


def report_markdown(src, run_id: str) -> str | None:
    return store.read_text_file(store.out_file(src.project_dir, "report", run_id))
