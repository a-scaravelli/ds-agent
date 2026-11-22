"""Recording of the performance analytics by WRAPPING the runner (ticket 29); nothing in ``evaluation/`` changes.

Wrapper order in ``build_context``: ``BundleRunner`` -> ``RecordingCandidate`` -> ``EventingCandidate`` (the last
one only for a supervised run). ``RecordingCandidate.fit_predict`` repeats what ``BundleRunner.fit_predict`` does
(``fit_model`` into a temporary model folder, ``predict_with_model``, delete the folder) so it can read the
optional ``describe.json`` of that folder and keep the validation predictions it returns unchanged.

Files, all under ``<project>/analytics/`` and written ONLY by the orchestrator process:

====================================  =======================================================================
``oof/<bundle>-<sha12>.csv.gz``       out-of-fold predictions (row, rep, fold, truth, prediction or p0..pk-1)
``describe/<bundle>-<sha12>.json``    the validated, normalised copy of the bundle's describe.json
``development-<sha12>.csv.gz``        once per project: the DEVELOPMENT features (row id + features, no target)
``importance/<bundle>.json``          written by the importance job (single writer of that file)
``index.jsonl``                       one line per recorded file: the commit point (v=1)
====================================  =======================================================================

Order: content to a temp name, ``fsync``, ``os.replace`` to its content-addressed name, then ONE index line
(``append_jsonl``: flock, one ``os.write``). A reader only opens a file named by an index line, so it never sees a
partial file. Truth and row ids come from ``FrozenSplit.data``, which refuses any non-development row: the
holdout is never an input. A recorder problem is caught, reported on stderr and never changes a prediction or fails
the attempt; ``BundleRunError`` of the inner runner always propagates.
"""

import hashlib
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_agent.analytics.describe import INVALID, MISSING, validate_describe
from kaggle_agent.analytics.oof_format import MAX_COMPRESSED, MAX_ROWS, OofError, encode_oof
from kaggle_agent.evaluation import FrozenSplit, get_metric
from kaggle_agent.proposals import DESCRIBE_FILE
from kaggle_agent.proposals.describe import MAX_DESCRIBE_BYTES

from .cockpit import append_jsonl

ANALYTICS_DIR = "analytics"
INDEX_FILE = "index.jsonl"
VERSION = 1
MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024
MAX_SNAPSHOT_COLUMNS = 500
MAX_FOLDER_BYTES = 1024 ** 3
MAX_CLASSES_LISTED = 50


def analytics_dir(project_folder) -> Path:
    return Path(project_folder) / ANALYTICS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _note(message: str) -> None:
    print(f"analytics: {message}", file=sys.stderr)


def folder_size(folder: Path) -> int:
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def write_content(folder: Path, relative: str, blob: bytes) -> None:
    """Temp file, fsync, ``os.replace`` to the final name (the parent folders are created here)."""
    target = folder / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _read_index(folder: Path) -> list[dict]:
    from .cockpit import read_jsonl
    return read_jsonl(folder / INDEX_FILE)[0]


class AnalyticsWriter:
    """Writes the content-addressed files and their index lines of one project."""

    def __init__(self, project_folder) -> None:
        self.folder = analytics_dir(project_folder)

    def line(self, **record) -> None:
        append_jsonl(self.folder / INDEX_FILE, {"v": VERSION, "at": _now(), **record}, sync=True)

    def put(self, kind: str, stem: str, suffix: str, blob: bytes, **extra) -> dict:
        digest = hashlib.sha256(blob).hexdigest()
        relative = f"{kind}/{stem}-{digest[:12]}{suffix}" if kind != "development" else f"{stem}-{digest[:12]}{suffix}"
        write_content(self.folder, relative, blob)
        record = {"kind": kind, "file": relative, "sha256": digest, "bytes": len(blob), "status": "ok", **extra}
        self.line(**record)
        return record

    def over_budget(self) -> bool:
        return folder_size(self.folder) >= MAX_FOLDER_BYTES

    def has_development(self) -> bool:
        return any(r.get("kind") == "development" and r.get("status") == "ok" for r in _read_index(self.folder))


def read_describe(model_dir: Path) -> bytes | None | str:
    """The bytes of describe.json in the temporary model folder, None if absent, or an error text.

    A deliberate, size-capped exception to 'the parent never opens model files': nothing is unpickled, only the
    bytes of one JSON file are read and validated.
    """
    path = Path(model_dir) / DESCRIBE_FILE
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return "the file cannot be read"
    if stat.S_ISLNK(info.st_mode):
        return "the file is a symbolic link"
    if not stat.S_ISREG(info.st_mode):
        return "the file is not a regular file"
    if info.st_size > MAX_DESCRIBE_BYTES:
        return f"the file is larger than {MAX_DESCRIBE_BYTES // 1024} KB"
    try:
        with open(path, "rb") as stream:
            return stream.read(MAX_DESCRIBE_BYTES + 1)
    except OSError:
        return "the file cannot be read"


class RecordingCandidate:
    """A ``Candidate`` that records OOF predictions, per-partition scores and describe.json of one bundle."""

    def __init__(self, inner, frame: pd.DataFrame, protocol, bundle_id: str, project_folder, split=None) -> None:
        self._inner = inner
        self._frame = frame
        self._protocol = protocol
        self._bundle_id = bundle_id
        self._writer = AnalyticsWriter(project_folder)
        self._split = split
        self._parts = None
        self._seen: dict[str, np.ndarray] = {}
        self._describe: bytes | None | str = None
        self._broken = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # -- the wrapped call -------------------------------------------------------------------------

    def fit_predict(self, x_train, y_train, x_valid, seed):
        inner = self._inner
        if not (hasattr(inner, "fit_model") and hasattr(inner, "predict_with_model")):
            predictions = inner.fit_predict(x_train, y_train, x_valid, seed)
        else:
            model_dir = Path(tempfile.mkdtemp(prefix="bundle-model-")).resolve()
            try:
                inner.fit_model(x_train, y_train, seed, model_dir)
                self._safe(self._take_describe, model_dir)
                predictions = inner.predict_with_model(model_dir, x_valid, seed)
            finally:
                shutil.rmtree(model_dir, ignore_errors=True)
        self._safe(self._keep, x_valid, predictions)
        return predictions

    def _safe(self, call, *args) -> None:
        try:
            call(*args)
        except BaseException as error:  # the recorder must never change a run (not even by a bug)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            self._broken = True
            _note(f"{self._bundle_id}: not recorded ({type(error).__name__}: {' '.join(str(error).split())[:200]})")

    def _take_describe(self, model_dir: Path) -> None:
        if self._describe is None:
            self._describe = read_describe(model_dir)

    def _partitions(self):
        if self._split is None:
            self._split = FrozenSplit.create(self._frame, self._protocol)
        if self._parts is None:
            self._parts = self._split.partitions()
        return self._parts

    def _keep(self, x_valid, predictions) -> None:
        if self._broken:
            return
        index = [int(i) for i in x_valid.index]
        part = next((p for p in self._partitions() if [int(i) for i in p.valid_rows] == index), None)
        if part is None:
            raise OofError("the validation rows match no partition")
        self._seen[part.id] = np.asarray(predictions)
        if len(self._seen) == len(self._partitions()):
            self._write_all()

    # -- writing ----------------------------------------------------------------------------------

    def _write_all(self) -> None:
        parts = {p.id: p for p in self._partitions()}
        rows, reps, folds, truths, preds, scores = [], [], [], [], [], {}
        metric = get_metric(self._protocol.metric)
        for pid, part in parts.items():
            _, y_valid = self._split.data(part.valid_rows)  # refuses any non-development row
            values = self._seen[pid]
            scores[pid] = float(metric.score(y_valid, values, self._protocol.metric_params))
            rows.append(np.asarray(part.valid_rows, dtype=np.int64))
            reps.append(np.full(len(part.valid_rows), part.repetition))
            folds.append(np.full(len(part.valid_rows), part.fold))
            truths.append(np.asarray(y_valid))
            preds.append(values)
        row = np.concatenate(rows)
        base = {"bundle_id": self._bundle_id, "partitions": len(parts)}
        if self._writer.over_budget():
            self._writer.line(kind="oof", status="budget_exhausted", **base)
            return
        if len(row) > MAX_ROWS:
            self._writer.line(kind="oof", status="too_large", rows=int(len(row)), **base)
        else:
            blob = encode_oof(row, np.concatenate(reps), np.concatenate(folds), np.concatenate(truths),
                              np.concatenate(preds))
            if len(blob) > MAX_COMPRESSED:
                self._writer.line(kind="oof", status="too_large", rows=int(len(row)), bytes=len(blob), **base)
            else:
                truth_all = np.concatenate(truths)
                classes = [] if self._protocol_kind() == "values" else [
                    c.item() if hasattr(c, "item") else c for c in sorted(set(truth_all.tolist()))]
                self._writer.put("oof", self._bundle_id, ".csv.gz", blob, bundle_id=self._bundle_id,
                                 rows=int(len(row)), partitions=len(parts), fold_scores=scores,
                                 classes=classes[:MAX_CLASSES_LISTED], input_kind=self._protocol_kind())
        self._write_describe()
        self._write_snapshot()

    def _protocol_kind(self) -> str:
        return get_metric(self._protocol.metric).input_kind

    def _write_describe(self) -> None:
        raw = self._describe
        if isinstance(raw, str):
            self._writer.line(kind="describe", bundle_id=self._bundle_id, status=INVALID, reason=raw)
            return
        result = validate_describe(raw)
        if result.status == MISSING:
            self._writer.line(kind="describe", bundle_id=self._bundle_id, status=MISSING)
        elif result.status == INVALID:
            self._writer.line(kind="describe", bundle_id=self._bundle_id, status=INVALID, reason=result.reason)
        else:
            import json
            blob = json.dumps(result.content, sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8")
            self._writer.put("describe", self._bundle_id, ".json", blob, bundle_id=self._bundle_id,
                             warnings=list(result.warnings)[:10])

    def _write_snapshot(self) -> None:
        if self._writer.has_development():
            return
        features, _ = self._split.development_data()
        if features.shape[1] > MAX_SNAPSHOT_COLUMNS:
            self._writer.line(kind="development", status="too_large", columns=int(features.shape[1]))
            return
        import gzip
        import io
        text = features.rename_axis("row").reset_index().to_csv(index=False, lineterminator="\n")
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as stream:
            stream.write(text.encode("utf-8"))
        blob = buffer.getvalue()
        if len(blob) > MAX_SNAPSHOT_BYTES:
            self._writer.line(kind="development", status="too_large", bytes=len(blob))
            return
        self._writer.put("development", "development", ".csv.gz", blob, rows=int(len(features)),
                         columns=[str(c) for c in features.columns][:MAX_SNAPSHOT_COLUMNS])
