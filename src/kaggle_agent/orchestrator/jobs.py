"""Heavy computations of the cockpit as small subcommands: ``validate-config``, ``preview``, ``profile``.

Each writes ONE JSON result to the file given by ``--out`` (atomically) and exits 0 when it could
compute something (``ok: false`` is a result, not a crash). The web never imports this package: it
starts these commands as background jobs through its controller and reads the JSON file.

``profile`` works on the DEVELOPMENT rows only (``FrozenSplit.development_data``); the holdout is
never read. ``preview`` freezes nothing: it only computes what ``start`` would freeze.
"""

import json
import math
from pathlib import Path

import pandas as pd

from kaggle_agent.evaluation import EvaluationEngine, FrozenSplit, Protocol

from .build import Dependencies, resolve_project
from .config import Config, load_config
from .errors import ConfigError, OrchestratorError
from .project import check_config, read_project, write_atomic

MAX_COLUMNS = 200


def _split_error(message: str) -> dict:
    key, sep, rest = message.partition(": ")
    if sep and " " not in key and len(key) < 60:
        return {"key": key, "message": rest}
    return {"key": "", "message": message}


def _resolved(config: Config) -> dict:
    comp, limits, protocol = config.competition, config.limits, config.protocol
    return {
        "competition": {"slug": comp.slug, "target": comp.target, "metric": comp.metric,
                        "direction": comp.direction, "train_file": comp.train_file,
                        "test_file": comp.test_file},
        "limits": {"rounds": limits.rounds, "candidates": limits.candidates,
                   "stagnation": limits.stagnation_rounds, "pause_every": limits.pause_every,
                   "fix_retries": limits.fix_retries},
        "protocol": protocol.to_dict(),
        "models": dict(config.models),
        "allowed_libraries": list(config.runtime.allowed_dependencies),
        "folder": str(config.paths.folder),
    }


def validate_config(config_path: str) -> dict:
    """Strict validation through ``load_config`` (secrets refused) and the frozen-field check."""
    try:
        config = load_config(config_path)
        state = read_project(config.paths)
        if state is not None:
            check_config(state, config)
    except OrchestratorError as error:
        return {"ok": False, "errors": [_split_error(str(error))], "resolved": None}
    return {"ok": True, "errors": [], "resolved": _resolved(config)}


def bar_text(k: float, partitions: int) -> str:
    return (f"An improvement counts as real only if the mean paired difference is larger than {k:g} x the "
            f"baseline noise (the spread of the baseline score over its {partitions} validation "
            "partitions). The number of the bar is known once the baseline has run.")


def _frame_and_protocol(config: Config, deps: Dependencies):
    state = read_project(config.paths)
    if state is None:
        if deps is None:
            from .build import real_dependencies
            deps = real_dependencies()
        state = resolve_project(config, deps)
    protocol = Protocol.from_dict(state.protocol)
    frame = pd.read_csv(config.paths.data / state.train_file)
    return state, protocol, frame


def preview(config_path: str, deps: Dependencies) -> dict:
    config = load_config(config_path)
    state, protocol, frame = _frame_and_protocol(config, deps)
    split = FrozenSplit.create(frame, protocol)
    parts = split.partitions()
    engine = EvaluationEngine(None, state.cycle_id, protocol)
    development = len(split.development_rows())
    return {
        "ok": True, "competition": state.competition, "target": state.target, "metric": state.metric,
        "direction": state.direction, "protocol": protocol.to_dict(), "protocol_hash": engine.current_hash(),
        "rows": int(len(frame)), "development_rows": development, "holdout_rows": int(len(frame)) - development,
        "holdout_share": protocol.holdout_share, "partitions": len(parts),
        "folds": [{"id": p.id, "train": int(len(p.train_rows)), "valid": int(len(p.valid_rows))} for p in parts],
        "threshold_k": protocol.threshold_k, "bar": None, "bar_text": bar_text(protocol.threshold_k, len(parts)),
        "already_started": read_project(config.paths) is not None,
    }


def _number(value) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


def profile(config_path: str, deps: Dependencies) -> dict:
    config = load_config(config_path)
    state, protocol, frame = _frame_and_protocol(config, deps)
    features, target = FrozenSplit.create(frame, protocol).development_data()
    dev = features.assign(**{protocol.target: target})
    rows = len(dev)
    columns, distributions, notes = [], {}, {"constant": [], "id_like": []}
    for name in list(dev.columns)[:MAX_COLUMNS]:
        col = dev[name]
        distinct = int(col.nunique(dropna=True))
        columns.append({"name": str(name), "dtype": str(col.dtype), "missing": int(col.isna().sum()),
                        "missing_fraction": _number(col.isna().mean()) if rows else 0.0, "distinct": distinct})
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            d = col.describe()
            distributions[str(name)] = {"kind": "numeric", **{k: _number(d.get(k)) for k in
                                        ("min", "25%", "50%", "75%", "max", "mean", "std")}}
        else:
            top = col.astype(str).value_counts().head(5)
            distributions[str(name)] = {"kind": "categorical", "top": {str(k): int(v) for k, v in top.items()}}
        if distinct <= 1:
            notes["constant"].append(str(name))
        if name != protocol.target and rows > 1 and distinct == rows and (
                pd.api.types.is_integer_dtype(col) or col.dtype == object or str(col.dtype).startswith("str")):
            notes["id_like"].append(str(name))
    counts = target.value_counts(dropna=False)
    balance = ({str(k): int(v) for k, v in counts.items()} if len(counts) <= 20 else
               {"kind": "continuous", **{k: _number(v) for k, v in target.describe().items()}})
    return {"ok": True, "rows": int(rows), "columns": columns, "distributions": distributions,
            "target": protocol.target, "target_balance": balance,
            "duplicate_rows": int(dev.duplicated().sum()), "constant_columns": notes["constant"],
            "id_like_columns": notes["id_like"], "part": "development"}


def run_job(name: str, config_path: str, out: str, deps: Dependencies | None) -> int:
    """Compute one job and write its JSON result to ``out``; 0 when a result was written."""
    try:
        if name == "validate-config":
            result = validate_config(config_path)
        else:
            result = (preview if name == "preview" else profile)(config_path, deps)
        code = 0
    except Exception as error:
        message = str(error) if isinstance(error, OrchestratorError) else f"{type(error).__name__}: {error}"
        result, code = {"ok": False, "error": " ".join(message.split())[:500]}, 1
    write_atomic(Path(out), json.dumps(result, ensure_ascii=False))
    return code
