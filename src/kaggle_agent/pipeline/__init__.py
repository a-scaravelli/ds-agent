"""Final report, holdout evaluation and end-to-end pipeline (design sections 6, 8 and 15, step 11, level 1).

After a cycle ends this package evaluates the chosen bundle once on the holdout (a failed run is a
recorded outcome, not an exception), persists the trained model inside a copy of the bundle, links
that copy to the chosen attempt with a ``pipeline:<ref>`` artifact, derives suggestions for the next
cycle by fixed rules, and stores a final report. ``python -m kaggle_agent.pipeline reproduce`` shows
that the stored code, protocol and data give the reported frozen-protocol metric, and
``python -m kaggle_agent.pipeline predict`` predicts from the stored model in a separate process,
writing JSON only (never pickle).

Not here: the orchestrator that starts the next cycle by itself and the first real competition
(step 12), a real ``BundleWriter``, real lead input, levels 2 and 3. The ``loop`` and ``web``
packages are used, not changed.
"""

from .errors import PersistedBundleError, PipelineError
from .flow import (
    PipelineContext,
    PipelineResult,
    link_pipeline,
    make_report_hook,
    run_pipeline,
    start_cycle,
)
from .holdout import INTERRUPTED, NO_USABLE_ATTEMPT, evaluate_chosen_holdout
from .persisted import (
    ModelStore,
    PersistedBundle,
    load_persisted,
    persist_bundle,
    persisted_root,
    runner_model_store_factory,
)
from .report import FinalReport, build_report, render_text, report_path, store_report
from .suggestions import derive_suggestions, store_suggestions, suggestion_id

__all__ = [
    "FinalReport",
    "INTERRUPTED",
    "ModelStore",
    "NO_USABLE_ATTEMPT",
    "PersistedBundle",
    "PersistedBundleError",
    "PipelineContext",
    "PipelineError",
    "PipelineResult",
    "build_report",
    "derive_suggestions",
    "evaluate_chosen_holdout",
    "link_pipeline",
    "load_persisted",
    "make_report_hook",
    "persist_bundle",
    "persisted_root",
    "render_text",
    "report_path",
    "run_pipeline",
    "runner_model_store_factory",
    "start_cycle",
    "store_report",
    "store_suggestions",
    "suggestion_id",
]
