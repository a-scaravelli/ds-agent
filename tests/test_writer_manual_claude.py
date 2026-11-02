"""MANUAL test: runs the real Claude CLI to write a bundle for a toy dataset.

Skipped unless the environment variable KAGGLE_AGENT_RUN_CLAUDE is set to 1. It needs the ``claude``
CLI logged in with a subscription, and it spends quota. The pytest suite never runs it.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest
from bundle_helpers import make_runner
from evaluation_helpers import frozen_engine
from writer_helpers import CYCLE, make_request, new_registry, task

from kaggle_agent.agents import ColumnSummary, DataSummary
from kaggle_agent.bundle import load_bundle, score_bundle
from kaggle_agent.llm.real import RealClaudeClient
from kaggle_agent.writer import ClaudeBundleWriter

ENABLED = os.environ.get("KAGGLE_AGENT_RUN_CLAUDE") == "1"


def toy_frame(n=90, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": (x1 + rng.normal(scale=0.5, size=n) > 0).astype(int)})


@pytest.mark.skipif(not ENABLED, reason="MANUAL test: set KAGGLE_AGENT_RUN_CLAUDE=1 to run the real Claude")
def test_manual_real_claude_writes_a_bundle_for_a_toy_dataset(tmp_path):
    """MANUAL: only the schema of the toy data goes to the writer; the bundle is then scored."""
    frame = toy_frame()
    summary = DataSummary(len(frame), "y", tuple(
        ColumnSummary(c, str(frame[c].dtype), 0.0, int(frame[c].nunique())) for c in frame.columns))
    client = RealClaudeClient(runner=subprocess.run, timeout_seconds=600)
    base = tmp_path.resolve()
    writer = ClaudeBundleWriter(
        client, new_registry(), CYCLE, base / "workspace",
        task(summary=summary), max_repairs=2,
    )
    request = make_request(base / "bundles" / "B1", "B1")
    folder = writer.write(request)
    assert load_bundle(folder).bundle_id == "B1"
    engine, _ = frozen_engine()
    _, runner = make_runner(folder)
    outcome = score_bundle(engine, runner, frame, attempt_id="A1", execution_id="A1-E1",
                           config={}, data_hash="toy")
    print(outcome.execution.outcome, outcome.execution.reason)
    assert outcome.execution.outcome in ("succeeded", "failed")
