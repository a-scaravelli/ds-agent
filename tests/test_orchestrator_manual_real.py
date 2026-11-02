"""MANUAL tests: a real run of the orchestrator on a small real competition (titanic).

Skipped unless the environment variable KAGGLE_AGENT_RUN_REAL is set to 1. They need Kaggle
credentials, the rules of the competition accepted on the Kaggle site, and the ``claude`` CLI logged
in with a subscription; they spend Claude quota and are never run by the pytest suite. A real
confirmed submission is NOT automated: it is a manual step described in the README.
"""

import os

import pytest

from kaggle_agent.orchestrator.__main__ import main

ENABLED = os.environ.get("KAGGLE_AGENT_RUN_REAL") == "1"
REASON = "MANUAL test: set KAGGLE_AGENT_RUN_REAL=1 to run against the real Kaggle and Claude"

CONFIG = """\
[competition]
slug = "titanic"
target = "Survived"
metric = "accuracy"

[project]
folder = "project"

[limits]
rounds = 1
candidates = 1
stagnation = 1
pause_every = 1
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.mark.skipif(not ENABLED, reason=REASON)
def test_manual_real_start_stops_waiting_for_the_lead(config, capsys):
    """MANUAL: start downloads the data, plans, and waits for the plan approval (exit code 10)."""
    assert main(["start", "--config", str(config)]) == 10
    out = capsys.readouterr().out
    assert "WAITING FOR THE LEAD" in out and "[plan]" in out
    assert main(["status", "--config", str(config)]) == 0


@pytest.mark.skipif(not ENABLED, reason=REASON)
def test_manual_real_dry_run_submit_after_the_lead_accepted(config, capsys):
    """MANUAL: approve, continue and accept by hand-written lead file, then a dry run (nothing is sent)."""
    project = config.parent / "project"
    assert main(["start", "--config", str(config)]) == 10
    steps = ['[plan]\ndecision = "approve"\n', '\n[guidance]\nround = 1\naction = "continue"\n',
             '\n[review]\ndecision = "accept"\n']
    text = ""
    for step in steps:
        text += step
        (project / "lead.toml").write_text(text, encoding="utf-8")
        code = main(["resume", "--config", str(config)])
        assert code in (0, 10)
    assert code == 0
    capsys.readouterr()
    assert main(["submit", "--config", str(config)]) == 0
    assert "DRY RUN" in capsys.readouterr().out
