"""Helpers of the orchestrator tests: a tiny synthetic competition and fakes for every external piece.

Nothing here touches the network, a real Claude or a real Kaggle. The default sandbox runs the bundle
child in this process (fast); ``real_sandbox=True`` uses the real ``SubprocessSandbox``.
"""

import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline_helpers import MAJORITY_MODEL

from kaggle_agent.bundle import child
from kaggle_agent.kaggle import FakeCompetition, FakeKaggleClient
from kaggle_agent.llm import LlmClient, RawResponse, Usage
from kaggle_agent.llm.interface import BackoffPolicy
from kaggle_agent.orchestrator import Dependencies
from kaggle_agent.orchestrator.__main__ import main
from kaggle_agent.sandbox import CRASHED, SUCCEEDED, Sandbox, SandboxResult, SubprocessSandbox

SLUG = "toy"
START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

COORDINATOR = "You are the coordinator"
DATA = "You are the data analyst"
FEATURES = "You are the feature engineering specialist"
MODELING = "You are the modeling specialist"
REVIEWER = "You are the reviewer"
WRITER = "You are the bundle writer"


class TickClock:
    """A clock that moves only when ``sleep`` is called or ``advance`` is."""

    def __init__(self) -> None:
        self.time = START

    def now(self) -> datetime:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.time += timedelta(seconds=seconds)

    def advance(self, **kwargs) -> None:
        self.time += timedelta(**kwargs)


def train_csv(rows: int = 60) -> str:
    lines = ["id,x1,x2,y"]
    for i in range(1, rows + 1):
        lines.append(f"{i},{i % 7},{i % 3},{1 if i % 3 == 0 else 0}")
    return "\n".join(lines) + "\n"


def features_csv(rows: int = 10) -> str:
    return "id,x1,x2\n" + "\n".join(f"{100 + i},{i % 7},{i % 3}" for i in range(rows)) + "\n"


def sample_csv(rows: int = 10, columns: str = "id,y") -> str:
    width = len(columns.split(","))
    return columns + "\n" + "\n".join(",".join([str(100 + i)] + ["0"] * (width - 1)) for i in range(rows)) + "\n"


def competition(score_script=(None, 0.8), daily_limit=5, evaluation="Submissions are evaluated on accuracy.",
                sample=None, test=None, train=None) -> FakeCompetition:
    return FakeCompetition(
        files={"train.csv": (train or train_csv()).encode(), "test.csv": (test or features_csv()).encode(),
               "sample_submission.csv": (sample or sample_csv()).encode()},
        description="A toy competition", rules="Be nice", evaluation=evaluation,
        sample_submission=sample or sample_csv(), daily_limit=daily_limit, score_script=score_script,
    )


class InProcessSandbox(Sandbox):
    """Runs ``bundle/child.py`` in this process: same code, no child process. Scripted failures allowed."""

    def __init__(self, fail_when=None) -> None:
        self.fail_when = fail_when
        self.runs: list[str] = []

    def run(self, policy, request):
        staging, stage = Path(request.args[0]), request.args[1]
        self.runs.append(stage)
        if self.fail_when is not None and self.fail_when(stage):
            return SandboxResult(CRASHED, exit_code=1, reason="child failed (scripted)")
        saved_path, saved_env = list(sys.path), os.environ.get("BUNDLE_MODEL_DIR")
        saved_modules = set(sys.modules)
        try:
            child.main(["child.py", str(staging), stage])
        except BaseException as error:  # the sandbox reports a crash as a result
            return SandboxResult(CRASHED, exit_code=1, reason=f"child exited with 1: {type(error).__name__}: {error}")
        finally:
            sys.path[:] = saved_path
            for name in set(sys.modules) - saved_modules:
                sys.modules.pop(name, None)
            if saved_env is None:
                os.environ.pop("BUNDLE_MODEL_DIR", None)
            else:
                os.environ["BUNDLE_MODEL_DIR"] = saved_env
        return SandboxResult(SUCCEEDED, exit_code=0)


class RoleAwareClaude(LlmClient):
    """A fake Claude that answers every role and the bundle writer according to the prompt.

    ``errors`` maps a 1-based call number to an exception raised by that call (once). ``log`` holds
    (call number, kind) of every invocation, ``prompts`` the prompts.
    """

    def __init__(self, errors=None, **kwargs) -> None:
        kwargs.setdefault("backoff", BackoffPolicy(max_retries=0))
        kwargs.setdefault("sleep", lambda seconds: None)
        super().__init__(**kwargs)
        self.errors = dict(errors or {})
        self.fail_on: dict[str, Exception] = {}  # kind -> exception raised by the next call of that kind
        self.prompts: list[str] = []
        self.kinds: list[str] = []
        self.models: list[str | None] = []
        self.counter = 0

    def _run(self, prompt, permissions, model):
        self.prompts.append(prompt)
        self.models.append(model)
        number = len(self.prompts)
        kind = self._kind(prompt)
        self.kinds.append(kind)
        error = self.errors.pop(number, None) or self.fail_on.pop(kind, None)
        if error is not None:
            raise error
        self.counter += 1
        n = self.counter
        if kind == "writer":
            bundle_id = re.search(r"Bundle id: (\S+)", prompt).group(1)
            folder = Path(permissions.working_dir) / "bundle"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "bundle.json").write_text(json.dumps({
                "bundle_id": bundle_id, "dependencies": [],
                "entry_points": {"fit": "model:fit", "predict": "model:predict"}}), encoding="utf-8")
            (folder / "model.py").write_text(f"# call {n}\n{MAJORITY_MODEL}", encoding="utf-8")
            text = '{"summary": "wrote a majority model"}'
        elif kind == "coordinator":
            text = json.dumps({"plan": {
                "metric": "accuracy", "validation": "3-fold", "feature_engineering": ["none"],
                "models": ["majority"], "baseline": "majority class"},
                "tasks": [f"Try idea {n}"], "goal": None})
        elif kind == "data":
            text = json.dumps({"profile": "a small table", "quality_issues": [], "leakage_hints": [],
                               "proposed_cleaning": []})
        elif kind in ("feature_engineering", "modeling"):
            text = json.dumps({"id": "x", "phase": "iteration", "hypothesis": f"hypothesis {n}",
                               "rationale": "why", "what_changed": f"change {n}", "measure": "accuracy",
                               "exploratory": True})
        else:
            text = json.dumps({"text": f"round summary {n}"})
        return RawResponse(text=text, usage=Usage(10, 5), model="fake-model")

    @staticmethod
    def _kind(prompt: str) -> str:
        for kind, marker in (("writer", WRITER), ("coordinator", COORDINATOR), ("data", DATA),
                             ("feature_engineering", FEATURES), ("modeling", MODELING), ("reviewer", REVIEWER)):
            if prompt.startswith(marker):
                return kind
        raise AssertionError(f"unknown prompt: {prompt[:60]!r}")


CONFIG = """\
[competition]
slug = "toy"
target = "from_submission_spec"
metric = "from_competition_info"

[project]
folder = "project"

[limits]
rounds = 2
candidates = 1
stagnation = 2
pause_every = 1
fix_retries = 1

[protocol]
validation_params = {{ n_folds = 3 }}

[runtime]
wait_seconds = 60
poll_interval = 5
{extra}
"""


class World:
    """One project in ``tmp_path`` with every external piece faked."""

    def __init__(self, tmp_path, comp=None, errors=None, real_sandbox=False, extra="", sandbox=None,
                 llm=None) -> None:
        self.base = Path(tmp_path).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.clock = TickClock()
        self.kaggle = FakeKaggleClient({SLUG: comp or competition()}, clock=self.clock.now, sleep=self.clock.sleep)
        self.claude = llm or RoleAwareClaude(errors)
        self.sandbox = sandbox or (SubprocessSandbox() if real_sandbox else InProcessSandbox())
        self.llm_factory_calls = 0
        self.deps = Dependencies(kaggle=self.kaggle, llm=self._llm, sandbox=self.sandbox,
                                 clock=self.clock.now, sleep=self.clock.sleep)
        self.config = self.base / "config.toml"
        self.config.write_text(CONFIG.format(extra=extra), encoding="utf-8")
        self.project = self.base / "project"
        self.lead_file = self.project / "lead.toml"

    def _llm(self):
        self.llm_factory_calls += 1
        return self.claude

    def lead(self, text: str) -> None:
        self.project.mkdir(parents=True, exist_ok=True)
        self.lead_file.write_text(text, encoding="utf-8")

    def run(self, *argv, deps="default"):
        """Run the command line; returns (exit code, stdout + stderr text)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([*argv, "--config", str(self.config)] if argv[0] != "submit" else
                        [argv[0], "--config", str(self.config), *argv[1:]],
                        deps=self.deps if deps == "default" else deps)
        return code, out.getvalue() + err.getvalue()

    def registry(self):
        from kaggle_agent.registry import SqliteRegistry
        return SqliteRegistry(str(self.project / "registry.db"), read_only=True)

    def state(self) -> str:
        registry = self.registry()
        try:
            return registry.list_cycles()[0].state
        finally:
            registry.close()


PLAN_APPROVE = '[plan]\ndecision = "approve"\n'
GUIDE_CONTINUE = '[guidance]\nround = {n}\naction = "continue"\n'
REVIEW_ACCEPT = '[review]\ndecision = "accept"\n'


def drive_to_review(world: World) -> None:
    """start -> approve -> continue after each round -> waiting for the review."""
    assert world.run("start")[0] == 10
    world.lead(PLAN_APPROVE)
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=1))
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2))
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_REVIEW", (code, world.state())


def drive_to_stopped(world: World) -> None:
    drive_to_review(world)
    world.lead(PLAN_APPROVE + GUIDE_CONTINUE.format(n=2) + REVIEW_ACCEPT)
    code, text = world.run("resume")
    assert code == 0 and world.state() == "STOPPED", text
