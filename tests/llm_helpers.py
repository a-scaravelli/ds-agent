"""Helpers shared by the llm tests: fake clock, canned CLI envelopes, fake runner, secrets."""

import json
import subprocess
from dataclasses import dataclass, field

from kaggle_agent.llm import ToolPermissions
from kaggle_agent.llm.fake import valid_attempt_json, valid_plan_json

SECRET_KEY = "zq81-vk3w-p0xj-mm47"
SECRET_TOKEN = "tkn7-hd92-rwlb-c5ne"


class Clock:
    """Float clock whose ``sleep`` advances time without waiting."""

    def __init__(self) -> None:
        self.time = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.time += seconds


def permissions(tmp_path, *tools) -> ToolPermissions:
    return ToolPermissions(working_dir=tmp_path, allowed_tools=tools)


def envelope(result="{}", is_error=False, usage=None, model_usage=None, **extra) -> str:
    """A canned ``claude -p --output-format json`` envelope (shape in design section 8)."""
    data = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "duration_ms": 1200,
        "result": result,
        "session_id": "11111111-2222-3333-4444-555555555555",
        "total_cost_usd": 0.01,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2,
            "output_tokens": 5,
        },
        "modelUsage": model_usage if model_usage is not None else {},
        **extra,
    }
    return json.dumps(data)


def completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@dataclass
class RunnerCall:
    argv: list
    input: str
    cwd: str
    env: dict
    timeout: float


@dataclass
class FakeRunner:
    """Replaces ``subprocess.run``: plays canned results or raises canned exceptions."""

    steps: list
    clock: Clock | None = None
    duration: float = 0.0
    calls: list = field(default_factory=list)

    def __call__(self, argv, **kwargs):
        self.calls.append(
            RunnerCall(list(argv), kwargs.get("input"), kwargs.get("cwd"),
                       dict(kwargs.get("env") or {}), kwargs.get("timeout"))
        )
        if self.clock is not None:
            self.clock.time += self.duration
        step = self.steps[len(self.calls) - 1]
        if isinstance(step, BaseException):
            raise step
        return step


def ok_step(result, **kwargs):
    return completed(envelope(result=result, **kwargs))


PLAN_JSON = valid_plan_json()
ATTEMPT_JSON = valid_attempt_json()
