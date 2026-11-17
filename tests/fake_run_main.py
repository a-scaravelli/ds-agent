"""The orchestrator command line over the FAKES of the tests (no network, no real Claude, no real Kaggle).

Started only by the run controller of the tests, through the injected ``orchestrator_argv``
``(sys.executable, <this file>, "--scenario", <json file>)``; it removes ``--scenario FILE`` from the
arguments and calls ``kaggle_agent.orchestrator.__main__.main(argv, deps=<fakes>)``: the real orchestrator
code path. The production code has no switch that selects fakes.

Scenario keys (all optional): ``block_file`` (every Claude call waits while this file exists),
``delay`` (seconds added to each Claude call), ``ignore_sigterm`` (SIGTERM never stops the process: the
SIGKILL path), ``real_sandbox`` (bundles run in real child processes), ``slow_bundle`` (seconds the bundle's
``fit`` sleeps, with ``bundle_pid_file`` receiving the child's pid), ``heartbeat`` (interval seconds),
``errors`` (call number -> "quota" | "login"), ``slug``.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator_helpers import (  # noqa: E402
    SLUG,
    InProcessSandbox,
    RoleAwareClaude,
    TickClock,
    competition,
)
from pipeline_helpers import MAJORITY_MODEL  # noqa: E402

from kaggle_agent.kaggle import FakeKaggleClient  # noqa: E402
from kaggle_agent.llm import AuthenticationError, QuotaExhaustedError  # noqa: E402
from kaggle_agent.orchestrator import Dependencies, cockpit  # noqa: E402
from kaggle_agent.orchestrator.__main__ import main  # noqa: E402
from kaggle_agent.sandbox import SubprocessSandbox  # noqa: E402


def slow_model(seconds: float, pid_file: str) -> str:
    return MAJORITY_MODEL.replace(
        "def fit(train):\n",
        f"def fit(train):\n    import time\n    open({pid_file!r}, 'w').write(str(os.getpid()))\n"
        f"    time.sleep({seconds})\n")


class ScenarioClaude(RoleAwareClaude):
    def __init__(self, scenario: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario

    def _run(self, prompt, permissions, model):
        block = self.scenario.get("block_file")
        while block and os.path.exists(block):
            time.sleep(0.05)
        if self.scenario.get("delay"):
            time.sleep(float(self.scenario["delay"]))
        response = super()._run(prompt, permissions, model)
        if self.scenario.get("slow_bundle") and self._kind(prompt) == "writer":
            path = Path(permissions.working_dir) / "bundle" / "model.py"
            path.write_text(slow_model(self.scenario["slow_bundle"], self.scenario["bundle_pid_file"])
                            + f"\n# {len(self.prompts)}\n", encoding="utf-8")
        return response


def build(scenario: dict) -> Dependencies:
    clock = TickClock()
    kaggle = FakeKaggleClient({scenario.get("slug", SLUG): competition()}, clock=clock.now, sleep=clock.sleep)
    errors = {}
    for number, kind in (scenario.get("errors") or {}).items():
        errors[int(number)] = AuthenticationError() if kind == "login" else QuotaExhaustedError("usage limit reached")
    claude = ScenarioClaude(scenario, errors=errors)
    sandbox = SubprocessSandbox() if scenario.get("real_sandbox") else InProcessSandbox()
    return Dependencies(kaggle=kaggle, llm=lambda: claude, sandbox=sandbox, clock=clock.now, sleep=clock.sleep)


if __name__ == "__main__":
    argv = sys.argv[1:]
    scenario = {}
    if "--scenario" in argv:
        at = argv.index("--scenario")
        scenario = json.loads(Path(argv[at + 1]).read_text(encoding="utf-8"))
        del argv[at:at + 2]
    if scenario.get("heartbeat"):
        cockpit.HEARTBEAT_INTERVAL = float(scenario["heartbeat"])
    if scenario.get("ignore_sigterm"):
        real = signal.signal
        signal.signal = lambda number, handler: real(number, signal.SIG_IGN if number == signal.SIGTERM else handler)
    sys.exit(main(argv, deps=build(scenario)))
