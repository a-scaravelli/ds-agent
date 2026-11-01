"""Work cycle and round pipeline (design sections 2, 11 and 15, step 9, level 1).

The autonomous cycle as fixed code: a state machine (``CycleMachine``) that persists every
transition in the registry and a sequential round pipeline (coordinator, data, feature
engineering, modeling, bundle writer, runner and frozen evaluation, reviewer, coordinator) with
C candidates per round. Everything external is injected through ``CycleContext``: the roles
(``RoleSet``), the ``BundleWriter``, the runner factory, the ``LeadInput``, the clock and the
hooks. All progress is derived from the registry (``CycleView``), so a resumed run repeats no
finished step.

Not here: the final report, the holdout evaluation and the pipeline export (step 11), the
orchestrator and any real Claude or Kaggle run (step 12), and bundle production by a model: the
``BundleWriter`` is an open interface with a fake for tests.
"""

from .candidate import CandidateRunner
from .context import CycleContext, Hooks
from .derive import PLACEHOLDER_PREFIX, CycleView, RoundState, is_placeholder
from .errors import CycleSuspended, LoopError, TransitionError
from .fake import FakeBundleWriter
from .machine import CycleMachine, CycleOutcome
from .roles import RoleSet, client_roles
from .round import RoundPipeline
from .states import STATES, TRANSITIONS, check_transition
from .writer import BundleRequest, BundleWriter, FixRequest, RoleOutputs

__all__ = [
    "BundleRequest",
    "BundleWriter",
    "CandidateRunner",
    "CycleContext",
    "CycleMachine",
    "CycleOutcome",
    "CycleSuspended",
    "CycleView",
    "FakeBundleWriter",
    "FixRequest",
    "Hooks",
    "LoopError",
    "PLACEHOLDER_PREFIX",
    "RoleOutputs",
    "RoleSet",
    "RoundPipeline",
    "RoundState",
    "STATES",
    "TRANSITIONS",
    "TransitionError",
    "check_transition",
    "client_roles",
    "is_placeholder",
]
