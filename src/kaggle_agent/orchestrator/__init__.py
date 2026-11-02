"""The orchestrator: one command line runs a real tabular Kaggle competition (design step 12, level 1).

It connects the existing packages: it builds a ``CycleContext`` from one TOML configuration
(``build``), reads the lead's decisions from a file (``lead_file``), runs the cycle and the pipeline
(``service``) and, only on the lead's explicit command after the review, sends the submission
(``submission``). ``python -m kaggle_agent.orchestrator start|resume|status|submit --config FILE``.

Level 1 only: no web input, no budget in money or time, one competition per project folder.
"""

from .build import Built, Dependencies, build_context, real_dependencies
from .codes import CommandResult
from .config import Config, load_config
from .errors import ConfigError, LeadFileError, OrchestratorError, ProjectError, SubmitRefused
from .lead_file import FileLeadInput

__all__ = [
    "Built",
    "CommandResult",
    "Config",
    "ConfigError",
    "Dependencies",
    "FileLeadInput",
    "LeadFileError",
    "OrchestratorError",
    "ProjectError",
    "SubmitRefused",
    "build_context",
    "load_config",
    "real_dependencies",
]
