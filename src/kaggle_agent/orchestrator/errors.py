"""Errors of the orchestrator; each one carries a message the lead can act on."""


class OrchestratorError(Exception):
    """Base class of the errors of this package."""


class ConfigError(OrchestratorError):
    """The configuration file is missing, unreadable or invalid; the message names the key path."""


class ProjectError(OrchestratorError):
    """The project folder is missing or does not match the configuration or the data."""


class LeadFileError(OrchestratorError):
    """The lead file is not valid TOML or holds an unknown key or a wrong value."""


class SubmitRefused(OrchestratorError):
    """A submission was refused; ``code`` is the exit code of the command."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)
