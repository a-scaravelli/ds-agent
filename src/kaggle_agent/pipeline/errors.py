"""Errors of the pipeline package."""


class PipelineError(Exception):
    """Base class of the errors of this package."""


class PersistedBundleError(PipelineError):
    """A persisted bundle folder is missing, malformed or changed since it was written."""
