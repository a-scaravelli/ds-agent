"""The Claude-driven bundle writer (design section 15, step 12, part 1, level 1).

``ClaudeBundleWriter`` implements the open ``loop.BundleWriter`` interface: Claude writes the bundle
files in an isolated staging directory with file tools only, the writer checks them with the bundle
loader and the contract checks, repairs them for a bounded number of turns, and copies only a
verified bundle to the requested folder. Errors are ``BundleError`` subclasses, so the loop maps
them as before; a Claude error (``LlmError``) propagates unchanged. The dollar cost of the writer
calls is not recorded. Out of scope here: the orchestrator, the command line, the real lead input,
the real submission and tools for the other roles.
"""

from .checks import DEFAULT_DEPENDENCIES, BundleLimits, check_bundle
from .claude_writer import WRITER_TOOLS, ClaudeBundleWriter, WriterCall, default_protected
from .errors import (
    BundleWriterError,
    InvalidBundleError,
    PreviousBundleError,
    WorkspaceViolationError,
    WriterConfigurationError,
)
from .prompt import WriterTask, build_prompt, prompt_hash, repair_prompt
from .reply import WriterReport

__all__ = [
    "BundleLimits",
    "BundleWriterError",
    "ClaudeBundleWriter",
    "DEFAULT_DEPENDENCIES",
    "InvalidBundleError",
    "PreviousBundleError",
    "WRITER_TOOLS",
    "WorkspaceViolationError",
    "WriterCall",
    "WriterConfigurationError",
    "WriterReport",
    "WriterTask",
    "build_prompt",
    "check_bundle",
    "default_protected",
    "prompt_hash",
    "repair_prompt",
]
