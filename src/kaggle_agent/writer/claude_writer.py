"""A ``BundleWriter`` driven by Claude (design section 15, step 12, part 1, level 1).

Claude writes the bundle files into an isolated staging directory with file tools only (Read,
Write, Edit, Glob: no Bash, no network). The writer checks the result with the existing loader and
contract checks and, on failure, runs a bounded number of repair turns giving Claude the exact check
error. Only a verified bundle is copied to ``request.folder``. It never runs the evaluation and never
sees the holdout or any label: only the schema of the development data and the data findings.

Recording: every Claude call is stored as an ``llm_calls`` row (tokens, model, outcome); the
``purpose`` text carries the bundle id, the turn and the first 12 hex characters of the prompt hash.
EXPLICIT LIMIT: the dollar cost of the writer calls is NOT recorded, because ``CallRecord`` and the
``llm_calls`` table have no cost field; tokens, model and prompt hash are recorded. Adding cost
needs a follow-up decision. Do not also build the client with ``on_call`` recording to the same
registry: ``WriterConfigurationError`` is raised.

Errors: an invalid bundle, a workspace violation or an unreadable previous bundle are
``BundleError`` subclasses; an ``LlmError`` of the client propagates unchanged.
"""

import functools
import shutil
from dataclasses import dataclass
from pathlib import Path

import kaggle_agent.bundle as bundle_package
from kaggle_agent.bundle import BundleLayoutError, load_bundle
from kaggle_agent.llm import CallRecord, LlmClient, ToolPermissions, entry_from_record, recording
from kaggle_agent.loop.writer import BundleRequest, BundleWriter
from kaggle_agent.registry import Registry
from kaggle_agent.sandbox import ProtectedPath

from .checks import DEFAULT_DEPENDENCIES, BundleLimits, check_bundle
from .errors import (
    BundleWriterError,
    InvalidBundleError,
    PreviousBundleError,
    WriterConfigurationError,
)
from .prompt import WriterTask, build_prompt, prompt_hash, repair_prompt
from .reply import WriterReport
from .workspace import Staging, read_previous_bundle

WRITER_TOOLS = ("Read", "Write", "Edit", "Glob")
PREVIOUS_CODE_CAP = 20_000
HASH_CHARS = 12


@dataclass(frozen=True)
class WriterCall:
    """One Claude invocation of the writer: the turn, the full prompt hash and the call record."""

    turn: int
    prompt_hash: str
    record: CallRecord


def default_protected() -> tuple[ProtectedPath, ...]:
    """The ``bundle`` and ``evaluation`` packages, the same entries the bundle runner protects."""
    bundle_dir = Path(bundle_package.__file__).resolve().parent
    package = bundle_dir.parent
    return tuple(
        ProtectedPath(f"{package.name}/{name}", path)
        for name, path in (("bundle", bundle_dir), ("evaluation", package / "evaluation"))
    )


class ClaudeBundleWriter(BundleWriter):
    """Writes a bundle with Claude, verifies it and copies it to ``request.folder``.

    ``max_repairs`` repair turns follow the first one (at most ``max_repairs + 1`` Claude calls).
    The writer gets no DataFrame, engine or holdout accessor: only the ``WriterTask`` schema.
    """

    def __init__(
        self,
        client: LlmClient,
        registry: Registry,
        cycle_id: str,
        workspace_root: Path,
        task: WriterTask,
        max_repairs: int = 2,
        allowed_dependencies: tuple[str, ...] = DEFAULT_DEPENDENCIES,
        protected: tuple[ProtectedPath, ...] | None = None,
        model: str | None = None,
        limits: BundleLimits = BundleLimits(),
    ) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must not be negative")
        self._check_wiring(client, registry)
        self.client = client
        self.registry = registry
        self.cycle_id = cycle_id
        self.workspace_root = Path(workspace_root)
        self.task = task
        self.max_repairs = max_repairs
        self.allowed_dependencies = tuple(allowed_dependencies)
        self.protected = default_protected() if protected is None else tuple(protected)
        self.model = model
        self.limits = limits
        self.calls: list[WriterCall] = []

    @staticmethod
    def _check_wiring(client: LlmClient, registry: Registry) -> None:
        hook = client.on_call
        if (
            isinstance(hook, functools.partial)
            and hook.func is recording.record_call
            and hook.args
            and hook.args[0] is registry
        ):
            raise WriterConfigurationError(
                "the client already records its calls in this registry with record_call; "
                "the writer stores the llm_calls rows itself, so build the client without that hook"
            )

    # -- previous bundle of a fix ------------------------------------------------------------

    def _previous_files(self, request: BundleRequest) -> dict[str, str]:
        previous_id = request.fix.previous_bundle_id
        candidates = []
        for record in self.registry.list_bundles(self.cycle_id):
            if record.id == previous_id and record.reference:
                candidates.append(Path(record.reference))
        candidates.append(request.folder.parent / previous_id)
        for folder in candidates:
            try:
                return read_previous_bundle(folder, PREVIOUS_CODE_CAP)
            except PreviousBundleError:
                continue
        raise PreviousBundleError(f"previous bundle {previous_id} not found in the registry or next to the new folder")

    # -- one Claude call -----------------------------------------------------------------------

    def _ask(self, request: BundleRequest, staging: Staging, prompt: str, turn: int) -> None:
        digest = prompt_hash(prompt)
        purpose = f"bundle writer: {request.bundle_id} turn {turn} prompt {digest[:HASH_CHARS]}"
        permissions = ToolPermissions(staging.directory, WRITER_TOOLS)
        before = len(self.client.call_log())
        try:
            self.client.ask(prompt, WriterReport, permissions, purpose, model=self.model)
        finally:
            for record in self.client.call_log()[before:]:
                self.registry.add_llm_call(entry_from_record(self.cycle_id, record))
                self.calls.append(WriterCall(turn, digest, record))

    # -- write -----------------------------------------------------------------------------------

    def write(self, request: BundleRequest) -> Path:
        folder = Path(request.folder)
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise BundleWriterError(f"the bundle folder is not empty: {folder}")
        previous = self._previous_files(request) if request.fix is not None else None
        base = build_prompt(request, self.task, self.allowed_dependencies, previous)
        prompt = base
        with Staging(self.workspace_root, folder, self.protected) as staging:
            error = ""
            for turn in range(1, self.max_repairs + 2):
                self._ask(request, staging, prompt, turn)
                staging.check()
                try:
                    staged = check_bundle(
                        staging.bundle, request.bundle_id, self.allowed_dependencies, self.limits
                    )
                except BundleLayoutError as exc:
                    error = str(exc)
                    prompt = repair_prompt(base, error)
                    continue
                folder.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    staging.bundle, folder, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
                final = load_bundle(folder)
                if final.folder_hash != staged.folder_hash:
                    raise BundleWriterError("the copied bundle differs from the checked bundle")
                return folder
        raise InvalidBundleError(error, self.max_repairs + 1)
