"""Hash of the protocol together with the evaluation code (design section 5).

The code part is computed at runtime from the source files of this package,
so nobody has to remember to bump a version constant. Library versions are
left out on purpose: they are pinned and do not change inside a cycle; they
are recorded in the execution resources instead. The hash depends on the
bytes of the files, so a reformat or different line endings change it, and
any edit of the package invalidates the frozen protocols of open cycles.
"""

import hashlib
import json
from pathlib import Path

from .protocol import Protocol


def evaluation_code_hash(package_dir: Path | str | None = None) -> str:
    """SHA-256 of the ``.py`` files of the package, by relative path, ignoring caches."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(
        (p for p in root.rglob("*.py") if "__pycache__" not in p.relative_to(root).parts),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def protocol_hash(protocol: Protocol, code_hash: str) -> str:
    """SHA-256 of the canonical protocol content and the evaluation code hash."""
    canonical = json.dumps(
        {"protocol": protocol.to_dict(), "code": code_hash},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
