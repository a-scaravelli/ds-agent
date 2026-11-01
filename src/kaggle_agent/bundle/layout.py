"""Loader and validator of a bundle folder (design section 8).

The checks read the files and parse them with ``ast``; the bundle code is never imported
or run here.
"""

import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.proposals import ValidationError

from .contract import ENTRY_POINT_NAMES, MANIFEST_NAME, MODEL_FOLDER, Manifest, parse_entry_point
from .errors import BundleLayoutError

# Import names that differ from the distribution name (normalized: lower case, ``_``).
IMPORT_ALIASES = {
    "sklearn": "scikit_learn",
    "cv2": "opencv_python",
    "pil": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
}
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class LoadedBundle:
    """A validated bundle: manifest, resolved folder and the hash of its content."""

    manifest: Manifest
    folder: Path
    folder_hash: str

    @property
    def bundle_id(self) -> str:
        return self.manifest.bundle_id


def folder_hash_of(folder: Path) -> str:
    """SHA-256 of a bundle folder, ignoring ``__pycache__`` and a top-level ``model/`` folder.

    For a folder without ``model/`` it equals ``kaggle_agent.sandbox.hash_path`` (sorted
    relative paths and bytes; a test pins it), so a persisted model never changes the hash.
    """
    root = Path(folder)
    files = []
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if path.is_file() and "__pycache__" not in parts and not (parts[0] == MODEL_FOLDER and len(parts) > 1):
            files.append(path)
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(item.read_bytes() + b"\0")
    return digest.hexdigest()


def schema_of(manifest: Manifest) -> dict:
    """JSON-compatible description of a bundle, stored in ``BundleRecord.schema``."""
    return {
        "bundle_id": manifest.bundle_id,
        "entry_points": dict(manifest.entry_points),
        "dependencies": list(manifest.dependencies),
        "fit": {"input": "DataFrame with the target column", "output": "model saved under BUNDLE_MODEL_DIR"},
        "predict": {"input": "bundle_id and a DataFrame without the target column",
                    "output": "one prediction per row (or one row of probabilities per row)"},
    }


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name.strip().lower())


def _declared(dependencies) -> set[str]:
    names = set()
    for item in dependencies:
        match = _DEPENDENCY_NAME.match(item.strip())
        if match is None:
            raise BundleLayoutError(f"invalid dependency {item!r}")
        names.add(_normalize(match.group(0)))
    return names


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _python_files(root: Path) -> list[Path]:
    found = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        found += [Path(current, f) for f in files if f.endswith(".py")]
    return sorted(found)


def _check_links(root: Path) -> None:
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in [*dirs, *files]:
            path = Path(current, name)
            if not path.is_symlink():
                continue
            rel = path.relative_to(root).as_posix()
            if not _inside(root, path):
                raise BundleLayoutError(f"{rel} is a link that points outside the bundle folder")
            if path.is_dir():
                raise BundleLayoutError(f"{rel} is a link to a folder, which is not allowed")


def _parse(path: Path, root: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise BundleLayoutError(f"{path.relative_to(root).as_posix()} cannot be parsed: {error}") from None


def _module_file(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _check_entry_point(root: Path, kind: str, text: str) -> None:
    try:
        module, function = parse_entry_point(text)
    except ValidationError as error:
        raise BundleLayoutError(f"entry point {kind}: {error.reason}") from None
    path = _module_file(root, module)
    if path is None:
        raise BundleLayoutError(f"entry point {kind}: module {module!r} not found in the bundle folder")
    if not _inside(root, path):
        raise BundleLayoutError(f"entry point {kind}: module {module!r} resolves outside the bundle folder")
    defined = {n.name for n in _parse(path, root).body if isinstance(n, ast.FunctionDef)}
    if function not in defined:
        raise BundleLayoutError(
            f"entry point {kind}: {path.relative_to(root).as_posix()} does not define {function}()"
        )


def _local_names(root: Path) -> set[str]:
    names = set()
    for item in root.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            names.add(item.name)
        elif item.suffix == ".py":
            names.add(item.stem)
    return names


def _check_imports(root: Path, manifest: Manifest) -> None:
    """Reject imports of third-party libraries that ``dependencies`` does not declare.

    Heuristic: it reads every static ``import`` of every ``.py`` file (also inside
    functions) and does not see dynamic imports such as ``importlib.import_module``.
    """
    allowed = _declared(manifest.dependencies)
    known = set(sys.stdlib_module_names) | _local_names(root)
    for path in _python_files(root):
        for node in ast.walk(_parse(path, root)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for full in names:
                top = full.split(".")[0]
                key = _normalize(top)
                if top in known or key in allowed or IMPORT_ALIASES.get(key) in allowed:
                    continue
                raise BundleLayoutError(
                    f"{path.relative_to(root).as_posix()} imports {top!r}, which is not declared "
                    f"in the dependencies of {MANIFEST_NAME}"
                )


def load_bundle(folder: Path | str) -> LoadedBundle:
    """Validate a bundle folder and return it with its content hash.

    Raises ``BundleLayoutError`` for a missing or invalid manifest, a missing ``fit`` or
    ``predict``, a path or link that escapes the folder, or an undeclared third-party import.
    """
    root = Path(folder).resolve()
    if not root.is_dir():
        raise BundleLayoutError(f"{folder} is not a folder")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleLayoutError(f"{MANIFEST_NAME} not found in {root.name}")
    _check_links(root)
    try:
        manifest = Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (ValueError, OSError) as error:  # ValidationError and JSON errors are ValueError
        raise BundleLayoutError(f"invalid {MANIFEST_NAME}: {error}") from None
    for kind in ENTRY_POINT_NAMES:
        _check_entry_point(root, kind, manifest.entry_points[kind])
    _check_imports(root, manifest)
    return LoadedBundle(manifest, root, folder_hash_of(root))
