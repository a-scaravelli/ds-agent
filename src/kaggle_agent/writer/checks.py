"""Static checks of a staged bundle. Never imports or runs the bundle, never evaluates it."""

import re
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.bundle import MANIFEST_NAME, MODEL_FOLDER, BundleLayoutError, LoadedBundle, load_bundle

DEFAULT_DEPENDENCIES = ("numpy", "pandas", "scikit-learn")  # the pinned libraries the child can import


@dataclass(frozen=True)
class BundleLimits:
    """Caps on the size of what the writer may produce."""

    max_files: int = 50
    max_bytes: int = 500_000


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name.strip().lower())


def _dependency_name(item: str) -> str:
    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", item.strip())
    return _normalize(match.group(0)) if match else _normalize(item)


def check_bundle(folder: Path, bundle_id: str, allowed_dependencies: tuple[str, ...],
                 limits: BundleLimits = BundleLimits()) -> LoadedBundle:
    """Load the staged bundle and add the writer checks; ``BundleLayoutError`` carries the text."""
    loaded = load_bundle(folder)
    if loaded.bundle_id != bundle_id:
        raise BundleLayoutError(
            f"the manifest has bundle_id {loaded.bundle_id!r}, expected {bundle_id!r}"
        )
    allowed = {_normalize(name) for name in allowed_dependencies}
    for item in loaded.manifest.dependencies:
        if _dependency_name(item) not in allowed:
            raise BundleLayoutError(
                f"dependency {item!r} in {MANIFEST_NAME} is not allowed; "
                f"allowed: {', '.join(sorted(allowed_dependencies))}"
            )
    if (loaded.folder / MODEL_FOLDER).exists():
        raise BundleLayoutError(f"the bundle must not contain a {MODEL_FOLDER}/ folder: fit writes it later")
    files = [p for p in loaded.folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if len(files) > limits.max_files:
        raise BundleLayoutError(f"too many files: {len(files)} (maximum {limits.max_files})")
    total = sum(p.stat().st_size for p in files)
    if total > limits.max_bytes:
        raise BundleLayoutError(f"the bundle is too large: {total} bytes (maximum {limits.max_bytes})")
    return loaded
