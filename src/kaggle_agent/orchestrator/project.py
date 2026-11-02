"""The project folder: layout, ``project.json`` and the checks that a resumed run uses the same data.

Layout under ``project.folder``::

    data/            the competition files downloaded by the Kaggle client
    registry.db      the SQLite registry
    workspace/       staging folders of the bundle writer and the bundles
    out/             persisted bundles, reports and submission files
    lead.toml        the lead's decisions (written by the lead, never by the program)
    lead.example.toml, NOTICES.txt, lead-ledger.json, project.json
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from kaggle_agent.kaggle import DownloadedFile
from kaggle_agent.kaggle.models import combined_hash

from .config import FROM_INFO, FROM_SPEC, Config, ProjectPaths
from .errors import ProjectError

PROJECT_FILE = "project.json"
NOTICES_FILE = "NOTICES.txt"
LEDGER_FILE = "lead-ledger.json"
EXAMPLE_FILE = "lead.example.toml"
VERSION = 1


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary sibling and rename it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectState:
    """What was decided when the project was created; later runs must agree with it."""

    cycle_id: str
    competition: str
    target: str
    metric: str
    direction: str
    input_kind: str
    protocol: dict
    protocol_config: dict
    train_file: str
    test_file: str
    files: dict
    version: int = VERSION

    @property
    def data_hash(self) -> str:
        return combined_hash(tuple(DownloadedFile(name, 0, sha) for name, sha in self.files.items()))


def cycle_id_for(slug: str) -> str:
    return f"{slug}-1"


def project_file(paths: ProjectPaths) -> Path:
    return paths.folder / PROJECT_FILE


def read_project(paths: ProjectPaths) -> ProjectState | None:
    path = project_file(paths)
    if not path.exists():
        return None
    try:
        return ProjectState(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        raise ProjectError(f"{path} is not a valid project file: {error}") from None


def write_project(paths: ProjectPaths, state: ProjectState) -> None:
    write_atomic(project_file(paths), json.dumps(asdict(state), indent=2, sort_keys=True))


def check_config(state: ProjectState, config: Config) -> None:
    """Refuse a configuration that disagrees with the project on what was frozen at start."""
    comp = config.competition
    wanted = (
        ("competition.slug", comp.slug, state.competition),
        ("competition.target", None if comp.target == FROM_SPEC else comp.target, state.target),
        ("competition.metric", None if comp.metric == FROM_INFO else comp.metric, state.metric),
        ("competition.train_file", comp.train_file, state.train_file),
        ("competition.test_file", comp.test_file, state.test_file),
        ("protocol", config.protocol.to_dict(), state.protocol_config),
    )
    for key, given, stored in wanted:
        if given is not None and given != stored:
            raise ProjectError(
                f"{key}: the configuration says {given!r} but the project was started with "
                f"{stored!r}; a started project cannot change it (use a new project folder)"
            )


def verify_data(paths: ProjectPaths, state: ProjectState) -> None:
    """The data files must be the ones recorded at start (a resume makes no Kaggle call)."""
    for name, expected in state.files.items():
        path = paths.data / name
        if not path.is_file():
            raise ProjectError(f"data file {name} is missing in {paths.data}")
        if file_sha256(path) != expected:
            raise ProjectError(f"data hash mismatch: {name} changed since the project was started")
