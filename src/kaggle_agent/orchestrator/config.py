"""The single human-readable configuration file of a competition (TOML, standard library only).

Sections (every key is optional unless marked required; any other key is an error)::

    [competition]
    slug = "titanic"                       # required
    target = "Survived"                    # or "from_submission_spec" (the 2nd column of the sample)
    metric = "accuracy"                    # or "from_competition_info" (needs an unambiguous text)
    direction = "from_metric"              # or "maximize" / "minimize" (must agree with the metric)
    train_file = "train.csv"
    test_file = "test.csv"

    [project]
    folder = "runs/titanic"                # required; relative paths are relative to this file
    registry = ""  workspace = ""  out = ""  data = ""   # optional overrides of the derived paths

    [limits]                               # rounds N, candidates C, stagnation K, pause_every M, ...
    [protocol]                             # validation/holdout scheme, share, seed, repetitions, k
    [models]                               # optional model name per role (and the writer)
    [runtime]                              # bundle_timeout, writer_max_repairs, wait_seconds, ...
    [lead]                                 # file = "<path of lead.toml>"

No secret is ever stored here: a key that looks like one (token, key, password, secret, KAGGLE_*,
ANTHROPIC_*) is refused. Kaggle and Claude credentials use their own mechanisms.
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.budget import CycleLimits
from kaggle_agent.evaluation import Protocol, get_metric
from kaggle_agent.proposals import ShortPlan, ValidationError
from kaggle_agent.writer import DEFAULT_DEPENDENCIES

from .errors import ConfigError

DIRECTIONS = ("maximize", "minimize")
FROM_INFO = "from_competition_info"
FROM_SPEC = "from_submission_spec"
FROM_METRIC = "from_metric"
ROLE_NAMES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer", "writer")
_SECRET = re.compile(r"token|key|password|secret|^kaggle_|^anthropic_", re.IGNORECASE)
CREDENTIALS_HINT = (
    "never store a secret in the configuration: Kaggle credentials use KAGGLE_USERNAME/KAGGLE_KEY or "
    "~/.kaggle/kaggle.json (or access_token), Claude uses the subscription login of the CLI"
)

_PROTOCOL_DEFAULTS = {
    "validation_scheme": "random",
    "validation_params": {"n_folds": 5},
    "holdout_scheme": "random",
    "holdout_params": {},
    "holdout_share": 0.2,
    "seed": 0,
    "repetitions": 1,
    "threshold_k": 1.0,
}
_LIMIT_KEYS = ("rounds", "candidates", "stagnation", "pause_every", "fix_retries",
               "max_consecutive_failures", "max_rejections")
_SECTIONS = {
    "competition": ("slug", "target", "metric", "direction", "train_file", "test_file"),
    "project": ("folder", "registry", "workspace", "out", "data"),
    "limits": _LIMIT_KEYS,
    "protocol": tuple(_PROTOCOL_DEFAULTS),
    "models": ROLE_NAMES,
    "runtime": ("bundle_timeout", "writer_max_repairs", "allowed_dependencies", "wait_seconds",
                "poll_interval"),
    "lead": ("file",),
}


@dataclass(frozen=True)
class CompetitionConfig:
    slug: str
    target: str = FROM_SPEC
    metric: str = FROM_INFO
    direction: str = FROM_METRIC
    train_file: str = "train.csv"
    test_file: str = "test.csv"


@dataclass(frozen=True)
class ProjectPaths:
    folder: Path
    registry: Path
    workspace: Path
    out: Path
    data: Path


@dataclass(frozen=True)
class ProtocolConfig:
    validation_scheme: str
    validation_params: dict
    holdout_scheme: str
    holdout_params: dict
    holdout_share: float
    seed: int
    repetitions: int
    threshold_k: float

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _PROTOCOL_DEFAULTS}

    def build(self, metric: str, direction: str, target: str) -> Protocol:
        """The ``evaluation.Protocol``; it validates everything again."""
        try:
            return Protocol(metric=metric, metric_params={}, direction=direction, target=target,
                            **self.to_dict())
        except ValidationError as error:
            raise ConfigError(f"protocol: {error}") from None


@dataclass(frozen=True)
class RuntimeConfig:
    bundle_timeout: float = 600.0
    writer_max_repairs: int = 2
    allowed_dependencies: tuple[str, ...] = DEFAULT_DEPENDENCIES
    wait_seconds: float = 600.0
    poll_interval: float = 30.0


@dataclass(frozen=True)
class Config:
    competition: CompetitionConfig
    paths: ProjectPaths
    limits: CycleLimits
    protocol: ProtocolConfig
    models: dict = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lead_file: Path | None = None
    source: Path | None = None

    @property
    def lead_path(self) -> Path:
        return self.lead_file or self.paths.folder / "lead.toml"

    def model_for(self, role: str) -> str | None:
        return self.models.get(role)

    def seed_plan(self, metric: str, target: str) -> ShortPlan:
        """The plan stored on the cycle: generated from the configuration, not from the coordinator."""
        validation = (f"{self.protocol.validation_scheme} validation with "
                      f"{self.protocol.validation_params.get('n_folds', 'default')} folds, frozen holdout")
        return ShortPlan(
            metric=metric,
            validation=validation,
            feature_engineering=("basic cleaning and encoding of the columns",),
            models=("simple models from scikit-learn",),
            baseline=baseline_text(metric, target),
        )


def baseline_text(metric: str, target: str) -> str:
    """The baseline item of the seed plan (the baseline attempt reads 'Baseline: ' plus this text)."""
    return f"simplest sensible model for {metric} on {target}"


# -- reading -------------------------------------------------------------------------------------


def _scan_secrets(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if _SECRET.search(str(key)):
                raise ConfigError(f"{where}: this looks like a secret. {CREDENTIALS_HINT}")
            _scan_secrets(item, where)


def _section(raw: dict, name: str, required: bool = False) -> dict:
    value = raw.get(name)
    if value is None:
        if required:
            raise ConfigError(f"[{name}]: the section is required")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: must be a table")
    unknown = sorted(set(value) - set(_SECTIONS[name]))
    if unknown:
        raise ConfigError(f"{name}.{unknown[0]}: unknown key (allowed: {', '.join(_SECTIONS[name])})")
    return value


def _text(section: dict, name: str, key: str, default: str | None = None) -> str | None:
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}.{key}: must be a non-empty string")
    return value


def _number(section: dict, name: str, key: str, default, kind=(int, float), minimum=None, exclusive=False):
    if key not in section:
        return default
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        word = "an integer" if kind == int else "a number"
        raise ConfigError(f"{name}.{key}: must be {word}")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        raise ConfigError(f"{name}.{key}: must be {'greater than' if exclusive else 'at least'} {minimum}")
    return value


def _path(value: str | None, base: Path, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: Path | str) -> Config:
    """Read and strictly validate a configuration file; ``ConfigError`` names the key path."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from None
    except OSError as error:
        raise ConfigError(f"{path}: cannot be read: {error}") from None
    _scan_secrets(raw, "")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"{unknown[0]}: unknown section (allowed: {', '.join(_SECTIONS)})")
    base = path.resolve().parent

    comp = _section(raw, "competition", required=True)
    slug = _text(comp, "competition", "slug")
    if slug is None:
        raise ConfigError("competition.slug: is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug):
        raise ConfigError("competition.slug: must be letters, digits, '-' or '_'")
    metric = _text(comp, "competition", "metric", FROM_INFO)
    if metric != FROM_INFO:
        try:
            get_metric(metric)
        except ValidationError as error:
            raise ConfigError(f"competition.metric: {error}") from None
    direction = _text(comp, "competition", "direction", FROM_METRIC)
    if direction != FROM_METRIC:
        if direction not in DIRECTIONS:
            raise ConfigError(f"competition.direction: must be {FROM_METRIC}, "
                              f"{' or '.join(DIRECTIONS)}")
        if metric != FROM_INFO and get_metric(metric).direction != direction:
            raise ConfigError(f"competition.direction: metric {metric!r} is naturally "
                              f"{get_metric(metric).direction}, not {direction}")
    competition = CompetitionConfig(
        slug=slug, target=_text(comp, "competition", "target", FROM_SPEC), metric=metric,
        direction=direction, train_file=_text(comp, "competition", "train_file", "train.csv"),
        test_file=_text(comp, "competition", "test_file", "test.csv"),
    )

    proj = _section(raw, "project", required=True)
    folder_text = _text(proj, "project", "folder")
    if folder_text is None:
        raise ConfigError("project.folder: is required")
    folder = _path(folder_text, base, base)
    paths = ProjectPaths(
        folder=folder,
        registry=_path(_text(proj, "project", "registry"), base, folder / "registry.db"),
        workspace=_path(_text(proj, "project", "workspace"), base, folder / "workspace"),
        out=_path(_text(proj, "project", "out"), base, folder / "out"),
        data=_path(_text(proj, "project", "data"), base, folder / "data"),
    )

    lim = _section(raw, "limits")
    defaults = CycleLimits()
    values = {}
    for key in _LIMIT_KEYS:
        field_name = "stagnation_rounds" if key == "stagnation" else key
        minimum = 0 if key == "fix_retries" else 1
        values[field_name] = _number(lim, "limits", key, getattr(defaults, field_name), int, minimum)
    limits = CycleLimits(**values)

    prot = _section(raw, "protocol")
    merged = {}
    for key, default in _PROTOCOL_DEFAULTS.items():
        merged[key] = prot.get(key, dict(default) if isinstance(default, dict) else default)
        if key.endswith("_params") and not isinstance(merged[key], dict):
            raise ConfigError(f"protocol.{key}: must be a table")
        if key.endswith("_scheme") and (not isinstance(merged[key], str) or not merged[key]):
            raise ConfigError(f"protocol.{key}: must be a non-empty string")
    _number(prot, "protocol", "holdout_share", 0.2, (int, float), 0, exclusive=True)
    if merged["holdout_share"] >= 1:
        raise ConfigError("protocol.holdout_share: must be below 1")
    _number(prot, "protocol", "seed", 0, int, 0)
    _number(prot, "protocol", "repetitions", 1, int, 1)
    _number(prot, "protocol", "threshold_k", 1.0, (int, float), 0)
    protocol = ProtocolConfig(**merged)

    mod = _section(raw, "models")
    models = {role: _text(mod, "models", role) for role in ROLE_NAMES if role in mod}

    run = _section(raw, "runtime")
    deps = run.get("allowed_dependencies", DEFAULT_DEPENDENCIES)
    if (not isinstance(deps, (list, tuple)) or not all(isinstance(d, str) and d.strip() for d in deps)):
        raise ConfigError("runtime.allowed_dependencies: must be a list of non-empty strings")
    runtime = RuntimeConfig(
        bundle_timeout=float(_number(run, "runtime", "bundle_timeout", 600.0, (int, float), 0, True)),
        writer_max_repairs=_number(run, "runtime", "writer_max_repairs", 2, int, 0),
        allowed_dependencies=tuple(deps),
        wait_seconds=float(_number(run, "runtime", "wait_seconds", 600.0, (int, float), 0)),
        poll_interval=float(_number(run, "runtime", "poll_interval", 30.0, (int, float), 0, True)),
    )

    lead = _section(raw, "lead")
    lead_text = _text(lead, "lead", "file")
    config = Config(competition=competition, paths=paths, limits=limits, protocol=protocol,
                    models=models, runtime=runtime, source=path.resolve(),
                    lead_file=None if lead_text is None else _path(lead_text, base, folder / "lead.toml"))
    # the protocol is validated as a whole as soon as the metric is known
    if metric != FROM_INFO:
        spec = get_metric(metric)
        config.protocol.build(metric, spec.direction, "target")
    return config
