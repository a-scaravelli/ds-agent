"""Deterministic ids, prefixed by the cycle id, so a resumed run never writes a record twice."""


def setup_round_id(cycle_id: str) -> str:
    return f"{cycle_id}-R0"


def round_id(cycle_id: str, index: int) -> str:
    """Id of the round at ``index`` (from 1) of the iteration phase."""
    return f"{cycle_id}-R{index}"


def candidate_id(round_id_: str, index: int) -> str:
    return f"{round_id_}-c{index}"


def bundle_id(candidate_id_: str, attempt_number: int) -> str:
    return f"{candidate_id_}-b{attempt_number}"


def fix_bundle_id(bundle_id_: str, retry: int) -> str:
    return f"{bundle_id_}-f{retry}"


def execution_id(candidate_id_: str, sequence: int) -> str:
    return f"{candidate_id_}-e{sequence}"


def report_id(round_id_: str) -> str:
    return f"{round_id_}-rep"


def guidance_id(round_id_: str) -> str:
    return f"{round_id_}-g"
