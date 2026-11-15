"""Turn registry records into JSON-safe plain values (design section 15, cockpit).

Frozen dataclasses become dicts, tuples and sets become lists, and a float that JSON cannot
carry (NaN, infinity) becomes ``None``. Nothing here reads or writes anything.
"""

import dataclasses
import math


def jsonable(value: object) -> object:
    """A copy of ``value`` made only of dict, list, str, int, float, bool and None."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
