"""The out-of-fold table: encoding and bounded decoding.

Format: gzip of a CSV (mtime 0, so equal input gives equal bytes). Columns: ``row`` (position in the training
frame, the same as ``Partition.valid_rows``), ``rep`` (repetition), ``fold``, ``truth``, then either
``prediction`` (one value per row) or ``p0..pk-1`` (a matrix; class ``i`` is the i-th SORTED class label,
as the metrics use them). Pure: bytes in, bytes out; the caller does the file work.
"""

import gzip
import io
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_ROWS = 200_000
MAX_COMPRESSED = 25 * 1024 * 1024
MAX_DECOMPRESSED = 200 * 1024 * 1024
BASE_COLUMNS = ("row", "rep", "fold", "truth")


class OofError(ValueError):
    """The table cannot be encoded or decoded (too large, bad columns, bad data)."""


@dataclass(frozen=True)
class OofTable:
    frame: pd.DataFrame
    classes: tuple          # sorted class labels (classification), empty for regression
    matrix: bool            # True when the columns p0..pk-1 are present

    @property
    def probability_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c not in BASE_COLUMNS and c != "prediction"]


def encode_oof(rows, reps, folds, truth, predictions, max_rows: int = MAX_ROWS) -> bytes:
    """The table as gzip CSV bytes; ``predictions`` is 1-D or a matrix (one row per input row)."""
    predictions = np.asarray(predictions)
    count = len(rows)
    if count > max_rows:
        raise OofError(f"{count} rows are over the limit of {max_rows}")
    if not (len(reps) == len(folds) == len(truth) == len(predictions) == count):
        raise OofError("columns of different length")
    data = {"row": np.asarray(rows, dtype=np.int64), "rep": np.asarray(reps, dtype=np.int64),
            "fold": np.asarray(folds, dtype=np.int64), "truth": np.asarray(truth)}
    if predictions.ndim == 1:
        data["prediction"] = predictions
    elif predictions.ndim == 2:
        for index in range(predictions.shape[1]):
            data[f"p{index}"] = predictions[:, index].astype(float)
    else:
        raise OofError("predictions must be one- or two-dimensional")
    text = pd.DataFrame(data).to_csv(index=False, lineterminator="\n")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as stream:
        stream.write(text.encode("utf-8"))
    return buffer.getvalue()


def _inflate(blob: bytes, limit: int) -> bytes:
    if len(blob) > MAX_COMPRESSED:
        raise OofError("the compressed file is over the size limit")
    inflater = zlib.decompressobj(wbits=31)
    try:
        out = inflater.decompress(blob, limit + 1)
    except zlib.error as error:
        raise OofError(f"not a gzip file ({error})") from None
    if len(out) > limit or inflater.unconsumed_tail:
        raise OofError("the decompressed file is over the size limit")
    return out


def decode_oof(blob: bytes, max_rows: int = MAX_ROWS, max_bytes: int = MAX_DECOMPRESSED,
               classification: bool = False) -> OofTable:
    """Decode with limits on rows and decompressed bytes and strict column checks.

    ``classification`` (or a probability matrix) makes ``classes`` the sorted labels of the truth."""
    text = _inflate(blob, max_bytes)
    try:
        frame = pd.read_csv(io.BytesIO(text), nrows=max_rows + 1)
    except Exception as error:
        raise OofError(f"not a readable table ({type(error).__name__})") from None
    if len(frame) > max_rows:
        raise OofError(f"more than {max_rows} rows")
    columns = list(frame.columns)
    if columns[:4] != list(BASE_COLUMNS):
        raise OofError(f"the first columns must be {', '.join(BASE_COLUMNS)}")
    rest = columns[4:]
    matrix = bool(rest) and rest[0] == "p0"
    if rest != ["prediction"] and not (matrix and rest == [f"p{i}" for i in range(len(rest))] and len(rest) >= 2):
        raise OofError("the prediction columns must be 'prediction' or p0..pk-1 (k >= 2)")
    for name in ("row", "rep", "fold"):
        if not pd.api.types.is_integer_dtype(frame[name]):
            raise OofError(f"column {name} must hold integers")
    for name in rest:
        if matrix and not pd.api.types.is_float_dtype(frame[name]):
            raise OofError(f"column {name} must hold numbers")
    if frame["truth"].isna().any():
        raise OofError("the truth column has missing values")
    classes: tuple = ()
    if matrix or classification:
        classes = tuple(sorted(frame["truth"].unique().tolist()))
    if matrix:
        if len(classes) > len(rest):
            raise OofError("more classes in the truth than probability columns")
    return OofTable(frame, classes, matrix)


MAX_SNAPSHOT_ROWS = 200_000
MAX_SNAPSHOT_COLUMNS = 500
MAX_SNAPSHOT_BYTES = 400 * 1024 * 1024


def decode_snapshot(blob: bytes, max_rows: int = MAX_SNAPSHOT_ROWS) -> pd.DataFrame:
    """The development features (a ``row`` column, then the features), indexed by row; bounded."""
    text = _inflate_snapshot(blob)
    try:
        frame = pd.read_csv(io.BytesIO(text), nrows=max_rows + 1)
    except Exception as error:
        raise OofError(f"not a readable table ({type(error).__name__})") from None
    if len(frame) > max_rows:
        raise OofError(f"more than {max_rows} rows")
    if not len(frame.columns) or frame.columns[0] != "row" or len(frame.columns) > MAX_SNAPSHOT_COLUMNS + 1:
        raise OofError("the first column must be row, with at most 500 features")
    if not pd.api.types.is_integer_dtype(frame["row"]):
        raise OofError("column row must hold integers")
    return frame.set_index("row")


def _inflate_snapshot(blob: bytes) -> bytes:
    inflater = zlib.decompressobj(wbits=31)
    try:
        out = inflater.decompress(blob, MAX_SNAPSHOT_BYTES + 1)
    except zlib.error as error:
        raise OofError(f"not a gzip file ({error})") from None
    if len(out) > MAX_SNAPSHOT_BYTES:
        raise OofError("the decompressed file is over the size limit")
    return out
