"""Pin database reads to the rows a past experiment used.

The study database keeps growing, so reading whole tables gives a different cohort over
time. A row manifest records, per table, which rows existed when an experiment ran (as
ranges of the table's ``id`` primary key) and a digest of their contents. With a manifest
set, ``DatabaseService.extract_from_database`` keeps only those rows and checks the digest,
so the cohort is rebuilt from the live database without storing any data in the repo.

Format (JSON)::

    {"tables": {"<table>": {"id_ranges": [[first, last], ...], "rows": n, "sha256": "..."}}}

Tables not listed are read in full.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def _canonical(value: Any) -> str:
    """Text form of one cell that is stable across reads (JSON cells key-sorted)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def table_digest(df: pd.DataFrame) -> str:
    """SHA-256 over a table's columns and rows, in ``id`` order."""
    df = df.sort_values("id", kind="stable") if "id" in df.columns else df
    h = hashlib.sha256("\x1f".join(map(str, df.columns)).encode())
    for row in df.itertuples(index=False, name=None):
        h.update(("\n" + "\x1f".join(_canonical(v) for v in row)).encode())
    return h.hexdigest()


def id_ranges(ids: np.ndarray) -> List[Tuple[int, int]]:
    """Collapses integer ids into inclusive ``(first, last)`` runs."""
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if ids.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(ids) != 1)
    starts = np.concatenate([[ids[0]], ids[breaks + 1]])
    ends = np.concatenate([ids[breaks], [ids[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


class RowManifest:
    """Filters extracted tables to the rows recorded in a manifest file."""

    def __init__(self, path: str):
        self.path = str(path)
        with open(Path(path).expanduser()) as f:
            self.tables: Dict[str, Dict[str, Any]] = json.load(f)["tables"]

    def apply(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Keeps the manifest's rows of ``table_name`` and checks they are unchanged."""
        spec = self.tables.get(table_name)
        if spec is None:
            log.warning(f"Row manifest {self.path} does not pin table '{table_name}'; using all {len(df)} rows")
            return df

        ids = df["id"].to_numpy()
        keep = np.zeros(len(df), dtype=bool)
        for first, last in spec["id_ranges"]:
            keep |= (ids >= first) & (ids <= last)
        out = df[keep]

        if len(out) != spec["rows"]:
            log.warning(
                f"Table '{table_name}': manifest lists {spec['rows']} rows, the database has {len(out)} of them "
                "(rows were deleted); results will differ from the pinned experiment"
            )
        elif table_digest(out) != spec["sha256"]:
            log.warning(
                f"Table '{table_name}': pinned rows have changed since the manifest was written; "
                "results will differ from the pinned experiment"
            )
        else:
            log.info(f"Table '{table_name}': {len(out)} pinned rows match the manifest")
        return out


def load_row_manifest(path: Optional[str]) -> Optional[RowManifest]:
    return RowManifest(path) if path else None
