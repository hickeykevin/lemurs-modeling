import json

import pandas as pd

from src.utils.row_manifest import RowManifest, id_ranges, table_digest


def _table(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(1, n + 1),
        "app_user_id": [5, 5, 6, 6, 7, 7][:n],
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "additional_data": [{"b": 1, "a": [1, 2]}] * n,
        "status": ["ok", None] * (n // 2),
    })


def _manifest(tmp_path, tables) -> RowManifest:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tables": tables}))
    return RowManifest(str(path))


def test_id_ranges_collapses_runs():
    assert id_ranges([1, 2, 3, 7, 8, 10]) == [(1, 3), (7, 8), (10, 10)]
    assert id_ranges([]) == []


def test_digest_ignores_row_order_and_json_key_order():
    df = _table()
    shuffled = df.sample(frac=1.0, random_state=0)
    shuffled["additional_data"] = [{"a": [1, 2], "b": 1}] * len(df)
    assert table_digest(df) == table_digest(shuffled)
    changed = df.copy()
    changed.loc[0, "status"] = "edited"
    assert table_digest(df) != table_digest(changed)


def test_apply_keeps_pinned_rows_and_drops_newer_ones(tmp_path):
    pinned = _table().iloc[:4]
    manifest = _manifest(tmp_path, {"step": {
        "id_ranges": id_ranges(pinned["id"].to_numpy()), "rows": 4, "sha256": table_digest(pinned),
    }})
    out = manifest.apply("step", _table())  # the live table has grown by two rows
    assert out["id"].tolist() == [1, 2, 3, 4]
    assert table_digest(out) == table_digest(pinned)


def test_apply_passes_through_unpinned_tables(tmp_path):
    manifest = _manifest(tmp_path, {})
    assert len(manifest.apply("calorie", _table())) == 6
