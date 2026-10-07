"""Download a local snapshot of the database tables the pipeline reads.

Each table is written as <out_dir>/<table>.pkl through DatabaseService's own cache (raw and
unfiltered, so the snapshot holds for any participant list or cohort setting), plus a
manifest.json with row and user counts. Runs then read it instead of opening an SSH tunnel:

    LEMURS_DB_CACHE_DIR=<out_dir> sbatch --export=ALL scripts/parallel_sweep.sbatch <sweep_id>

Run on a compute node (sbatch scripts/make_db_snapshot.sbatch), not the login node:
    .venv/bin/python scripts/make_db_snapshot.py data/db_snapshots/<date>
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

TABLES = ["survey_response", "answer", "question", "demographic", "step", "distance", "calorie"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tables", nargs="+", default=TABLES)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists() and any(out.glob("*.pkl")):
        sys.exit(f"{out} already holds a snapshot; pick a new directory")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)  # participant health data
    os.environ["LEMURS_DB_CACHE_DIR"] = str(out)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils.database_service import DatabaseService

    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"), "tables": {}}
    with DatabaseService() as db:
        for table in args.tables:
            df = db.extract_from_database(table)
            info = {"rows": len(df), "columns": len(df.columns)}
            if "app_user_id" in df.columns:
                info["users"] = int(df["app_user_id"].nunique())
            for col in ("start_timestamp", "timestamp", "created_at"):
                if col in df.columns and len(df):
                    info["time_range"] = [str(df[col].min()), str(df[col].max())]
                    break
            manifest["tables"][table] = info
            print(f"{table:16s} {info}", flush=True)
    for p in out.glob("*.pkl"):
        p.chmod(0o600)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"snapshot written to {out}")


if __name__ == "__main__":
    main()
