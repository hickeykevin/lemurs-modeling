#!/bin/bash
#SBATCH --job-name=db_snapshot
#SBATCH --output=logs/db_snapshot_%j.log
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Downloads the database tables once into a local snapshot (see scripts/make_db_snapshot.py).
# Usage: sbatch scripts/make_db_snapshot.sbatch [out_dir]   (default data/db_snapshots/<today>)
set -eo pipefail
cd /home/khickey/lemurs-modeling
OUT=${1:-data/db_snapshots/$(date +%F)}
mkdir -p data/db_snapshots && chmod 700 data/db_snapshots
.venv/bin/python scripts/make_db_snapshot.py "$OUT"
