#!/bin/bash
#SBATCH --job-name=gather_results
#SBATCH --output=logs/gather_results_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=short

mkdir -p logs reports/sweep_results
echo "Starting sweep results gathering for: 9dvq8uiy, a6sl4omd, zoznzd8y..."
uv run python scripts/gather_sweep_results.py
echo "Finished gathering results."
