"""Characterises the temporal granularity of the passive sensing records.

HealthKit / Health Connect records are intervals whose lengths range from a few
seconds to many hours. Their length bounds how much within-day timing any
sampler or pretrained encoder can recover, so this script quantifies it on the
same cleaned cohort the models train on (same preprocessors, enrollment windows
and 99th-percentile clipping as ``CohortBuilder``).

Outputs (``--out``, default ``reports/data_granularity``):
    duration_by_source.csv     record-duration buckets: share of records and of value
    user_summary.csv           per-user effective resolution, coverage, overlap
    bin_misallocation.csv      value RollingSampler assigns outside its true bin, per Δt
    source_overlap.csv         minutes covered by >1 source (possible double counting)
    duration_hist.png          value-weighted record-duration histogram by source
    minute_heatmaps.png        minute-level step grids for sample users
    summary.md                 headline numbers

Usage:
    .venv/bin/python scripts/analyze_record_granularity.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.components.cohort_builder import CohortBuilder
from src.data.components.label_aggregators import RuleBasedAggregator
from src.data.components.preprocessing import (
    CaloriePreprocessor,
    DistancePreprocessor,
    StepPreprocessor,
)
from src.data.components.samplers import MinuteGridSampler

MODALITY_COLS = {"step": "steps", "distance": "distance", "calorie": "calories"}
# (label, lo, hi) in minutes, half-open [lo, hi); "point" is exactly zero length.
DURATION_BUCKETS = [
    ("point", 0.0, 0.0),
    ("<1m", 0.0, 1.0),
    ("1-10m", 1.0, 10.0),
    ("10-60m", 10.0, 60.0),
    ("1-3h", 60.0, 180.0),
    ("3-6h", 180.0, 360.0),
    ("6-12h", 360.0, 720.0),
    (">=12h", 720.0, np.inf),
]
BIN_HOURS = [1, 4, 6, 8, 12]


def source_group(src: str) -> str:
    s = str(src)
    if s == "iPhone":
        return "iPhone"
    if s == "Apple Watch":
        return "Apple Watch"
    if s == "HealthKit":
        return "HealthKit (other)"
    if "androidx" in s:
        return "Android (Health Connect)"
    return "Other app"


def user_os(modality_dfs) -> pd.Series:
    """Mirrors CohortBuilder._apply_os_filtering's source-string rule."""
    sources = pd.concat(
        [df[["app_user_id", "app_source"]] for df in modality_dfs.values()]
    ).dropna()
    out = {}
    for uid, g in sources.groupby("app_user_id"):
        s = g["app_source"].astype(str)
        ios = s.str.contains("iPhone|Apple Watch|HealthKit").any()
        android = s.str.contains("androidx").any()
        out[uid] = "ios" if ios and not android else ("android" if android and not ios else "both")
    return pd.Series(out, name="os")


def bucket(minutes: pd.Series) -> pd.Series:
    labels = pd.Series("point", index=minutes.index)
    for name, lo, hi in DURATION_BUCKETS[1:]:
        labels[(minutes > 0) & (minutes >= lo) & (minutes < hi)] = name
    return labels


def prepare(df: pd.DataFrame, mod: str, clip_limit: float) -> pd.DataFrame:
    df = df.copy()
    df["value"] = df[MODALITY_COLS[mod]].astype(float)
    df["minutes"] = (df["end_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 60.0
    df = df[df["minutes"] >= 0]
    df["bucket"] = bucket(df["minutes"])
    df["source"] = df["app_source"].map(source_group)
    df["clipped"] = df["value"] >= clip_limit
    return df


def duration_table(records: dict) -> pd.DataFrame:
    rows = []
    for mod, df in records.items():
        for src, g in list(df.groupby("source")) + [("ALL", df)]:
            total_n, total_v = len(g), g["value"].sum()
            for name, _, _ in DURATION_BUCKETS:
                b = g[g["bucket"] == name]
                rows.append(
                    {
                        "modality": mod,
                        "source": src,
                        "bucket": name,
                        "n_records": len(b),
                        "pct_records": 100 * len(b) / max(total_n, 1),
                        "pct_value": 100 * b["value"].sum() / max(total_v, 1e-9),
                        "pct_clipped_in_bucket": 100 * b["clipped"].mean() if len(b) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, cw[-1] / 2)])


def misallocation(df: pd.DataFrame) -> pd.DataFrame:
    """Share of value RollingSampler places outside the bin(s) it truly spans.

    RollingSampler floors each record's ``start_timestamp`` to the bin width and
    credits its whole value there. Its bins are clock-aligned multiples of Δt
    (the window end is floored to Δt), so the correctly allocated mass of a
    record is its overlap with the bin containing its start.
    """
    rows = []
    s = df["start_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    e = df["end_timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    v = df["value"].values
    dur = np.maximum(e - s, 0)
    for h in BIN_HOURS:
        w = int(h * 3600 * 1e9)
        bin_end = (s // w + 1) * w
        inside = np.where(dur > 0, np.clip((np.minimum(e, bin_end) - s) / np.maximum(dur, 1), 0, 1), 1.0)
        crosses = e > bin_end
        rows.append(
            {
                "bin_hours": h,
                "pct_records_crossing_boundary": 100 * crosses.mean(),
                "pct_value_misallocated": 100 * (v * (1 - inside)).sum() / max(v.sum(), 1e-9),
            }
        )
    return pd.DataFrame(rows)


def day_grids(df: pd.DataFrame, users, days_by_user) -> dict:
    """Minute-level calendar-day grids per (user, date) using MinuteGridSampler."""
    sampler = MinuteGridSampler(lookback_hours=24, align_freq="1D")
    grids = {}
    for uid in users:
        for day in days_by_user.get(uid, []):
            anchor = pd.Timestamp(day) + pd.Timedelta(days=1)
            out = sampler(anchor, uid, {"m": df}, {"m": "value"}, ["m"])
            grids[(uid, day)] = out
    return grids


def source_overlap(df: pd.DataFrame, days_by_user) -> pd.DataFrame:
    """Minutes covered by more than one source group on the same user-day."""
    sampler = MinuteGridSampler(lookback_hours=24, align_freq="1D")
    rows = []
    for uid, days in days_by_user.items():
        u = df[df["app_user_id"] == uid]
        srcs = u["source"].unique()
        if len(srcs) < 2:
            continue
        for day in days:
            anchor = pd.Timestamp(day) + pd.Timedelta(days=1)
            covered = []
            totals = {}
            for src in srcs:
                g = u[u["source"] == src]
                out = sampler(anchor, uid, {"m": g}, {"m": "value"}, ["m"])
                covered.append(out[:, 1])
                totals[src] = out[:, 0].sum()
            n_cov = np.sum(covered, axis=0)
            if (n_cov > 0).sum() == 0:
                continue
            rows.append(
                {
                    "app_user_id": uid,
                    "date": day,
                    "minutes_any_source": int((n_cov > 0).sum()),
                    "minutes_multi_source": int((n_cov > 1).sum()),
                    **{f"total_{k}": v for k, v in totals.items()},
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/data_granularity")
    ap.add_argument("--n-heatmap-users", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builder = CohortBuilder(
        modalities=list(MODALITY_COLS),
        modality_cols=MODALITY_COLS,
        preprocessors={
            "step": StepPreprocessor(),
            "calorie": CaloriePreprocessor(),
            "distance": DistancePreprocessor(),
        },
        aggregator=RuleBasedAggregator(
            combination_logic="any",
            rules={
                "rule_1": {"ids": [2, 3, 5, 7], "op": "ge", "val": 2.0},
                "rule_2": {"ids": [8, 12, 13], "op": "any_gt", "val": 0},
            },
        ),
        collapse_strategy="none",
    )
    modality_dfs, master_df, _ = builder.build()
    os_map = user_os(modality_dfs)

    records = {}
    for mod, df in modality_dfs.items():
        clip_limit = df[MODALITY_COLS[mod]].max()  # values at the max were clipped at p99
        records[mod] = prepare(df, mod, clip_limit)

    # Study days per user: first to last labelled survey date.
    days_by_user = {}
    for uid, g in master_df.groupby("app_user_id"):
        first, last = g["record_timestamp"].min().normalize(), g["record_timestamp"].max().normalize()
        days_by_user[uid] = list(pd.date_range(first, last, freq="D"))

    # 1. Duration buckets by source.
    dur = duration_table(records)
    dur.to_csv(out / "duration_by_source.csv", index=False)

    # 2. Per-user summary (steps).
    steps = records["step"]
    grids = day_grids(steps, list(days_by_user), days_by_user)
    user_rows = []
    for uid, g in steps.groupby("app_user_id"):
        if uid not in days_by_user:
            continue
        uday = [grids[(uid, d)] for d in days_by_user[uid] if (uid, d) in grids]
        cov = np.array([x[:, 1].mean() for x in uday]) if uday else np.array([np.nan])
        days_with_data = int(np.sum([x[:, 1].any() for x in uday])) if uday else 0
        user_rows.append(
            {
                "app_user_id": uid,
                "os": os_map.get(uid, "?"),
                "sources": ", ".join(sorted(g["source"].unique())),
                "n_records": len(g),
                "study_days": len(days_by_user[uid]),
                "days_with_any_step_record": days_with_data,
                "records_per_day_with_data": len(g) / max(days_with_data, 1),
                "median_record_min": g["minutes"].median(),
                "value_weighted_median_record_min": weighted_median(g["minutes"].values, g["value"].values),
                "pct_steps_in_records_ge_1h": 100 * g.loc[g["minutes"] >= 60, "value"].sum() / max(g["value"].sum(), 1e-9),
                "pct_steps_in_records_ge_6h": 100 * g.loc[g["minutes"] >= 360, "value"].sum() / max(g["value"].sum(), 1e-9),
                "median_pct_day_minutes_covered": 100 * np.nanmedian(cov),
            }
        )
    users = pd.DataFrame(user_rows).sort_values(["os", "value_weighted_median_record_min"])
    users.to_csv(out / "user_summary.csv", index=False)

    # 3. RollingSampler misallocation per Δt, per modality and OS.
    mis_rows = []
    for mod, df in records.items():
        df = df.assign(os=df["app_user_id"].map(os_map))
        for os_name, g in list(df.groupby("os")) + [("ALL", df)]:
            m = misallocation(g)
            m.insert(0, "os", os_name)
            m.insert(0, "modality", mod)
            mis_rows.append(m)
    mis = pd.concat(mis_rows, ignore_index=True)
    mis.to_csv(out / "bin_misallocation.csv", index=False)

    # 4. Overlapping sources (e.g. iPhone + Apple Watch both logging steps).
    ovl = source_overlap(steps, days_by_user)
    ovl.to_csv(out / "source_overlap.csv", index=False)

    # 5. Figures.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    bins = np.logspace(-2, np.log10(24 * 60 * 2), 50)
    for ax, (mod, df) in zip(axes, records.items()):
        for src, g in df[df["minutes"] > 0].groupby("source"):
            ax.hist(g["minutes"], bins=bins, weights=g["value"] / df["value"].sum() * 100, histtype="step", lw=1.6, label=src)
        ax.set_xscale("log")
        for x, lbl in [(1, "1m"), (60, "1h"), (360, "6h"), (720, "12h")]:
            ax.axvline(x, color="grey", ls=":", lw=0.8)
            ax.text(x, ax.get_ylim()[1] * 0.95, lbl, fontsize=8, color="grey")
        ax.set_title(f"{mod}: share of total value by record length")
        ax.set_xlabel("record duration (minutes, log)")
    axes[0].set_ylabel("% of modality total")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "duration_hist.png", dpi=140)
    plt.close(fig)

    # Heatmaps: pick users spanning OS and effective resolution.
    pick = []
    for os_name, g in users.groupby("os"):
        g = g.sort_values("value_weighted_median_record_min")
        idx = np.linspace(0, len(g) - 1, min(len(g), max(2, args.n_heatmap_users // 2))).astype(int)
        pick += g.iloc[np.unique(idx)]["app_user_id"].tolist()
    pick = pick[: args.n_heatmap_users]
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 2.2 * len(pick)), squeeze=False)
    for ax, uid in zip(axes[:, 0], pick):
        mats = [grids[(uid, d)][:, 0] for d in days_by_user[uid] if (uid, d) in grids]
        mat = np.log1p(np.vstack(mats)) if mats else np.zeros((1, 1440))
        ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
        row = users[users["app_user_id"] == uid].iloc[0]
        ax.set_title(
            f"user {uid} ({row['os']}; {row['sources']}) — value-weighted median record "
            f"{row['value_weighted_median_record_min']:.0f} min",
            fontsize=9,
        )
        ax.set_ylabel("study day")
        ax.set_xticks(np.arange(0, 1441, 120))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 2)], fontsize=7)
    axes[-1, 0].set_xlabel("minute of day (log1p steps/min)")
    fig.tight_layout()
    fig.savefig(out / "minute_heatmaps.png", dpi=120)
    plt.close(fig)

    # 6. Headline summary.
    def pct(mod, src, buckets, col="pct_value"):
        d = dur[(dur["modality"] == mod) & (dur["source"] == src) & (dur["bucket"].isin(buckets))]
        return d[col].sum()

    long_b = ["1-3h", "3-6h", "6-12h", ">=12h"]
    lines = ["# Record granularity summary", ""]
    lines.append(f"Users with labels: {master_df['app_user_id'].nunique()}; OS: {os_map.reindex(master_df['app_user_id'].unique()).value_counts().to_dict()}")
    lines.append("")
    lines.append("## Share of each modality's total value carried by records >= 1 hour")
    for mod, df in records.items():
        parts = [f"{src}: {pct(mod, src, long_b):.1f}%" for src in sorted(df["source"].unique())]
        lines.append(f"- **{mod}** (all: {pct(mod, 'ALL', long_b):.1f}%) — " + "; ".join(parts))
    lines.append("")
    lines.append("## RollingSampler: % of value assigned outside its true bin")
    piv = mis[mis["os"] == "ALL"].pivot(index="modality", columns="bin_hours", values="pct_value_misallocated")
    lines.append("```\n" + piv.round(1).to_string() + "\n```")
    lines.append("")
    if not ovl.empty:
        frac = ovl["minutes_multi_source"].sum() / max(ovl["minutes_any_source"].sum(), 1)
        lines.append(
            f"## Source overlap (steps)\n{ovl['app_user_id'].nunique()} users have >1 step source; "
            f"{100 * frac:.1f}% of their covered minutes are covered by 2+ sources."
        )
        lines.append("")
    lines.append("## Per-user step granularity")
    cols = ["app_user_id", "os", "sources", "value_weighted_median_record_min", "pct_steps_in_records_ge_1h",
            "pct_steps_in_records_ge_6h", "days_with_any_step_record", "study_days", "median_pct_day_minutes_covered"]
    lines.append("```\n" + users[cols].round(1).to_string(index=False) + "\n```")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    main()
