"""Descriptive analytics of the suicide-risk label: per user, and over the study timeline.

Answers two questions:
  1. Per user: how many surveys are at risk vs not at risk? Is each user's label
     100% homogeneous (always one state), and how often do they switch?
  2. How are the states distributed over the ~4-week timeline?

Two survey sets:
  * modeling cohort: the surveys that become model samples (those with sensor
    coverage), i.e. the "chunks" -- read from the cyclical XGBoost run's test
    predictions, which cover every cohort survey exactly once;
  * all daily surveys: every morning/afternoon survey after duplicate removal.

Labels are built exactly as the pipeline builds them (suicide_risk aggregator,
collapse_strategy="none"). Study day 1 is the calendar day of the user's first
daily survey; week = ceil(day / 7).

Outputs go to reports/label_descriptives/.

Usage (compute node):
    PYTHONPATH=. .venv/bin/python scripts/verify_between_person/label_descriptives.py
"""

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.components.cohort_builder import CohortBuilder  # noqa: E402
from src.utils.database_service import DatabaseService  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "configs" / "data" / "aggregator"
COHORT = REPO / "logs" / "verify_between_person" / "runs" / "xgboost" / "pooled_predictions.csv"
OUT = REPO / "reports" / "label_descriptives"

# Chart colours (dataviz reference palette): at risk = categorical slot 2, not at risk = neutral.
AT_RISK, NOT_AT_RISK = "#eb6834", "#c9c8c2"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def build_labels(db) -> pd.DataFrame:
    """Every daily survey with suicide_risk and its two components."""
    frames = {}
    for name in ("suicide_risk", "suicide_ideation", "suicide_preparation"):
        agg = hydra.utils.instantiate(OmegaConf.load(AGG / f"{name}.yaml"))
        b = CohortBuilder(modalities=[], modality_cols={}, preprocessors=None, aggregator=agg, collapse_strategy="none")
        df = b._extract_and_aggregate_labels(db)
        frames[name] = df[["survey_response_id", "app_user_id", "record_timestamp", "survey_id", "answer"]].rename(
            columns={"answer": name}
        )
    out = frames["suicide_risk"]
    for name in ("suicide_ideation", "suicide_preparation"):
        out = out.merge(frames[name][["survey_response_id", name]], on="survey_response_id", how="left")
    out["record_timestamp"] = pd.to_datetime(out.record_timestamp)
    out = out.sort_values(["app_user_id", "record_timestamp"]).reset_index(drop=True)
    first_day = out.groupby("app_user_id").record_timestamp.transform("min").dt.normalize()
    out["study_day"] = (out.record_timestamp.dt.normalize() - first_day).dt.days + 1
    out["week"] = np.ceil(out.study_day / 7).astype(int)
    return out


def per_user(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in df.groupby("app_user_id"):
        y = g.suicide_risk.astype(int).to_numpy()
        prev = np.r_[np.nan, y[:-1]]
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, y, 0]) != 0))[::2] if y.any() else np.array([0])
        rows.append(dict(
            app_user_id=uid,
            surveys=len(y),
            at_risk=int(y.sum()),
            not_at_risk=int(len(y) - y.sum()),
            pct_at_risk=y.mean(),
            pattern="always not at risk" if y.sum() == 0 else ("always at risk" if y.all() else "mixed"),
            onsets=int(((prev == 0) & (y == 1)).sum()),
            recoveries=int(((prev == 1) & (y == 0)).sum()),
            at_risk_episodes=int(len(runs)) if y.any() else 0,
            longest_at_risk_run=int(runs.max()),
            ideation_surveys=int(g.suicide_ideation.sum()),
            preparation_surveys=int(g.suicide_preparation.sum()),
            first_day=int(g.study_day.min()),
            last_day=int(g.study_day.max()),
        ))
    return pd.DataFrame(rows).sort_values(["pct_at_risk", "surveys"], ascending=[False, False]).reset_index(drop=True)


def summary(df: pd.DataFrame, users: pd.DataFrame) -> dict:
    pos = users.sort_values("at_risk", ascending=False).at_risk.to_numpy()
    total = pos.sum()
    return {
        "surveys": len(df),
        "users": len(users),
        "at-risk surveys": f"{int(df.suicide_risk.sum())} ({df.suicide_risk.mean():.1%})",
        "users always not at risk": int((users.pattern == "always not at risk").sum()),
        "users always at risk": int((users.pattern == "always at risk").sum()),
        "users with both states": int((users.pattern == "mixed").sum()),
        "users with any at-risk survey": int((users.at_risk > 0).sum()),
        "onsets (not at risk -> at risk)": int(users.onsets.sum()),
        "recoveries (at risk -> not at risk)": int(users.recoveries.sum()),
        "share of at-risk surveys from top 3 users": f"{pos[:3].sum() / total:.0%}" if total else "-",
        "share of at-risk surveys from top 5 users": f"{pos[:5].sum() / total:.0%}" if total else "-",
        "surveys per user (median, range)": f"{int(users.surveys.median())} ({users.surveys.min()}-{users.surveys.max()})",
    }


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["prev"] = d.groupby("app_user_id").suicide_risk.shift(1)
    g = d.groupby("week")
    out = pd.DataFrame({
        "surveys": g.size(),
        "users": g.app_user_id.nunique(),
        "at_risk_surveys": g.suicide_risk.sum().astype(int),
        "pct_at_risk": g.suicide_risk.mean(),
        "users_with_any_at_risk": g.apply(lambda x: x.loc[x.suicide_risk == 1, "app_user_id"].nunique()),
        "onsets": g.apply(lambda x: int(((x.prev == 0) & (x.suicide_risk == 1)).sum())),
    })
    # Rate among only the users who are ever at risk (removes the always-negative majority).
    ever = d.groupby("app_user_id").suicide_risk.transform("max") == 1
    out["surveys_ever_at_risk_users"] = d[ever].groupby("week").size()
    out["pct_at_risk_among_ever_at_risk_users"] = d[ever].groupby("week").suicide_risk.mean()
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def plot_timeline(df: pd.DataFrame, users: pd.DataFrame, path: Path, title: str):
    order = users.app_user_id.tolist()
    ypos = {u: i for i, u in enumerate(order)}
    d = df.assign(y=df.app_user_id.map(ypos), x=df.study_day + np.where(df.survey_id == 1, 0.25, -0.25))
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(order) + 1.6), dpi=150)
    for state, color, z in ((0, NOT_AT_RISK, 2), (1, AT_RISK, 3)):
        s = d[d.suicide_risk == state]
        ax.scatter(s.x, s.y, s=22, color=color, edgecolors="white", linewidths=0.6, zorder=z)
    for w in (7.5, 14.5, 21.5, 28.5):
        ax.axvline(w, color=GRID, lw=1, zorder=1)
    labels = [f"{u}  ({r.at_risk}/{r.surveys})" for u, r in zip(order, users.itertuples())]
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Study day (day 1 = first daily survey); left dot = morning, right dot = afternoon")
    ax.set_ylabel("Participant  (at-risk / total surveys)")
    ax.set_xlim(0.3, max(29.0, d.study_day.max() + 0.7))
    style(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=AT_RISK, label="At risk"),
        Line2D([], [], marker="o", ls="", color=NOT_AT_RISK, label="Not at risk"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.9 / (0.24 * len(order) + 1.6)), ncol=2, frameon=False,
        fontsize=9, labelcolor=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weekly(wk: pd.DataFrame, path: Path, title: str):
    wk = wk[wk.index <= 5]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=150, sharey=True)
    for ax, col, ncol, sub in (
        (axes[0], "pct_at_risk", "surveys", "All participants"),
        (axes[1], "pct_at_risk_among_ever_at_risk_users", "surveys_ever_at_risk_users", "Only participants ever at risk"),
    ):
        bars = ax.bar(wk.index, wk[col] * 100, width=0.6, color=AT_RISK, zorder=2)
        for b, (w, r) in zip(bars, wk.iterrows()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{r[col]:.0%}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(wk.index, [
            f"{'Days 29+' if w == 5 else f'Week {w}'}\n{int(r[ncol])} surveys" for w, r in wk.iterrows()
        ])
        ax.set_title(sub, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, zorder=0)
        style(ax)
    axes[0].set_ylabel("% of surveys at risk")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    """Markdown table; pct_* columns as percentages, everything else as plain values."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.astype(object).itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if c.startswith("pct"):
                cells.append("—" if pd.isna(v) else f"{v:.0%}")
            elif isinstance(v, float) and v.is_integer():
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = DatabaseService()
    assert db.connect()
    try:
        all_surveys = build_labels(db)
    finally:
        db.disconnect()

    cohort_keys = pd.read_csv(COHORT)
    cohort_keys = cohort_keys[cohort_keys.stage == "test"][["app_user_id", "record_timestamp"]]
    cohort_keys["record_timestamp"] = pd.to_datetime(cohort_keys.record_timestamp).dt.floor("s")
    a = all_surveys.assign(ts_s=all_surveys.record_timestamp.dt.floor("s"))
    cohort = a.merge(cohort_keys.rename(columns={"record_timestamp": "ts_s"}), on=["app_user_id", "ts_s"]).drop(columns="ts_s")
    assert len(cohort) == len(cohort_keys), f"matched {len(cohort)} of {len(cohort_keys)} cohort surveys"

    report = ["# Suicide-risk label: per-user and timeline descriptives", ""]
    for name, df, tag in (("Modeling cohort (model samples)", cohort, "cohort"),
                          ("All daily surveys", all_surveys, "all")):
        users = per_user(df)
        wk = weekly(df)
        users.to_csv(OUT / f"per_user_{tag}.csv", index=False)
        wk.to_csv(OUT / f"weekly_{tag}.csv")
        plot_timeline(df, users, OUT / f"timeline_{tag}.png", f"Suicide-risk state per survey, {name.lower()}")
        plot_weekly(wk, OUT / f"weekly_{tag}.png", f"Share of surveys at risk by study week, {name.lower()}")
        s = summary(df, users)
        report += [f"## {name}", "", "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s.items()] + [""]
        report += ["### By study week", "", md_table(wk.reset_index()), ""]
        report += ["### Per user (sorted by % at risk)", "", md_table(users), ""]
        print(f"done {name}", flush=True)
    (OUT / "label_descriptives.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
