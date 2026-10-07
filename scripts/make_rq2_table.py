"""Regenerate the RQ2 comprehensive table (tab:rq2_comprehensive) from the sweep results CSV.

Aggregation is the same as notebooks/paper_figures_and_tables.py: runs on the selected modality
combination (step_distance), grouped by model and featurizer, reporting mean and SD (ddof=1) of each
pooled metric across the runs, i.e. across the 20 lookback x bin-size settings.

Rows follow the layout of the table in working_paper/sections/results.tex. The Foundation block holds
TabPFN-3.5 and TiRex, frozen and fine-tuned. Bold marks the best value per column within each block
(RQ2.1 / RQ2.2); a dagger marks the best pipeline per category by balanced accuracy.

With --check <results.tex>, every number of the existing table's non-foundation rows is compared with
the regenerated value (3 decimals) and mismatches are listed.

Usage:
    .venv/bin/python scripts/make_rq2_table.py [--csv reports/sweep_results/all_sweeps_results.csv]
        [--out reports/tables/rq2_comprehensive.tex] [--check working_paper/sections/results.tex]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

METRICS = ["pooled_auroc", "pooled_f1", "pooled_balanced_accuracy", "pooled_precision", "pooled_recall", "pooled_specificity"]
HEADERS = ["AUC", "F1", "Bal.\\ Acc.", "Prec.", "Recall (Sens.)", "Spec."]

# (block, category, label, experiment, model_pkg, featurizer_pkg)
ROWS = [
    ("2.1", "Tabular-based", "Logistic reg.", "RQ1_flaml_models", "flaml_lr", "raw"),
    ("2.1", "Tabular-based", "SVM", "RQ1_flaml_models", "flaml_svm", "raw"),
    ("2.1", "Tabular-based", "Random forest", "RQ1_flaml_models", "flaml_rf", "raw"),
    ("2.1", "Tabular-based", "XGBoost", "RQ1_flaml_models", "flaml_xgboost", "raw"),
    ("2.1", "Sequence-based", "LSTM", "RQ1_dl_models", "lstm", None),
    ("2.1", "Sequence-based", "Transformer", "RQ1_dl_models", "transformer", None),
    ("2.1", "Foundation", "TabPFN-3.5 (frozen)", "RQ1_FM_tabpfn", "flaml_tabpfn", "raw"),
    ("2.1", "Foundation", "TabPFN-3.5 (fine-tuned)", "RQ1_FM_tabpfn", "flaml_tabpfn_ft", "raw"),
    ("2.1", "Foundation", "TiRex (frozen)", "RQ1_FM_tirex", "tirex", None),
    ("2.1", "Foundation", "TiRex (fine-tuned)", "RQ1_FM_tirex_ft", "tirex_ft", None),
]
for _model, _name in (("flaml_lr", "Logistic reg."), ("flaml_svm", "SVM"), ("flaml_rf", "Random forest"), ("flaml_xgboost", "XGBoost")):
    ROWS += [("2.2", _name, "+ Summary Stats", "RQ2_flaml_features", _model, "summary_stats"),
             ("2.2", _name, "+ Catch22", "RQ2_flaml_features", _model, "catch22")]

# Daggers as in the paper (best per category by balanced accuracy, selected for RQ3); the Foundation
# dagger is computed. Tabular's dagger sits on XGBoost + Summary Stats in the RQ2.2 block.
DAGGER_FIXED = {("2.1", "Sequence-based"), ("2.2", "XGBoost")}


def aggregate(df, modality):
    sub = df[df["modalities_choice"] == modality]
    out = []
    for block, cat, label, exp, model, feat in ROWS:
        g = sub[(sub["experiment"] == exp) & (sub["model_pkg"] == model)]
        if feat is not None:
            g = g[g["featurizer_pkg"] == feat]
        rec = {"block": block, "category": cat, "label": label, "n": int(g["pooled_auroc"].notna().sum())}
        for m in METRICS:
            rec[m] = (g[m].mean(), g[m].std(ddof=1))
        out.append(rec)
    return out


def daggers(rows):
    marks = set()
    best = {}
    for i, r in enumerate(rows):
        key = (r["block"], r["category"])
        if key == ("2.1", "Tabular-based"):
            continue  # the paper selects the featurized XGBoost (RQ2.2) as the tabular pipeline
        if key[0] == "2.2" and key not in DAGGER_FIXED:
            continue
        if key not in best or r["pooled_balanced_accuracy"][0] > rows[best[key]]["pooled_balanced_accuracy"][0]:
            best[key] = i
    return set(best.values())


def fmt(v, bold):
    m, s = v
    m_txt = f"\\textbf{{{m:.3f}}}" if bold else f"{m:.3f}"
    return f"\\ms{{{m_txt}}}{{{s:.3f}}}"


def to_latex(rows):
    best_col = {}
    for block in ("2.1", "2.2"):
        for m in METRICS:
            vals = [(r[m][0], i) for i, r in enumerate(rows) if r["block"] == block]
            best_col[(block, m)] = max(vals)[1]
    dag = daggers(rows)
    lines = [
        "\\providecommand{\\ms}[2]{#1{\\scriptsize$\\pm$#2}}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\caption{RQ2: Performance across model categories and featurization schemes on the selected modality "
        "combination (S+D: steps + distance). Values are mean{\\scriptsize$\\pm$}std over 5 cross-validation folds. "
        "$\\dagger$ denotes the best performing pipeline per category, selected for RQ3.}",
        "\\label{tab:rq2_comprehensive}",
        "\\begin{tabular}{ll cccccc}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Model / Pipeline} & " + " & ".join(f"\\textbf{{{h}}}" for h in HEADERS) + " \\\\",
        "\\midrule",
    ]
    titles = {"2.1": "RQ2.1: Base models (raw input)", "2.2": "RQ2.2: Tabular featurization (summary statistics vs.\\ catch22)"}
    for block in ("2.1", "2.2"):
        if block == "2.2":
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{titles[block]}}}}} \\\\")
        cats = list(dict.fromkeys(r["category"] for r in rows if r["block"] == block))
        for ci, cat in enumerate(cats):
            idx = [i for i, r in enumerate(rows) if r["block"] == block and r["category"] == cat]
            for j, i in enumerate(idx):
                r = rows[i]
                first = f"\\multirow{{{len(idx)}}}{{*}}{{{cat}}}\n" if j == 0 else ""
                label = r["label"] + ("$^\\dagger$" if i in dag else "")
                cells = " & ".join(fmt(r[m], best_col[(block, m)] == i) for m in METRICS)
                lines.append(f"{first}  & {label} & {cells} \\\\")
            if ci < len(cats) - 1:
                lines.append("\\cmidrule(lr){1-8}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def parse_existing(tex_path):
    """Numbers per row of the existing tab:rq2_comprehensive, keyed by (block, category, label)."""
    tex = Path(tex_path).read_text()
    body = tex[tex.index("\\label{tab:rq2_comprehensive}"):]
    body = body[: body.index("\\end{table*}")]
    out, block, cat = {}, None, None
    for line in body.splitlines():
        if "RQ2.1" in line:
            block = "2.1"
        elif "RQ2.2" in line:
            block = "2.2"
        m = re.search(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", line)
        if m:
            cat = m.group(1)
            continue
        if "\\ms{" not in line:
            continue
        label = line.split("&")[1].strip().replace("$^\\dagger$", "")
        pairs = re.findall(r"\\ms\{(?:\\textbf\{)?([0-9.]+)\}?\}\{([0-9.]+)\}", line)
        out[(block, cat, label)] = [(float(a), float(b)) for a, b in pairs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/sweep_results/all_sweeps_results.csv")
    ap.add_argument("--modality", default="step_distance")
    ap.add_argument("--out", default="reports/tables/rq2_comprehensive.tex")
    ap.add_argument("--check", default=None, help="results.tex holding the existing table")
    args = ap.parse_args()

    rows = aggregate(pd.read_csv(args.csv), args.modality)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_latex(rows))
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['block']} {r['category']:15s} {r['label']:26s} n={r['n']:2d}  " +
              "  ".join(f"{r[m][0]:.3f}±{r[m][1]:.3f}" for m in METRICS))

    if args.check:
        existing = parse_existing(args.check)
        checked, bad = 0, []
        for r in rows:
            key = (r["block"], r["category"], r["label"])
            if key not in existing or r["category"] == "Foundation":
                continue
            for m, (em, es) in zip(METRICS, existing[key]):
                checked += 2
                gm, gs = round(r[m][0], 3), round(r[m][1], 3)
                if abs(gm - em) > 1e-9 or abs(gs - es) > 1e-9:
                    bad.append(f"{key} {m}: paper {em:.3f}±{es:.3f} vs regenerated {gm:.3f}±{gs:.3f}")
        missing = [k for k in existing if k[1] != "Foundation" and k not in {(r['block'], r['category'], r['label']) for r in rows}]
        print(f"\ncheck against {args.check}: {checked} numbers compared, {len(bad)} differ; unmatched rows: {missing or 'none'}")
        for b in bad:
            print("  DIFF", b)


if __name__ == "__main__":
    main()
