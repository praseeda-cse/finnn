"""Analyse the FinSight user-study responses (Google Forms → Responses → Download CSV).

    python scripts/analyse_user_study.py responses.csv [--out user_study/results]

Columns are found by matching question text (see user_study/USER_STUDY.md), so the order of
columns in the export does not matter. Likert answers may be 1–5 or the text labels.
Prints a report and writes summary.json plus charts (PNG) to --out.
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}

SUS_ITEMS = [  # standard SUS wording; odd items positive, even items negative
    "would like to use this system frequently",
    "unnecessarily complex",
    "easy to use",
    "support of a technical person",
    "well integrated",
    "too much inconsistency",
    "learn to use this system very quickly",
    "very cumbersome",
    "felt very confident",
    "learn a lot of things",
]
TRUST_ITEMS = {
    "evidence_easy_to_check": "highlighted evidence made it easy",
    "equation_helped": "equation helped me understand",
    "unit_important": "showing the unit",
    "prefer_not_sure": "rather the system say",
    "would_check_page": "check the page myself",
    "trust_over_chatbot": "more than a general chatbot",
}
ATTENTION = "show you are reading carefully"
ROLE = "best describes you"
TRUST_A, TRUST_B, PREFER = "trust Answer A", "trust Answer B", "Which answer do you trust more"
TASK3_USED = "What did you use to decide"
FEATURES, NPS = "most useful to you", "recommend FinSight"
FINANCE_ROLES = ("Commerce", "MBA", "CA / CMA", "finance, accounts")


def col(df, needle, required=True):
    hits = [c for c in df.columns if needle.lower() in c.lower()]
    if not hits:
        if required:
            sys.exit(f"No column containing {needle!r}. Check the question text in the form.")
        return None
    return hits[0]


def likert(series):
    def one(v):
        if pd.isna(v):
            return None
        s = str(v).strip().lower()
        return LIKERT.get(s, pd.to_numeric(s.split()[0], errors="coerce"))
    return series.map(one).astype(float)


def sus_score(df):
    items = [likert(df[col(df, t)]) for t in SUS_ITEMS]
    total = sum((x - 1) if i % 2 == 0 else (5 - x) for i, x in enumerate(items))
    return total * 2.5


def pct(x):
    return round(100 * x, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="user_study/results")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    n_raw = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c.lower() != "timestamp"])
    att = col(df, ATTENTION, required=False)
    if att:
        df = df[likert(df[att]) == 4]
    n = len(df)
    print(f"Responses: {n_raw} received, {n} kept (dropped {n_raw - n}: duplicates or failed attention check)")
    if n < 30:
        print("  note: fewer than the recommended 30 participants")

    out = {"n_received": n_raw, "n_kept": n}

    roles = df[col(df, ROLE)].value_counts()
    out["roles"] = roles.to_dict()
    print("\nParticipants by role:\n" + roles.to_string())

    df["SUS"] = sus_score(df)
    out["sus_mean"], out["sus_sd"] = round(df.SUS.mean(), 1), round(df.SUS.std(), 1)
    grade = "excellent" if out["sus_mean"] >= 80.3 else "good" if out["sus_mean"] >= 68 else "below average"
    print(f"\nSUS: {out['sus_mean']} ± {out['sus_sd']} (68 = average; this is {grade})")

    a, b = likert(df[col(df, TRUST_A)]), likert(df[col(df, TRUST_B)])
    paired = pd.concat([a, b], axis=1).dropna()
    stat = wilcoxon(paired.iloc[:, 1], paired.iloc[:, 0]) if (paired.iloc[:, 1] != paired.iloc[:, 0]).any() else None
    out["trust_A_median"], out["trust_B_median"] = a.median(), b.median()
    out["trust_wilcoxon_p"] = None if stat is None else float(stat.pvalue)
    prefer_b = df[col(df, PREFER)].astype(str).str.startswith("B").mean()
    out["prefer_B_pct"] = pct(prefer_b)
    print(f"\nTrust, answer only (A) vs explained (B): median {a.median()} vs {b.median()}, "
          f"Wilcoxon p = {out['trust_wilcoxon_p']}; {out['prefer_B_pct']}% prefer B")

    print("\nTrust items (% agree = 4 or 5, median):")
    out["trust_items"] = {}
    for key, needle in TRUST_ITEMS.items():
        c = col(df, needle, required=False)
        if c:
            v = likert(df[c])
            out["trust_items"][key] = {"agree_pct": pct((v >= 4).mean()), "median": v.median()}
            print(f"  {key:24s} {out['trust_items'][key]['agree_pct']:5.1f}%  median {v.median()}")

    t3_cols = [c for c in df.columns if "Is the answer correct" in c]
    if len(t3_cols) >= 3:  # third "Is the answer correct?" is Task 3, where FinSight is wrong
        caught = df[t3_cols[2]].astype(str).str.startswith("Wrong")
        used = df[col(df, TASK3_USED)].astype(str)
        uses_ev = used.str.contains("highlighted|equation", case=False)
        out["task3_caught_pct"] = pct(caught.mean())
        out["task3_caught_with_evidence_pct"] = pct(caught[uses_ev].mean()) if uses_ev.any() else None
        out["task3_caught_without_evidence_pct"] = pct(caught[~uses_ev].mean()) if (~uses_ev).any() else None
        print(f"\nTask 3 (system wrong): {out['task3_caught_pct']}% caught the error; "
              f"{out['task3_caught_with_evidence_pct']}% of those who used evidence/equation vs "
              f"{out['task3_caught_without_evidence_pct']}% of those who did not")

    feats = df[col(df, FEATURES)].dropna().str.split(", ").explode().str.strip().value_counts()
    out["features"] = feats.to_dict()
    print("\nRequested features:\n" + feats.to_string())

    nps = pd.to_numeric(df[col(df, NPS)], errors="coerce").dropna()
    out["nps"] = round(100 * ((nps >= 9).mean() - (nps <= 6).mean()), 1)
    print(f"\nNet Promoter Score: {out['nps']}")

    fin = df[col(df, ROLE)].astype(str).str.contains("|".join(FINANCE_ROLES), regex=True)
    if fin.any() and (~fin).any():
        p = mannwhitneyu(df.SUS[fin].dropna(), df.SUS[~fin].dropna()).pvalue
        out["sus_finance"], out["sus_other"], out["sus_group_p"] = (
            round(df.SUS[fin].mean(), 1), round(df.SUS[~fin].mean(), 1), float(p))
        print(f"\nSUS finance vs other participants: {out['sus_finance']} vs {out['sus_other']} (Mann-Whitney p = {p:.3f})")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        items = out["trust_items"]
        ax.barh(list(items), [v["agree_pct"] for v in items.values()], color="#2B6CB0")
        ax.set_xlim(0, 100); ax.set_xlabel("% agree (4–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "trust_items.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(6, 3))
        feats.sort_values().plot.barh(ax=ax, color="#C05621"); ax.set_xlabel("participants"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "features.png"), dpi=200)
        print(f"\nCharts and summary.json written to {args.out}/")
    except ImportError:
        print(f"\nmatplotlib not installed: summary.json written to {args.out}/, no charts")


if __name__ == "__main__":
    main()
