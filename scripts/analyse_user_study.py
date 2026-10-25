"""Analyse the FinSight survey responses (Google Forms → Responses → Download CSV).

    python scripts/analyse_user_study.py responses.csv [--out user_study/results]

Columns are found by matching question text (see user_study/USER_STUDY.md §4), so the order of
columns in the export does not matter. Reports overall and per group (learner, professional,
technical, other; investors as a separate cut) and writes summary.json plus charts to --out.
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import wilcoxon

ROLE, INVEST, CONFIDENT = "describes you best", "invest in stocks", "How confident are you"
TODAY = {"how_they_get_numbers": "how do you usually get it", "hardest": "What is hardest",
         "used_ai": "Have you used an AI tool"}
SPOT = "EMEA's total assets in 2019"          # correct response: Wrong
PREFER, TRUST_A, TRUST_B = "Which answer would you trust more", "trust Tool A", "trust Tool B"
VERIFY, MOST_USEFUL = "would you check the answer yourself", "Which part of Tool B"
ITEMS = {
    "rows_help_catch_mistakes": "help me catch mistakes",
    "calculation_helps_understand": "calculation would help me understand",
    "unit_important": "Showing the unit clearly",
    "prefer_not_sure": "rather the tool say",
    "prefer_offline": "runs on my own computer",
}
FEATURES, ADOPT = "Which features would be most useful", "how would you use it"
GROUPS = ["Learner", "Professional", "Technical", "Other"]


def group(role):
    r = str(role)
    if r.startswith("Student – Commerce") or r.startswith("Student – CA"):
        return "Learner"
    if r.startswith("Working professional – finance"):
        return "Professional"
    if r.startswith("Student – Engineering"):
        return "Technical"
    return "Other"


def col(df, needle, required=True):
    hits = [c for c in df.columns if needle.lower() in c.lower()]
    if not hits:
        if required:
            sys.exit(f"No column containing {needle!r}. Check the question text in the form.")
        return None
    return hits[0]


def num(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def counts(series):
    """Value counts; checkbox answers ("a, b") are split into one count per option."""
    return series.dropna().astype(str).str.split(", ").explode().str.strip().value_counts()


def pct(x):
    return round(100 * float(x), 1)


def per_group(df, values, fmt="{:.0f}%"):
    return "  ".join(f"{g}={fmt.format(values[df.group == g].mean())}" for g in GROUPS if (df.group == g).any())


def show_counts(title, series):
    c = counts(series)
    print(f"\n{title}")
    for k, v in c.items():
        print(f"  {v:3d}  {k}")
    return c.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="user_study/results")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    n_raw = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c.lower() != "timestamp"]).copy()
    n = len(df)
    print(f"Responses: {n_raw} received, {n} kept (dropped {n_raw - n} duplicates)")
    if n < 30:
        print("  note: fewer than the recommended 30 participants")
    out = {"n_received": n_raw, "n_kept": n}

    df["group"] = df[col(df, ROLE)].map(group)
    df["investor"] = df[col(df, INVEST)].astype(str).str.startswith("Yes")
    df["confident"] = num(df[col(df, CONFIDENT)]) >= 4
    out["groups"] = df.group.value_counts().to_dict()
    out["investors"] = int(df.investor.sum())
    print("Groups: " + ", ".join(f"{k} {v}" for k, v in out["groups"].items()) + f"; investors {out['investors']}")
    if (df.group == "Technical").mean() > 0.5:
        print("  note: more than half are engineering/IT; results mostly reflect non-target users")

    # RQ1: the problem today
    out["today"] = {k: show_counts(col(df, needle), df[col(df, needle)]) for k, needle in TODAY.items()}

    # RQ2: spotting a wrong answer with no source shown
    caught = df[col(df, SPOT)].astype(str).str.startswith("Wrong")
    accepted = df[col(df, SPOT)].astype(str).str.startswith("Correct")
    out["spot_caught_pct"], out["spot_accepted_wrong_pct"] = pct(caught.mean()), pct(accepted.mean())
    out["spot_caught_by_group"] = (df.assign(c=caught).groupby("group").c.mean() * 100).round(1).to_dict()
    print(f"\nWrong AI answer, no source shown: {out['spot_caught_pct']}% caught it, "
          f"{out['spot_accepted_wrong_pct']}% accepted it as correct")
    print(f"  caught by group: {per_group(df, caught * 100)}")
    print(f"  caught, confident readers (4–5) vs others: {pct(caught[df.confident].mean()) if df.confident.any() else '-'}% "
          f"vs {pct(caught[~df.confident].mean()) if (~df.confident).any() else '-'}%")

    # RQ3: trust, bare vs explained
    df["trust_A"], df["trust_B"] = num(df[col(df, TRUST_A)]), num(df[col(df, TRUST_B)])
    paired = df[["trust_A", "trust_B"]].dropna()
    differs = (paired.trust_B != paired.trust_A).any()
    out["trust_wilcoxon_p"] = float(wilcoxon(paired.trust_B, paired.trust_A).pvalue) if differs else None
    out["trust_A_mean"], out["trust_B_mean"] = round(df.trust_A.mean(), 2), round(df.trust_B.mean(), 2)
    out["prefer_B_pct"] = pct(df[col(df, PREFER)].astype(str).str.contains("Tool B").mean())
    print(f"\nTrust, Tool A (answer only) vs Tool B (calculation + source): mean {out['trust_A_mean']} vs "
          f"{out['trust_B_mean']} (Wilcoxon p = {out['trust_wilcoxon_p']}); {out['prefer_B_pct']}% prefer B")
    print(f"  trust in B by group: {per_group(df, df.trust_B, '{:.2f}')}")
    out["would_verify"] = show_counts("Would still check Tool B's answer:", df[col(df, VERIFY)])
    out["most_useful_part"] = show_counts("Most useful part of Tool B:", df[col(df, MOST_USEFUL)])

    # design choices
    print("\nOpinion items, % agree (4–5): overall | by group")
    out["items"] = {}
    for key, needle in ITEMS.items():
        agree = (num(df[col(df, needle)]) >= 4).astype(float)
        out["items"][key] = {"agree_pct": pct(agree.mean()),
                             "by_group": (df.assign(a=agree).groupby("group").a.mean() * 100).round(1).to_dict()}
        print(f"  {key:30s} {out['items'][key]['agree_pct']:5.1f}% | {per_group(df, agree * 100)}")

    # RQ4: features and adoption
    feats = counts(df[col(df, FEATURES)])
    out["features"] = show_counts("Requested features:", df[col(df, FEATURES)])
    out["features_top3"] = {}
    for g, sub in [(g, df[df.group == g]) for g in GROUPS] + [("Investors", df[df.investor])]:
        if len(sub):
            top = counts(sub[col(df, FEATURES)]).head(3)
            out["features_top3"][g] = top.to_dict()
            print(f"  top 3, {g}: " + "; ".join(f"{k} ({v})" for k, v in top.items()))
    out["adoption"] = show_counts("Would use a tool like FinSight:", df[col(df, ADOPT)])

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.barh(list(out["items"]), [v["agree_pct"] for v in out["items"].values()], color="#2B6CB0")
        ax.set_xlim(0, 100); ax.set_xlabel("% agree (4–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "opinion_items.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(6, 3))
        feats.sort_values().plot.barh(ax=ax, color="#C05621"); ax.set_xlabel("participants"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "features.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["A: answer only", "B: calculation + source"], [out["trust_A_mean"], out["trust_B_mean"]],
               color=["#A0AEC0", "#2F855A"])
        ax.set_ylim(1, 5); ax.set_ylabel("mean trust (1–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "trust_A_vs_B.png"), dpi=200)
        print(f"\nCharts and summary.json written to {args.out}/")
    except ImportError:
        print(f"\nmatplotlib not installed: summary.json written to {args.out}/, no charts")


if __name__ == "__main__":
    main()
