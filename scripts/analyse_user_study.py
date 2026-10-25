"""Analyse the FinSight user-study responses (Google Forms → Responses → Download CSV).

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

ROLE, INVEST = "describes you best", "invest in stocks"
TASK_A, TASK_B, USED = "Task A: Is FinSight", "Task B: Is FinSight", "what did you use to decide"
TIME_HAND = "how long would these two questions take"
PREFER, TRUST_A, TRUST_B = "Which answer do you trust more", "trust Answer A", "trust Answer B"
UMUX = ["capabilities meet my requirements", "FinSight is easy to use"]  # UMUX-Lite (Lewis et al., 2013)
ITEMS = {
    "evidence_helped_check": "highlighted evidence made it easy",
    "equation_helped": "equation helped me understand",
    "prefer_not_sure": "rather the system say",
}
FEATURES, ADOPT = "most useful to you", "If FinSight were a real product"
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


def per_group(df, values, fmt="{:.1f}"):
    return "  ".join(f"{g}={fmt.format(values[df.group == g].mean())}" for g in GROUPS if (df.group == g).any())


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
    out["groups"] = df.group.value_counts().to_dict()
    out["investors"] = int(df.investor.sum())
    print("Groups: " + ", ".join(f"{k} {v}" for k, v in out["groups"].items()) + f"; investors {out['investors']}")
    if (df.group == "Technical").mean() > 0.5:
        print("  note: more than half are engineering/IT; results mostly reflect non-target users")

    # tasks
    a_ok = df[col(df, TASK_A)].astype(str).str.startswith("Correct")
    caught = df[col(df, TASK_B)].astype(str).str.startswith("Wrong")
    uses_ev = df[col(df, USED)].astype(str).str.contains("highlighted|equation", case=False)
    out["taskA_judged_correct_pct"] = pct(a_ok.mean())
    out["taskB_caught_pct"] = pct(caught.mean())
    out["taskB_caught_using_evidence_pct"] = pct(caught[uses_ev].mean()) if uses_ev.any() else None
    out["taskB_caught_not_using_evidence_pct"] = pct(caught[~uses_ev].mean()) if (~uses_ev).any() else None
    print(f"\nTask A judged correct: {out['taskA_judged_correct_pct']}%")
    print(f"Task B (FinSight wrong): {out['taskB_caught_pct']}% caught the error — "
          f"{out['taskB_caught_using_evidence_pct']}% of those who used the evidence/equation vs "
          f"{out['taskB_caught_not_using_evidence_pct']}% of those who did not")
    print(f"  caught by group: {per_group(df, caught * 100, '{:.0f}%')}")
    out["time_by_hand"] = counts(df[col(df, TIME_HAND)]).to_dict()
    print(f"Time by hand for the two questions: {out['time_by_hand']}")

    # trust: bare vs explained
    df["trust_A"], df["trust_B"] = num(df[col(df, TRUST_A)]), num(df[col(df, TRUST_B)])
    paired = df[["trust_A", "trust_B"]].dropna()
    differs = (paired.trust_B != paired.trust_A).any()
    out["trust_wilcoxon_p"] = float(wilcoxon(paired.trust_B, paired.trust_A).pvalue) if differs else None
    out["trust_A_mean"], out["trust_B_mean"] = round(df.trust_A.mean(), 2), round(df.trust_B.mean(), 2)
    out["prefer_B_pct"] = pct(df[col(df, PREFER)].astype(str).str.startswith("B").mean())
    print(f"\nTrust, answer only (A) vs explained (B): mean {out['trust_A_mean']} vs {out['trust_B_mean']} "
          f"(Wilcoxon p = {out['trust_wilcoxon_p']}); {out['prefer_B_pct']}% prefer B")

    # usability: UMUX-Lite → SUS-equivalent
    u1, u2 = num(df[col(df, UMUX[0])]), num(df[col(df, UMUX[1])])
    df["umux"] = ((u1 - 1) + (u2 - 1)) / 8 * 100
    df["sus_eq"] = 0.65 * df.umux + 22.9
    out["umux_lite_mean"], out["sus_equivalent_mean"] = round(df.umux.mean(), 1), round(df.sus_eq.mean(), 1)
    grade = "excellent" if out["sus_equivalent_mean"] >= 80.3 else "good" if out["sus_equivalent_mean"] >= 68 else "below average"
    print(f"\nUsability: UMUX-Lite {out['umux_lite_mean']} → SUS-equivalent {out['sus_equivalent_mean']} "
          f"(68 = average; {grade})   by group: {per_group(df, df.sus_eq)}")

    # explanation items
    print("\nOpinion items, % agree (4–5): overall | by group")
    out["items"] = {}
    for key, needle in ITEMS.items():
        agree = (num(df[col(df, needle)]) >= 4).astype(float)
        out["items"][key] = {"agree_pct": pct(agree.mean()),
                             "by_group": (df.assign(a=agree).groupby("group").a.mean() * 100).round(1).to_dict()}
        print(f"  {key:22s} {out['items'][key]['agree_pct']:5.1f}% | {per_group(df, agree * 100, '{:.0f}%')}")

    # features and adoption
    feats = counts(df[col(df, FEATURES)])
    out["features"] = feats.to_dict()
    print("\nRequested features:\n" + feats.to_string())
    out["features_top3"] = {}
    for g, sub in [(g, df[df.group == g]) for g in GROUPS] + [("Investors", df[df.investor])]:
        if len(sub):
            top = counts(sub[col(df, FEATURES)]).head(3)
            out["features_top3"][g] = top.to_dict()
            print(f"  top 3, {g}: " + "; ".join(f"{k} ({v})" for k, v in top.items()))
    out["adoption"] = counts(df[col(df, ADOPT)]).to_dict()
    print(f"\nWould use FinSight: {out['adoption']}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.barh(list(out["items"]), [v["agree_pct"] for v in out["items"].values()], color="#2B6CB0")
        ax.set_xlim(0, 100); ax.set_xlabel("% agree (4–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "opinion_items.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(6, 3))
        feats.sort_values().plot.barh(ax=ax, color="#C05621"); ax.set_xlabel("participants"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "features.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["A: answer only", "B: explained"], [out["trust_A_mean"], out["trust_B_mean"]], color=["#A0AEC0", "#2F855A"])
        ax.set_ylim(1, 5); ax.set_ylabel("mean trust (1–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "trust_A_vs_B.png"), dpi=200)
        print(f"\nCharts and summary.json written to {args.out}/")
    except ImportError:
        print(f"\nmatplotlib not installed: summary.json written to {args.out}/, no charts")


if __name__ == "__main__":
    main()
