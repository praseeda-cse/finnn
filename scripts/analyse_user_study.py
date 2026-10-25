"""Analyse the FinSight user-study responses (Google Forms → Responses → Download CSV).

    python scripts/analyse_user_study.py responses.csv [--out user_study/results]

Columns are found by matching question text (see user_study/USER_STUDY.md), so the order of
columns in the export does not matter. Likert answers may be 1–5 or the text labels.
Reports overall and per persona (L learner, I investor, P professional, T technical/other),
and writes summary.json plus charts (PNG) to --out.
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import kruskal, wilcoxon

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
PERSONA_QUESTIONS = {  # persona section → question-text needles (only that persona answers them)
    "L": ["courses or exam preparation", "hardest when you do this", "check your answers now",
          "help me learn", "check my own working", "depend on it instead of learning"],
    "I": ["How long have you been investing", "before investing", "If you don't always read them",
          "Where do you get company numbers", "most want to ask about a company", "investment decision only if"],
    "P": ["Your role", "Hours per week", "Tools you use for this now", "cost of a wrong number",
          "upload client or company documents", "before you would use it at work", "save me time when reviewing"],
}
ATTENTION = "show you are reading carefully"
ROLE = "describes you best"
TRUST_A, TRUST_B, PREFER = "trust Answer A", "trust Answer B", "Which answer do you trust more"
TASK3_USED = "What did you use to decide"
TIME_FS, TIME_HAND = "How long did the three tasks take", "by hand, without FinSight"
FEATURES, ADOPT, NPS = "most useful to you", "If FinSight were a real product", "recommend FinSight"
PERSONA_NAMES = {"L": "Learner", "I": "Investor", "P": "Professional", "T": "Technical/other"}


def persona(role):
    r = str(role)
    if r.startswith("Student – Commerce") or r.startswith("Student – CA"):
        return "L"
    if r.startswith("Retail investor"):
        return "I"
    if r.startswith("Finance professional"):
        return "P"
    return "T"


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


def counts(series):
    """Value counts; checkbox answers ("a, b") are split into one count per option."""
    return series.dropna().astype(str).str.split(", ").explode().str.strip().value_counts()


def sus_score(df):
    items = [likert(df[col(df, t)]) for t in SUS_ITEMS]
    total = sum((x - 1) if i % 2 == 0 else (5 - x) for i, x in enumerate(items))
    return total * 2.5


def pct(x):
    return round(100 * float(x), 1)


def by_persona(df, values, fmt="{:.1f}"):
    return "  ".join(f"{p}={fmt.format(values[df.persona == p].mean())}"
                     for p in PERSONA_NAMES if (df.persona == p).any())


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
    df = df.copy()
    n = len(df)
    print(f"Responses: {n_raw} received, {n} kept (dropped {n_raw - n}: duplicates or failed attention check)")
    if n < 30:
        print("  note: fewer than the recommended 30 participants")
    out = {"n_received": n_raw, "n_kept": n}

    df["persona"] = df[col(df, ROLE)].map(persona)
    out["personas"] = df.persona.value_counts().rename(PERSONA_NAMES).to_dict()
    print("\nParticipants by persona: " + ", ".join(f"{k} {v}" for k, v in out["personas"].items()))
    for p, need in (("L", 12), ("I", 8), ("P", 6)):
        if (df.persona == p).sum() < need:
            print(f"  note: {PERSONA_NAMES[p]} below quota ({(df.persona == p).sum()} < {need})")

    # persona sections
    out["persona_answers"] = {}
    for p, needles in PERSONA_QUESTIONS.items():
        sub = df[df.persona == p]
        if sub.empty:
            continue
        print(f"\n--- {PERSONA_NAMES[p]} section (n={len(sub)}) ---")
        for needle in needles:
            c = col(df, needle, required=False)
            if not c:
                continue
            v = likert(sub[c])
            if v.notna().mean() > 0.8 and v.dropna().between(1, 5).all():   # a 1–5 item
                res = {"agree_pct": pct((v >= 4).mean()), "median": v.median()}
                print(f"{c[:70]}\n    {res['agree_pct']}% agree, median {res['median']}")
            else:
                res = counts(sub[c]).to_dict()
                print(f"{c[:70]}\n    " + "; ".join(f"{k}: {v}" for k, v in res.items()))
            out["persona_answers"][f"{p}: {c}"] = res

    # usability
    df["SUS"] = sus_score(df)
    out["sus_mean"], out["sus_sd"] = round(df.SUS.mean(), 1), round(df.SUS.std(), 1)
    out["sus_by_persona"] = df.groupby("persona").SUS.mean().round(1).to_dict()
    grade = "excellent" if out["sus_mean"] >= 80.3 else "good" if out["sus_mean"] >= 68 else "below average"
    print(f"\nSUS: {out['sus_mean']} ± {out['sus_sd']} (68 = average; this is {grade})   by persona: {by_persona(df, df.SUS)}")

    # bare vs explained answer
    df["trust_A"], df["trust_B"] = likert(df[col(df, TRUST_A)]), likert(df[col(df, TRUST_B)])
    paired = df[["trust_A", "trust_B"]].dropna()
    diff = paired.trust_B != paired.trust_A
    out["trust_wilcoxon_p"] = float(wilcoxon(paired.trust_B, paired.trust_A).pvalue) if diff.any() else None
    out["trust_A_median"], out["trust_B_median"] = df.trust_A.median(), df.trust_B.median()
    out["prefer_B_pct"] = pct(df[col(df, PREFER)].astype(str).str.startswith("B").mean())
    print(f"\nTrust, answer only (A) vs explained (B): median {out['trust_A_median']} vs {out['trust_B_median']}, "
          f"Wilcoxon p = {out['trust_wilcoxon_p']}; {out['prefer_B_pct']}% prefer B")
    print(f"  trust in B by persona: {by_persona(df, df.trust_B)}")

    groups = [g.dropna() for _, g in df.groupby("persona").SUS if g.notna().sum() >= 3]
    if len(groups) >= 2:
        out["sus_kruskal_p"] = float(kruskal(*groups).pvalue)
        tb = [g.dropna() for _, g in df.groupby("persona").trust_B if g.notna().sum() >= 3]
        out["trustB_kruskal_p"] = float(kruskal(*tb).pvalue)
        print(f"  personas differ? Kruskal–Wallis p: SUS {out['sus_kruskal_p']:.3f}, trust in B {out['trustB_kruskal_p']:.3f}")

    # trust items
    print("\nTrust items: % agree (4–5), overall | by persona")
    out["trust_items"] = {}
    for key, needle in TRUST_ITEMS.items():
        c = col(df, needle, required=False)
        if c:
            v = likert(df[c])
            agree = (v >= 4).astype(float)
            out["trust_items"][key] = {"agree_pct": pct(agree.mean()), "median": v.median(),
                                       "agree_pct_by_persona": (df.assign(a=agree).groupby("persona").a.mean() * 100).round(1).to_dict()}
            print(f"  {key:24s} {out['trust_items'][key]['agree_pct']:5.1f}% | {by_persona(df, agree * 100, '{:.0f}%')}")

    # tasks
    task_cols = [c for c in df.columns if "Is the answer correct" in c]
    if len(task_cols) >= 3:  # third one is Task 3, where FinSight is wrong
        for i, c in enumerate(task_cols[:2], 1):
            out[f"task{i}_judged_correct_pct"] = pct(df[c].astype(str).str.startswith("Correct").mean())
        caught = df[task_cols[2]].astype(str).str.startswith("Wrong")
        uses_ev = df[col(df, TASK3_USED)].astype(str).str.contains("highlighted|equation", case=False)
        out["task3_caught_pct"] = pct(caught.mean())
        out["task3_caught_with_evidence_pct"] = pct(caught[uses_ev].mean()) if uses_ev.any() else None
        out["task3_caught_without_evidence_pct"] = pct(caught[~uses_ev].mean()) if (~uses_ev).any() else None
        print(f"\nTasks 1–2 judged correct: {out['task1_judged_correct_pct']}%, {out['task2_judged_correct_pct']}%")
        print(f"Task 3 (system wrong): {out['task3_caught_pct']}% caught the error — "
              f"{out['task3_caught_with_evidence_pct']}% of those who used evidence/equation vs "
              f"{out['task3_caught_without_evidence_pct']}% of those who did not")
        print(f"  caught by persona: {by_persona(df, caught * 100, '{:.0f}%')}")

    tf, th = col(df, TIME_FS, required=False), col(df, TIME_HAND, required=False)
    if tf and th:
        out["time_with_finsight"], out["time_by_hand"] = counts(df[tf]).to_dict(), counts(df[th]).to_dict()
        print(f"\nTime for 3 tasks with FinSight: {out['time_with_finsight']}\n             by hand:         {out['time_by_hand']}")

    feats = counts(df[col(df, FEATURES)])
    out["features"] = feats.to_dict()
    print("\nRequested features:\n" + feats.to_string())
    out["features_by_persona"] = {}
    for p in PERSONA_NAMES:
        sub = df[df.persona == p]
        if len(sub):
            top = counts(sub[col(df, FEATURES)]).head(3)
            out["features_by_persona"][p] = top.to_dict()
            print(f"  top 3 for {PERSONA_NAMES[p]}: " + "; ".join(f"{k} ({v})" for k, v in top.items()))

    ad = col(df, ADOPT, required=False)
    if ad:
        out["adoption"] = counts(df[ad]).to_dict()
        print(f"\nWould use FinSight: {out['adoption']}")

    nps = pd.to_numeric(df[col(df, NPS)], errors="coerce")
    out["nps"] = round(100 * ((nps >= 9).mean() - (nps <= 6).mean()), 1)
    print(f"Net Promoter Score: {out['nps']}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        items = out["trust_items"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(list(items), [v["agree_pct"] for v in items.values()], color="#2B6CB0")
        ax.set_xlim(0, 100); ax.set_xlabel("% agree (4–5)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "trust_items.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(6, 3))
        feats.sort_values().plot.barh(ax=ax, color="#C05621"); ax.set_xlabel("participants"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "features.png"), dpi=200)
        fig, ax = plt.subplots(figsize=(5, 3))
        present = [p for p in PERSONA_NAMES if (df.persona == p).any()]
        ax.boxplot([df.SUS[df.persona == p].dropna() for p in present])
        ax.set_xticks(range(1, len(present) + 1), [PERSONA_NAMES[p] for p in present])
        ax.axhline(68, ls="--", c="grey", lw=1); ax.set_ylabel("SUS (0–100)"); fig.tight_layout()
        fig.savefig(os.path.join(args.out, "sus_by_persona.png"), dpi=200)
        print(f"\nCharts and summary.json written to {args.out}/")
    except ImportError:
        print(f"\nmatplotlib not installed: summary.json written to {args.out}/, no charts")


if __name__ == "__main__":
    main()
