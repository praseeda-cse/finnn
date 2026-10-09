"""CPU-only checks of voting, reward and expert-iteration selection, using synthetic samples
built from gold targets (correct), corrupted programs (wrong) and wrong evidence.

    python scripts/test_trust.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finsight.data import load_split
from finsight.prompt import gold_target
from finsight.reinforce import reward, select_targets
from finsight.trust import choose_threshold, risk_coverage, selective_report, vote

docs, qs = load_split(os.path.join(os.path.dirname(__file__), "..", "data", "tatdqa_dev.json.gz"))
arith = [q for q in qs if q["answer_type"] == "arithmetic" and gold_target(q)][:300]
rng = random.Random(0)


def wrong_program(t):
    return t.replace("program: ", "program: 1.5 * (") .replace("\nscale", ")\nscale")


def wrong_evidence(t):
    return "evidence: 999\n" + t.split("\n", 1)[1]


q = arith[0]
gold = gold_target(q)
r_ok, r_bad, r_wrongev = (reward(x, q, docs[q["doc_uid"]]) for x in (gold, wrong_program(gold), wrong_evidence(gold)))
print("reward gold / wrong answer / right-for-wrong-reason:",
      round(r_ok["reward"], 2), round(r_bad["reward"], 2), round(r_wrongev["reward"], 2))
assert r_ok["reward"] > r_wrongev["reward"] > r_bad["reward"] - 1 and r_ok["correct"] and not r_bad["correct"]

v = vote([gold, gold, wrong_program(gold)], q, docs[q["doc_uid"]])
print("vote 2 right vs 1 wrong -> confidence", round(v["confidence"], 2), "agreement", round(v["agreement"], 2))
assert abs(v["agreement"] - 2 / 3) < 1e-9 and v["confidence"] > 0.6

# Synthetic population: question i is answered correctly with probability that grows with "difficulty"
confs, correct, samples = [], [], {}
for q in arith:
    g = gold_target(q)
    p_right = rng.random()
    raws = [g if rng.random() < p_right else wrong_program(g) for _ in range(5)]
    if rng.random() < 0.2:
        raws = [wrong_evidence(r) for r in raws]
    samples[q["uid"]] = raws
    v = vote(raws, q, docs[q["doc_uid"]])
    from finsight.reinforce import answer_correct
    confs.append(v["confidence"])
    correct.append(int(answer_correct(v["answer"], v["scale"], q)))
print(f"voted accuracy on synthetic set: {100 * sum(correct) / len(correct):.1f}%")
curve = risk_coverage(confs, correct)
print("risk-coverage (coverage, accuracy):", [(round(c, 2), round(a, 2)) for c, a, _ in curve[::max(1, len(curve) // 6)]])
t = choose_threshold(confs, correct, 0.05)
print("threshold for <=5% error:", t, selective_report(confs, correct, t) if t is not None else "none")

targets, stats = select_targets(arith, docs, samples)
print("expert-iteration selection:", stats)
assert all(t.startswith("evidence:") and "999" not in t.split("\n")[0] for t in targets.values())
print("all checks passed")
