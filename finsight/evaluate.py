"""Score predictions with the official TAT-QA/TAT-DQA metric, plus per-type and evidence numbers."""
from collections import defaultdict

from .tatqa_metric import TaTQAEmAndF1


def score(questions, predictions):
    """questions: compact-format gold questions.
    predictions: {uid: {"answer": [...], "scale": str, "evidence": [ids]}}; missing uids count as wrong.
    Returns a dict of overall and per-answer-type EM/F1 (percent) and evidence recall/precision."""
    overall = TaTQAEmAndF1()
    per_type = defaultdict(TaTQAEmAndF1)
    ev_hit = ev_gold = ev_pred = 0
    rows = []
    for q in questions:
        p = predictions.get(q["uid"], {})
        answer, scale = p.get("answer") or None, p.get("scale", "")
        gold = {"answer": q["answer"], "answer_type": q["answer_type"], "scale": q["scale"]}
        overall(ground_truth=gold, prediction=answer, pred_scale=scale)
        per_type[q["answer_type"]](ground_truth=gold, prediction=answer, pred_scale=scale)
        detail = overall.get_raw()[-1]
        rows.append({"uid": q["uid"], "type": q["answer_type"], "em": detail["em"], "f1": detail["f1"]})
        pe, ge = set(p.get("evidence", [])), set(q["evidence"])
        ev_hit += len(pe & ge)
        ev_gold += len(ge)
        ev_pred += len(pe)

    em, f1, scale_acc, _ = overall.get_overall_metric()
    result = {"n": len(questions), "EM": 100 * em, "F1": 100 * f1, "scale_acc": 100 * scale_acc,
              "evidence_recall": 100 * ev_hit / max(1, ev_gold),
              "evidence_precision": 100 * ev_hit / max(1, ev_pred)}
    for t, m in sorted(per_type.items()):
        t_em, t_f1, _, _ = m.get_overall_metric()
        result[f"{t}_EM"] = 100 * t_em
        result[f"{t}_F1"] = 100 * t_f1
        result[f"{t}_n"] = m._count
    result["rows"] = rows
    return result


def print_report(result, title=""):
    print(f"==== {title} (n={result['n']}) ====")
    print(f"EM {result['EM']:.2f} | F1 {result['F1']:.2f} | scale acc {result['scale_acc']:.2f} | "
          f"evidence R/P {result['evidence_recall']:.1f}/{result['evidence_precision']:.1f}")
    for t in ("span", "multi-span", "arithmetic", "count"):
        if f"{t}_n" in result:
            print(f"  {t:<11} n={result[f'{t}_n']:<5} EM {result[f'{t}_EM']:.2f}  F1 {result[f'{t}_F1']:.2f}")
