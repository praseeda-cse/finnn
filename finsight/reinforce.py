"""Stage 4 (compute-light version): the evidence-aware reward (Step 4.2) and expert iteration
(Step 4.5). The model samples k answers per training question; only answers that are correct
AND grounded in the right evidence are kept and used as new fine-tuning targets."""
import json

from .prompt import parse_output, to_prediction
from .tatqa_metric import TaTQAEmAndF1
from .verify import verify


def answer_correct(answer, scale, q):
    m = TaTQAEmAndF1()
    m(ground_truth={"answer": q["answer"], "answer_type": q["answer_type"], "scale": q["scale"]},
      prediction=answer or None, pred_scale=scale)
    return m.get_raw()[-1]["em"] == 1


def evidence_f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred or not gold:
        return 0.0
    hit = len(pred & gold)
    if hit == 0:
        return 0.0
    p, r = hit / len(pred), hit / len(gold)
    return 2 * p * r / (p + r)


def reward(raw, q, doc, tau=0.5):
    """R = 1.0*AnswerCorrect + 0.3*ScaleCorrect + 0.5*EvidenceF1 + 0.1*FormatValid
           - 0.5*[AnswerCorrect and EvidenceF1 < tau]       (right for the wrong reason)"""
    parsed = parse_output(raw)
    answer, scale = to_prediction(parsed)
    correct = answer_correct(answer, scale, q)
    ev = evidence_f1(parsed["evidence"], q["evidence"])
    checks, _ = verify(parsed, q, doc)
    r = 1.0 * correct + 0.3 * (scale == q["scale"]) + 0.5 * ev + 0.1 * bool(checks.get("format"))
    r -= 0.5 * (correct and ev < tau)
    return {"reward": r, "correct": correct, "evidence_f1": ev, "parsed": parsed, "checks": checks}


def canonical(parsed):
    """Rewrite a parsed sample as a clean 4-line target."""
    lines = ["evidence: " + ", ".join(map(str, parsed["evidence"])), f"type: {parsed['type']}"]
    if parsed["program"] is not None:
        lines.append(f"program: {parsed['program']}")
    elif parsed["items"] is not None:
        lines.append("items: " + json.dumps(parsed["items"], ensure_ascii=False))
    else:
        lines.append("answer: " + json.dumps(parsed["answer"], ensure_ascii=False))
    lines.append(f"scale: {parsed['scale']}")
    return "\n".join(lines)


def select_targets(questions, docs, samples, tau=0.5):
    """Keep, per question, the highest-reward sample that is correct, grounded (EvidenceF1 >= tau)
    and passes the operand check. Returns ({uid: target text}, stats)."""
    targets, stats = {}, {"questions": 0, "solved_any": 0, "kept": 0, "right_wrong_reason": 0}
    for q in questions:
        raws = samples.get(q["uid"])
        if not raws:
            continue
        stats["questions"] += 1
        scored = [reward(r, q, docs[q["doc_uid"]], tau) for r in raws]
        if any(s["correct"] for s in scored):
            stats["solved_any"] += 1
        stats["right_wrong_reason"] += sum(1 for s in scored if s["correct"] and s["evidence_f1"] < tau)
        good = [s for s in scored if s["correct"] and s["evidence_f1"] >= tau
                and s["checks"].get("operands_grounded") is not False]
        if good:
            targets[q["uid"]] = canonical(max(good, key=lambda s: s["reward"])["parsed"])
            stats["kept"] += 1
    return targets, stats
