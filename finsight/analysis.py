"""Error breakdown: for every wrong answer, test which single repair would make it correct under
the official scorer. The counts show which post-processing fix is worth adding."""
from collections import Counter

from .tatqa_metric import TaTQAEmAndF1
from .tatqa_utils import to_number

TYPE_NAMES = {"span", "multi-span", "arithmetic", "count"}


def _correct(q, answer, scale):
    m = TaTQAEmAndF1()
    m(ground_truth={"answer": q["answer"], "answer_type": q["answer_type"], "scale": q["scale"]},
      prediction=answer or None, pred_scale=scale)
    return m.get_raw()[-1]["em"] == 1


def _num(answer):
    return to_number(str(answer[0])) if len(answer) == 1 else None


def categorize(q, p):
    """Category of one wrong prediction p ({"answer", "scale", "type"})."""
    answer, scale = p.get("answer") or [], p.get("scale", "")
    if not answer:
        return "no answer"
    if _correct(q, answer, q["scale"]):
        return "unit (scale) only"
    v = _num(answer)
    if v is not None:
        for label, w in (("sign only", -v), ("x100 / ÷100", v * 100), ("x100 / ÷100", v / 100)):
            for sc in (scale, q["scale"]):
                if _correct(q, [str(round(w, 4))], sc):
                    return label
        g = to_number(str(q["answer"] if not isinstance(q["answer"], list) else q["answer"][0]))
        if g not in (None, 0) and abs(v - g) / abs(g) < 0.01:
            return "rounding (<1%)"
    if p.get("type") in TYPE_NAMES and p["type"] != q["answer_type"]:
        return f"wrong type ({p['type']} for {q['answer_type']})"
    return f"wrong value ({q['answer_type']})"


def error_breakdown(questions, preds, rows):
    """rows: result['rows'] from evaluate.score. Returns Counter of categories over wrong answers."""
    em = {r["uid"]: r["em"] for r in rows}
    return Counter(categorize(q, preds.get(q["uid"], {})) for q in questions if em.get(q["uid"]) == 0)


def print_breakdown(counter, total):
    wrong = sum(counter.values())
    print(f"{wrong} wrong of {total}. Potential gain if each category were fixed:")
    for cat, n in counter.most_common():
        print(f"  {n:4d}  (+{100 * n / total:4.1f} EM)  {cat}")
