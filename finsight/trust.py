"""Steps 5.6-5.7: verification-weighted voting over sampled answers, confidence, and
selective answering (abstain when confidence is below a threshold chosen for a target error rate)."""
from collections import defaultdict

from .prompt import parse_output, to_prediction
from .tatqa_utils import normalize_answer
from .verify import verify


def _key(answer, scale):
    return (tuple(sorted(normalize_answer(str(a)) for a in answer)), scale)


def vote(raw_samples, question, doc):
    """raw_samples: list of k raw model outputs for one question.
    Each sample votes for its executed (answer, scale) with weight 0.5 + 0.5 * verifier score.
    Returns the prediction dict of the best-verified sample in the winning group, with
    `confidence` = winning weight / total weight and `agreement` = share of samples in the group."""
    groups = defaultdict(list)
    for raw in raw_samples:
        parsed = parse_output(raw)
        answer, scale = to_prediction(parsed)
        if not answer:
            continue
        checks, vscore = verify(parsed, question, doc)
        groups[_key(answer, scale)].append({"answer": answer, "scale": scale, "evidence": parsed["evidence"],
                                            "type": parsed["type"], "raw": raw, "checks": checks,
                                            "vscore": vscore, "weight": 0.5 + 0.5 * vscore})
    if not groups:
        return {"answer": [], "scale": "", "evidence": [], "type": "", "raw": raw_samples[0] if raw_samples else "",
                "confidence": 0.0, "agreement": 0.0, "vscore": 0.0}
    total = sum(s["weight"] for g in groups.values() for s in g)
    best = max(groups.values(), key=lambda g: sum(s["weight"] for s in g))
    pick = dict(max(best, key=lambda s: s["vscore"]))
    pick["confidence"] = sum(s["weight"] for s in best) / total
    pick["agreement"] = len(best) / len(raw_samples)
    return pick


def risk_coverage(confidences, correct):
    """Accuracy when answering only the most confident fraction. Returns [(coverage, accuracy, threshold)]."""
    pairs = sorted(zip(confidences, correct), key=lambda p: -p[0])
    out, right = [], 0
    for i, (c, ok) in enumerate(pairs, 1):
        right += ok
        if i == len(pairs) or pairs[i][0] != c:  # only cut between distinct confidence values
            out.append((i / len(pairs), right / i, c))
    return out


def choose_threshold(confidences, correct, target_error=0.05):
    """Lowest confidence threshold whose answered set has a conservative error estimate
    (errors + 1) / (answered + 1) <= target_error on this calibration set (finite-sample
    correction as in conformal risk control). Returns None if no threshold achieves it."""
    pairs = sorted(zip(confidences, correct), key=lambda p: -p[0])
    best, wrong = None, 0
    for i, (c, ok) in enumerate(pairs, 1):
        wrong += 1 - ok
        if (i == len(pairs) or pairs[i][0] != c) and (wrong + 1) / (i + 1) <= target_error:
            best = c
    return best


def selective_report(confidences, correct, threshold):
    answered = [ok for c, ok in zip(confidences, correct) if c >= threshold]
    n = len(confidences)
    return {"threshold": threshold, "coverage": 100 * len(answered) / n,
            "accuracy_on_answered": 100 * sum(answered) / max(1, len(answered)),
            "error_on_answered": 100 * (1 - sum(answered) / max(1, len(answered))), "abstained": n - len(answered)}
