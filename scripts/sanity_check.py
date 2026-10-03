"""No-GPU checks of the whole output pipeline.

1. Oracle: gold target -> parse -> execute -> official scorer. This is the ceiling the
   format allows; anything below ~100% is a bug in our plumbing, not the model.
2. Shows one rendered document and context-length statistics.

    python scripts/sanity_check.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finsight.data import load_split, render_doc
from finsight.evaluate import print_report, score
from finsight.prompt import gold_target, parse_output, to_prediction

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

for split in ("train", "dev", "test"):
    docs, questions = load_split(os.path.join(DATA, f"tatdqa_{split}.json.gz"))
    preds, no_target = {}, 0
    for q in questions:
        target = gold_target(q)
        if target is None:
            no_target += 1
            continue
        parsed = parse_output(target)
        answer, scale = to_prediction(parsed)
        preds[q["uid"]] = {"answer": answer, "scale": scale, "evidence": parsed["evidence"]}
    result = score(questions, preds)
    print_report(result, f"ORACLE {split} ({no_target} questions have no expressible target)")
    lengths = [len(render_doc(d)) for d in docs.values()]
    print(f"  context chars: median {statistics.median(lengths):.0f}, p95 {sorted(lengths)[int(.95 * len(lengths))]}, "
          f"max {max(lengths)}  (~4 chars per token)\n")

docs, questions = load_split(os.path.join(DATA, "tatdqa_dev.json.gz"))
q = questions[3]
print(render_doc(docs[q["doc_uid"]])[:1500])
print("\nQ:", q["question"])
print(gold_target(q))
