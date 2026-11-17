import csv, re, sys
from collections import Counter

def norm(s):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).split())

def em(pred, ref):
    return float(norm(pred) == norm(ref))

def f1(pred, ref):
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    n = sum((Counter(p) & Counter(r)).values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(r)
    return 2 * precision * recall / (precision + recall)

if __name__ == "__main__":
    print("This evaluator expects a predictions CSV with columns: model,prediction,reference_answer.")
    if len(sys.argv) < 2:
        print("Usage: python evaluation/evaluate.py predictions.csv")
        raise SystemExit(0)
    scores = {}
    with open(sys.argv[1], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            scores.setdefault(model, []).append((em(row["prediction"], row["reference_answer"]),
                                                 f1(row["prediction"], row["reference_answer"])))
    for model, values in scores.items():
        print(f"{model}: EM={sum(x[0] for x in values)/len(values):.3f}, "
              f"F1={sum(x[1] for x in values)/len(values):.3f}, n={len(values)}")
