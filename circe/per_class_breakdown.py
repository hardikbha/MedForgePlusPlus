"""
Per-class (Real / Implant / Remove) accuracy and F1 breakdown, matching the structure of
MedForge's own published Table 1 (Real / Forgery:Implant / Forgery:Remove columns).

Accuracy per class = recall: fraction of images with that ground-truth type predicted correctly.
F1 per class = binary F1 (that type vs Real), computed on the subset excluding the OTHER forgery
type, i.e. Implant-F1 uses {implant, real} only and Remove-F1 uses {remove, real} only. Real-F1
uses all three types (real vs everything else). This is the standard one-vs-rest decomposition for
a binary detector evaluated across ground-truth subgroups, and is the only well-defined reading of
"per-class F1" for a real/deepfake binary verdict -- MedForge's own exact per-class F1 methodology
is not published, so this may not match their internal computation bit-for-bit, but it is the
standard construction and clearly documented here.

Every number is computed directly from a saved eval_medforge.py generation file -- nothing
invented, nothing re-run.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DF = {"remove", "inject", "edit", "deepfake", "removal", "injection", "implant"}


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        pred = parse_output(r["output"])["verdict"] or "real"
        rows.append((r["type"], pred))
    return rows


def class_recall(rows, types):
    want = "real" if types == {"real"} else "deepfake"
    n = c = 0
    for t, pred in rows:
        if t not in types:
            continue
        n += 1
        if pred == want:
            c += 1
    return (c / n * 100 if n else 0.0), n


def class_f1(rows, fake_types, exclude_types):
    tp = fp = fn = tn = 0
    for t, pred in rows:
        if t in exclude_types:
            continue
        gt = "deepfake" if t in fake_types else "real"
        if gt == "deepfake" and pred == "deepfake":
            tp += 1
        elif gt == "deepfake" and pred == "real":
            fn += 1
        elif gt == "real" and pred == "deepfake":
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) * 100 if (prec + rec) else 0.0


def real_f1_all(rows):
    tp = fp = fn = tn = 0
    for t, pred in rows:
        gt = "real" if t == "real" else "deepfake"
        if gt == "real" and pred == "real":
            tp += 1
        elif gt == "real" and pred == "deepfake":
            fn += 1
        elif gt == "deepfake" and pred == "real":
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) * 100 if (prec + rec) else 0.0


def report(name, path, fake_type_a="edit", fake_type_b="remove"):
    rows = load(path)
    racc, rn = class_recall(rows, {"real"})
    iacc, inn = class_recall(rows, {fake_type_a})
    xacc, xn = class_recall(rows, {fake_type_b})
    rf1 = real_f1_all(rows)
    if1 = class_f1(rows, {fake_type_a}, {fake_type_b})
    xf1 = class_f1(rows, {fake_type_b}, {fake_type_a})
    avg_acc = (racc * rn + iacc * inn + xacc * xn) / (rn + inn + xn)
    avg_f1 = (rf1 + if1 + xf1) / 3
    print(f"--- {name} ---")
    print(f"  Real:    acc={racc:6.2f}  f1={rf1:6.2f}  n={rn}")
    print(f"  Implant: acc={iacc:6.2f}  f1={if1:6.2f}  n={inn}")
    print(f"  Remove:  acc={xacc:6.2f}  f1={xf1:6.2f}  n={xn}")
    print(f"  AVG:     acc={avg_acc:6.2f}  f1={avg_f1:6.2f}")
    return dict(real_acc=racc, implant_acc=iacc, remove_acc=xacc, avg_acc=avg_acc,
                real_f1=rf1, implant_f1=if1, remove_f1=xf1, avg_f1=avg_f1)


if __name__ == "__main__":
    B = _release_path('AAAI/MedForge++')
    report("MedForge In-Domain (SFT, exact protocol)", f"{B}/gen_mfp_indomain.jsonl")
    report("MedForge Cross-Forgery (interp a=0.10)", f"{B}/gen_interp_alpha0.10.jsonl")
    report("MedForge Cross-Model (SFT, exact protocol)", f"{B}/gen_mfp_crossmodel.jsonl")
