"""Evaluation after every experience + the continual-learning metrics.

Two groups of numbers:
  * normal classification metrics at one point in time (accuracy, precision, recall, F1, MCC, AUC, ...)
    -> reuses calculate_metrics / binary_view from your metrics.py, plus a few extras
  * continual-learning metrics over time (average accuracy, forgetting, backward transfer)"""
import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, log_loss, precision_recall_fscore_support,
                             roc_auc_score)

from cl_common import config as cfg
from cl_common.data import make_loader, seen_test_dataset
from metrics import binary_view, calculate_metrics      # your existing code


@torch.no_grad()
def predict_probs(model, loader):
    """Softmax probabilities for every image (averaged over the 5 crops when EVAL_TTA is on)."""
    device = torch.device(cfg.DEVICE)
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        crops = 1
        if x.ndim == 5:                                   # [batch, crops, 3, H, W]
            crops = x.shape[1]
            x = x.flatten(0, 1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        p = torch.softmax(out.float(), dim=1)
        if crops > 1:
            p = p.view(-1, crops, p.shape[-1]).mean(1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def expected_calibration_error(probs, labels, bins=15):
    """How far the model's confidence is from its real accuracy (0 = perfectly calibrated)."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def evaluate_seen(model, k):
    """Test the model on every class seen up to experience k (0-based) and compute all metrics.
    Predictions are made among the classes seen so far (a class that was never trained cannot be predicted)."""
    names = ["real"] + cfg.GENERATORS[:k + 1]
    cols = [cfg.CLASS_TO_IDX[n] for n in names]
    new_index = {old: new for new, old in enumerate(cols)}

    probs, labels = predict_probs(model, make_loader(seen_test_dataset(k), train=False))
    probs = probs[:, cols]
    probs = probs / probs.sum(1, keepdims=True)
    labels = np.array([new_index[int(v)] for v in labels])
    preds = probs.argmax(1)
    n = len(names)

    m, cm, report = calculate_metrics(labels, preds, probs, names)           # accuracy, P/R/F1, kappa, MCC, AUC
    b, is_ai, ai_score = binary_view(labels, probs, names, "multiclass")     # real-vs-AI view
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(n)), zero_division=0)

    m["precision_macro"] = float(np.mean(prec))
    m["recall_macro"] = float(np.mean(rec))
    m["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    m["top2_accuracy"] = float((np.argsort(-probs, 1)[:, :2] == labels[:, None]).any(1).mean()) if n > 2 else 1.0
    m["log_loss"] = float(log_loss(labels, probs, labels=list(range(n))))
    m["ece"] = expected_calibration_error(probs, labels)

    per_class = {}
    for i, name in enumerate(names):
        try:
            auc = float(roc_auc_score(labels == i, probs[:, i]))
        except ValueError:
            auc = float("nan")
        per_class[name] = {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                           "auc": auc, "support": int(support[i])}

    per_source = b.pop("per_source_accuracy")
    return {
        "acc": {name: float(rec[i]) for i, name in enumerate(names)},    # 6-way accuracy per class (= recall)
        "ai_det": per_source,                                            # real-vs-AI accuracy per class
        "avg_acc": float(np.mean(rec)),                                  # every seen class counts equally
        "avg_ai_det": float(np.mean(list(per_source.values()))),
        "macro_f1": float(m["macro_f1"]),
        "detailed": {"classification": {k_: float(v) for k_, v in m.items()},
                     "real_vs_ai": {k_: float(v) for k_, v in b.items()},
                     "per_class": per_class},
        "report": report, "cm": cm, "names": names,                      # kept in memory only (plots / text report)
        "labels": labels, "preds": preds, "probs": probs, "is_ai": is_ai, "ai_score": ai_score,
    }


# ----------------------------------------------------------------------------- CL metrics
# acc_matrix[t][name] = accuracy on class `name` after experience t.  (generator j is learned in experience j)
def forgetting(acc_matrix, t):
    """Average over the generators learned before experience t of
       (best accuracy it ever had) - (accuracy now)."""
    if t == 0:
        return None
    drops = []
    for j, gen in enumerate(cfg.GENERATORS[:t]):
        best = max(acc_matrix[s][gen] for s in range(j, t))
        drops.append(best - acc_matrix[t][gen])
    return float(np.mean(drops))


def backward_transfer(acc_matrix, t):
    """Average over old generators of (accuracy now) - (accuracy right after learning it).
    Negative = learning new generators hurt the old ones."""
    if t == 0:
        return None
    return float(np.mean([acc_matrix[t][g] - acc_matrix[j][g] for j, g in enumerate(cfg.GENERATORS[:t])]))


def average_incremental_accuracy(avg_acc_list):
    """Mean of the 'average accuracy' measured after every experience (one number for the whole journey)."""
    return float(np.mean(avg_acc_list))


# ----------------------------------------------------------------------------- printing
def print_table(title, matrix, key):
    names = ["real"] + cfg.GENERATORS[:len(matrix)]
    head = f"{'':12s}" + "".join(f"{'after E' + str(t + 1):>10s}" for t in range(len(matrix)))
    lines = [title, head]
    for n in names:
        row = "".join(f"{matrix[t][key][n] * 100:9.1f}%" if n in matrix[t][key] else f"{'-':>10s}"
                      for t in range(len(matrix)))
        lines.append(f"{n:12s}{row}")
    avg = "".join(f"{m['avg_' + key] * 100:9.1f}%" for m in matrix)
    lines.append(f"{'AVERAGE':12s}{avg}")
    text = "\n".join(lines)
    print(text)
    return text


def format_final_report(h, summary):
    """Plain-text report of the last evaluation: every metric + per-class table."""
    c, b = h["detailed"]["classification"], h["detailed"]["real_vs_ai"]
    pct = lambda v: f"{v * 100:6.2f}%"
    nan = float("nan")
    lines = [
        f"CLASSIFICATION ({len(h['names'])} classes: {', '.join(h['names'])})",
        f"  accuracy {pct(c['accuracy'])} | balanced accuracy {pct(c['balanced_accuracy'])} | top-2 accuracy {pct(c['top2_accuracy'])}",
        f"  precision: macro {pct(c['precision_macro'])}, weighted {pct(c['precision'])}",
        f"  recall:    macro {pct(c['recall_macro'])}, weighted {pct(c['recall'])}",
        f"  F1:        macro {pct(c['macro_f1'])}, weighted {pct(c['f1'])}",
        f"  MCC {c['mcc']:.4f} | Cohen kappa {c['kappa']:.4f} | ROC-AUC (macro one-vs-rest) {c['roc_auc']:.4f}",
        f"  log loss {c['log_loss']:.4f} | calibration error (ECE) {c['ece']:.4f}",
        "",
        "REAL vs AI-GENERATED",
        f"  accuracy {pct(b['accuracy'])} | precision {pct(b['precision_ai'])} | recall (AI caught) {pct(b['recall_ai'])} | "
        f"real recall {pct(b['real_recall'])} | F1 {pct(b['f1_ai'])}",
        f"  ROC-AUC {b.get('roc_auc', nan):.4f} | average precision {b.get('avg_precision', nan):.4f} | EER {b.get('eer', nan):.4f}",
        "",
        "PER CLASS", h["report"],
        "CONTINUAL LEARNING",
        f"  final average accuracy {pct(summary['final_avg_acc'])} | average incremental accuracy {pct(summary['avg_incremental_acc'])}",
    ]
    if summary["final_forgetting"] is not None:
        lines.append(f"  forgetting {summary['final_forgetting'] * 100:.2f} points | "
                     f"backward transfer {summary['final_backward_transfer'] * 100:.2f} points")
    return "\n".join(lines)


def print_experience_line(h, k):
    c, b = h["detailed"]["classification"], h["detailed"]["real_vs_ai"]
    print(f"  -> after E{k + 1}: average accuracy {h['avg_acc'] * 100:.1f}% | macro-F1 {c['macro_f1'] * 100:.1f}% | "
          f"MCC {c['mcc']:.3f} | AUC {c['roc_auc']:.3f} | real-vs-AI acc {b['accuracy'] * 100:.1f}% (AUC {b.get('roc_auc', float('nan')):.3f})")
    print("     per class: " + ", ".join(f"{n} {a * 100:.1f}%" for n, a in h["acc"].items()))
