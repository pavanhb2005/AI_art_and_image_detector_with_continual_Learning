"""Reads results/<method>/metrics.json for every method and makes one comparison table + plot."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_common import config as cfg

METHODS = ["naive", "replay", "replay_kd", "ewc", "kd"]

# (column title, key in summary, show as percent?)
COLUMNS = [("accuracy", "final_accuracy", True), ("avg class acc", "final_avg_acc", True),
           ("macro-F1", "final_macro_f1", True), ("MCC", "final_mcc", False), ("AUC", "final_roc_auc", False),
           ("real-vs-AI acc", "final_real_vs_ai_accuracy", True), ("real-vs-AI AUC", "final_real_vs_ai_auc", False),
           ("avg incr. acc", "avg_incremental_acc", True), ("forgetting", "final_forgetting", True),
           ("bwd transfer", "final_backward_transfer", True)]


def _fmt(v, pct):
    if v is None:
        return "-"
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def compare_results():
    results = {}
    for m in METHODS:
        path = os.path.join(cfg.RESULTS_DIR, m, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
    if not results:
        print("No results found yet.")
        return

    lines = [f"{'method':10s}" + "".join(f"{title:>16s}" for title, _, _ in COLUMNS)]
    for m, r in results.items():
        lines.append(f"{m:10s}" + "".join(f"{_fmt(r['summary'].get(key), pct):>16s}" for _, key, pct in COLUMNS))
    lines.append("\n(all values in %, except MCC and the two AUCs; forgetting and backward transfer are percentage points; "
                 "everything is measured after the last experience, on all classes seen so far)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(cfg.RESULTS_DIR, "comparison.txt"), "w") as f:
        f.write(text + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m, r in results.items():
        x = list(range(1, len(r["avg_acc"]) + 1))
        axes[0].plot(x, [a * 100 for a in r["avg_acc"]], marker="o", label=m)
        axes[1].plot(x[1:], [f * 100 for f in r["forgetting"][1:]], marker="o", label=m)
    axes[0].set_title("Average 6-way accuracy over all classes seen so far")
    axes[0].set_ylabel("accuracy (%)")
    axes[1].set_title("Forgetting of old generators (lower is better)")
    axes[1].set_ylabel("percentage points")
    for ax in axes:
        ax.set_xlabel("after experience")
        ax.set_xticks(list(range(1, max(len(r["avg_acc"]) for r in results.values()) + 1)))
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()
    print(f"Saved comparison.txt and comparison.png to {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    compare_results()
