"""Two entry points used by every method:

  run_experiment(strategy, name)   learn ALL experiences from scratch (E1 ... E5), save everything
  add_generator(strategy, name)    take the saved model, add the generators that are new in config.py
                                   (one extra experience each) and update the saved model + tables

Everything is saved to results/<name>/."""
import json
import os

import torch
import torch.nn as nn

from cl_common import config as cfg
from cl_common.data import experience_dataset
from cl_common.evaluation import (average_incremental_accuracy, backward_transfer, evaluate_seen, forgetting,
                                  format_final_report, print_experience_line, print_table)
from cl_common.trainer import train_stage
from models import create_model          # your existing ViT-B/16 + 512 -> 256 -> 6 head
from utils import set_seed
from visualize import save_binary_roc, save_confusion_matrix, save_roc_curve


class CLError(Exception):
    """A problem the user can fix (shown as a plain message, no traceback)."""


# ----------------------------------------------------------------------------- one experience
def learn_experience(model, strategy, k, history):
    """Train on experience k (real + GENERATORS[k]), let the strategy remember, then test on all seen classes."""
    assert len(history) == k, "history must contain exactly the experiences before this one"
    gen = cfg.GENERATORS[k]
    print(f"\n=== Experience {k + 1}/{len(cfg.GENERATORS)}: real + {gen} ===")
    data = experience_dataset(k)
    if k == 0:     # very first experience: warm up the brand-new head before touching the backbone
        train_stage(model, strategy.make_loader(data), strategy, cfg.HEAD_EPOCHS, True, "head warm-up")
    train_stage(model, strategy.make_loader(data), strategy, cfg.EPOCHS_PER_EXPERIENCE, False, f"E{k + 1}")

    strategy.after_experience(model, data)      # replay: fill memory | EWC: measure importance | KD: copy teacher
    history.append(evaluate_seen(model, k))
    print_experience_line(history[-1], k)


# ----------------------------------------------------------------------------- saving
def save_results(name, model, history):
    out_dir = os.path.join(cfg.RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    last = len(history) - 1
    acc_matrix = [h["acc"] for h in history]

    report = [print_table("\n6-way accuracy per class (did it name the exact source?)", history, "acc"),
              print_table("\nReal-vs-AI accuracy per class (real called real / generator called AI)", history, "ai_det")]
    c, b = history[last]["detailed"]["classification"], history[last]["detailed"]["real_vs_ai"]
    summary = {
        "final_avg_acc": history[last]["avg_acc"],                     # mean per-class accuracy (= balanced accuracy)
        "final_accuracy": c["accuracy"], "final_macro_f1": c["macro_f1"], "final_weighted_f1": c["f1"],
        "final_mcc": c["mcc"], "final_kappa": c["kappa"], "final_roc_auc": c["roc_auc"],
        "final_log_loss": c["log_loss"], "final_ece": c["ece"],
        "final_avg_real_vs_ai": history[last]["avg_ai_det"],
        "final_real_vs_ai_accuracy": b["accuracy"], "final_real_vs_ai_f1": b["f1_ai"],
        "final_real_vs_ai_auc": b.get("roc_auc"), "final_real_vs_ai_eer": b.get("eer"),
        "avg_incremental_acc": average_incremental_accuracy([h["avg_acc"] for h in history]),
        "final_forgetting": forgetting(acc_matrix, last),
        "final_backward_transfer": backward_transfer(acc_matrix, last),
    }
    print("\nSUMMARY:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})

    metrics = {"method": name, "generators": cfg.GENERATORS[:len(history)], "summary": summary,
               "acc": acc_matrix, "ai_det": [h["ai_det"] for h in history],
               "avg_acc": [h["avg_acc"] for h in history], "avg_ai_det": [h["avg_ai_det"] for h in history],
               "forgetting": [forgetting(acc_matrix, t) for t in range(len(history))],
               "detailed": [h["detailed"] for h in history]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(out_dir, "tables.txt"), "w") as f:
        f.write("\n".join(report) + f"\n\nSUMMARY: {summary}\n")
    with open(os.path.join(out_dir, "final_report.txt"), "w") as f:
        f.write(format_final_report(history[last], summary) + "\n")

    h = history[last]                                  # plots of the final model (all classes seen so far)
    save_confusion_matrix(h["cm"], h["names"], os.path.join(out_dir, "confusion_matrix_final.png"))
    save_roc_curve(h["labels"], h["probs"], h["names"], os.path.join(out_dir, "roc_curve_final.png"))
    save_binary_roc(h["is_ai"], h["ai_score"], os.path.join(out_dir, "roc_real_vs_ai_final.png"))

    ckpt_path = os.path.join(out_dir, "final_model.pth")
    if os.path.exists(ckpt_path):                      # keep exactly one step of undo
        os.replace(ckpt_path, os.path.join(out_dir, "previous_model.pth"))
    torch.save({"model": model.state_dict(), "class_names": cfg.CLASS_NAMES, "source_names": cfg.CLASS_NAMES,
                "task": "multiclass", "config": {"preprocess": cfg.PREPROCESS, "image_size": cfg.IMAGE_SIZE},
                "generators": cfg.GENERATORS[:len(history)], "class_to_idx": dict(cfg.CLASS_TO_IDX)}, ckpt_path)
    print(f"\nSaved results to {out_dir}")
    return metrics


# ----------------------------------------------------------------------------- 1) from scratch
def run_experiment(strategy, name):
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"\n########## {name.upper()} | device: {device} ##########")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found -> ViT-B/16 on CPU is extremely slow.")

    model = create_model(cfg.NUM_CLASSES, pretrained=True).to(device)
    history = []       # one entry per experience: all metrics on all classes seen so far
    for k in range(len(cfg.GENERATORS)):
        learn_experience(model, strategy, k, history)
    return save_results(name, model, history)


# ----------------------------------------------------------------------------- 2) add new generator(s)
def expand_head(model, num_classes):
    """Give the model more outputs: the old rows of the last layer are copied, the new rows start fresh."""
    old = model.heads.head[-1]
    if old.out_features == num_classes:
        return
    if old.out_features > num_classes:
        raise CLError(f"The saved model has {old.out_features} outputs but config.py only has {num_classes} classes.")
    new = nn.Linear(old.in_features, num_classes).to(old.weight.device)
    with torch.no_grad():
        new.weight[:old.out_features] = old.weight
        new.bias[:old.out_features] = old.bias
    model.heads.head[-1] = new


def add_generator(strategy, name):
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    out_dir = os.path.join(cfg.RESULTS_DIR, name)
    ckpt_path, metrics_path = os.path.join(out_dir, "final_model.pth"), os.path.join(out_dir, "metrics.json")
    if not (os.path.exists(ckpt_path) and os.path.exists(metrics_path)):
        raise CLError(f"No trained '{name}' model found in {out_dir}. Run {name}\\run.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    learned = ckpt.get("generators") or [n for n in ckpt["class_names"] if n != "real"]   # older checkpoints
    if cfg.GENERATORS[:len(learned)] != learned:
        raise CLError(f"The saved model learned {learned}, but GENERATORS in config.py starts with "
                      f"{cfg.GENERATORS[:len(learned)]}. Keep the old generators in the same order and only append new ones.")
    saved_map = ckpt.get("class_to_idx") or {}
    if any(cfg.CLASS_TO_IDX.get(n) != i for n, i in saved_map.items()):
        raise CLError("CLASS_TO_IDX in config.py changed for an existing class. Only append new classes with new numbers.")
    pending = cfg.GENERATORS[len(learned):]
    if not pending:
        raise CLError(f"Nothing to add: '{name}' already learned every generator in config.py {learned}. "
                      "Add the new generator to CLASS_TO_IDX, NUM_CLASSES and GENERATORS first.")

    with open(metrics_path) as f:
        old = json.load(f)
    if len(old["acc"]) != len(learned):
        raise CLError(f"metrics.json has {len(old['acc'])} experiences but the model learned {len(learned)}.")
    history = [{"acc": old["acc"][i], "ai_det": old["ai_det"][i], "avg_acc": old["avg_acc"][i],
                "avg_ai_det": old["avg_ai_det"][i], "detailed": (old.get("detailed") or [None] * len(learned))[i]}
               for i in range(len(learned))]

    print(f"\n########## {name.upper()} | adding {pending} to a model that knows {learned} | device: {device} ##########")
    model = create_model(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    expand_head(model, cfg.NUM_CLASSES)
    model.to(device)

    # The method's memory/state is not stored in the checkpoint: rebuild it from the old experiences' data
    # (replay: the same memory images, because the random seed is the same | EWC: importance | KD: teacher).
    print("Rebuilding the method's memory of the old experiences ...")
    for k in range(len(learned)):
        strategy.after_experience(model, experience_dataset(k))

    for k in range(len(learned), len(cfg.GENERATORS)):
        learn_experience(model, strategy, k, history)
    return save_results(name, model, history)


def add_generator_cli(strategy, name):
    """Same as add_generator but prints a plain message instead of a traceback for fixable problems."""
    try:
        add_generator(strategy, name)
    except CLError as e:
        raise SystemExit(f"\n{e}")
