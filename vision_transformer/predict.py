"""Classify one or more images (or a whole folder) and show how confident the model is.

   python predict.py photo.jpg                     newest saved model (continual-learning or offline)
   python predict.py photo.jpg --choice best       the saved model with the best recorded macro-F1
   python predict.py folder_with_images            every image in the folder
   python predict.py photo.jpg --model continual_learning\\results\\replay\\final_model.pth
   python predict.py --list                        show which saved models were found
   python predict.py photo.jpg --show              also display the image"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from config import DEVICE
from dataset import IMG_EXT, build_transform
from utils import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- finding a saved model
def _recorded_score(model_path):
    """macro-F1 stored next to a model (continual-learning metrics.json, or the offline test/val metrics), else None."""
    folder = os.path.dirname(model_path)
    for fname, read in (("metrics.json", lambda j: j["summary"]["final_macro_f1"]),
                        ("test_metrics.json", lambda j: j["classification"]["macro_f1"]),
                        ("val_metrics.json", lambda j: j["classification"]["macro_f1"])):
        f = os.path.join(folder, fname)
        if os.path.isfile(f):
            try:
                with open(f) as fh:
                    return float(read(json.load(fh)))
            except Exception:
                continue
    return None


def find_models():
    """All saved models: continual_learning/results/<method>/final_model.pth and runs/<run>/best_model.pth."""
    paths = []
    cl_root = os.path.join(HERE, "continual_learning")
    for results in ("results", "results_smoke"):          # smoke-test models are only used if there is nothing else
        folder = os.path.join(cl_root, results)
        found = [os.path.join(folder, d, "final_model.pth") for d in sorted(os.listdir(folder))] if os.path.isdir(folder) else []
        found = [p for p in found if os.path.isfile(p)]
        if found:
            paths += found
            break
    runs = os.path.join(HERE, "runs")
    if os.path.isdir(runs):
        paths += [p for p in (os.path.join(runs, d, "best_model.pth") for d in sorted(os.listdir(runs))) if os.path.isfile(p)]
    return [{"path": p, "mtime": os.path.getmtime(p), "score": _recorded_score(p)} for p in paths]


def pick_model(choice):
    models = find_models()
    if not models:
        return None, "no saved model found"
    if choice == "best":
        scored = [m for m in models if m["score"] is not None]
        if scored:
            m = max(scored, key=lambda m: m["score"])
            return m["path"], f"best recorded macro-F1 = {m['score'] * 100:.2f}%"
        print("(no model has recorded metrics -> using the newest one instead)")
    m = max(models, key=lambda m: m["mtime"])
    return m["path"], "newest saved model"


# ----------------------------------------------------------------------------- confidence numbers
def analyse(crop_probs, names, real_idx, task):
    """crop_probs: [crops, classes] softmax outputs. Returns the final decision + confidence measures."""
    probs = crop_probs.mean(0)
    p_ai = float(probs[1]) if task == "binary" else float(1.0 - probs[real_idx])
    crop_ai = crop_probs[:, 1] if task == "binary" else 1.0 - crop_probs[:, real_idx]
    order = np.argsort(-probs)
    top, second = int(order[0]), int(order[1])
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / math.log(len(probs)))
    return {
        "probs": probs, "p_ai": p_ai, "top": top, "second": second, "order": order,
        "margin": float(probs[top] - probs[second]), "entropy": entropy,
        "crops": len(crop_probs),
        "source_agree": int((crop_probs.argmax(1) == top).sum()),
        "ai_agree": int(((crop_ai >= 0.5) == (p_ai >= 0.5)).sum()),
    }


def confidence_word(p):
    c = max(p, 1 - p)
    return "very high" if c >= 0.95 else "high" if c >= 0.85 else "moderate" if c >= 0.70 else "low"


def print_result(path, r, names, task):
    verdict = "AI-GENERATED" if r["p_ai"] >= 0.5 else "REAL"
    print(f"{os.path.basename(path)}")
    print(f"  Verdict:        {verdict}  (P(AI) = {r['p_ai'] * 100:.2f}%, confidence: {confidence_word(r['p_ai'])})")
    if task == "multiclass":
        t, s = r["top"], r["second"]
        print(f"  Most likely:    {names[t]} ({r['probs'][t] * 100:.2f}%)   runner-up: {names[s]} ({r['probs'][s] * 100:.2f}%)   "
              f"margin: {r['margin'] * 100:.1f} points")
        print("  Probabilities:  " + " | ".join(f"{names[i]} {r['probs'][i] * 100:.1f}%" for i in r["order"]))
    print(f"  Certainty:      entropy {r['entropy']:.2f} (0 = sure, 1 = clueless)", end="")
    if r["crops"] > 1:
        print(f" | {r['ai_agree']}/{r['crops']} crops agree on real-vs-AI"
              + (f", {r['source_agree']}/{r['crops']} on the source" if task == "multiclass" else ""))
    else:
        print()
    notes = []
    if task == "multiclass" and r["probs"][r["top"]] < 0.6:
        notes.append("the exact source is uncertain - it may come from a generator this model has not learned")
    if r["crops"] > 1 and r["ai_agree"] < r["crops"]:
        notes.append("different parts of the image give different answers")
    if r["entropy"] >= 0.85:
        notes.append("the model is unsure overall (probability is spread over many classes), "
                     "so even the real-vs-AI score is unreliable")
    if max(r["p_ai"], 1 - r["p_ai"]) < 0.7:
        notes.append("real-vs-AI decision is close to a coin flip; treat it with caution")
    for n in notes:
        print(f"  Note:           {n}")
    print()


# ----------------------------------------------------------------------------- main
def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in IMG_EXT)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="image file(s) or folder(s)")
    ap.add_argument("--model", default=None, help="a specific saved model (.pth); overrides --choice")
    ap.add_argument("--choice", choices=["latest", "best"], default="latest",
                    help="which saved model to use automatically (default: latest)")
    ap.add_argument("--list", action="store_true", help="list the saved models that were found and exit")
    ap.add_argument("--show", action="store_true", help="display each image with its prediction")
    ap.add_argument("--no_tta", action="store_true", help="use one centre crop instead of 5 crops")
    a = ap.parse_args()

    if a.list:
        for m in sorted(find_models(), key=lambda m: -m["mtime"]):
            score = "no recorded metrics" if m["score"] is None else f"macro-F1 {m['score'] * 100:.2f}%"
            print(f"{os.path.relpath(m['path'], HERE)}  |  {score}")
        return

    if a.model:
        model_path, why = a.model, "chosen by --model"
    else:
        model_path, why = pick_model(a.choice)
    if not model_path:
        raise SystemExit("No saved model found. Train first (train.py or a continual_learning method) "
                         "or pass --model path\\to\\model.pth")
    device = torch.device(DEVICE)
    model, ckpt = load_checkpoint(model_path, device)
    cfg, names, task = ckpt["config"], ckpt["class_names"], ckpt["task"]
    real_idx = next(i for i, n in enumerate(ckpt["source_names"]) if n.lower() == "real")
    tta = (not a.no_tta) and cfg["preprocess"] == "crop"
    tf = build_transform(cfg["preprocess"], cfg["image_size"], train=False, tta=tta)
    print(f"Model: {model_path}  ({why})\nClasses: {names} | task: {task} | {'5-crop' if tta else 'single-crop'} prediction\n")

    paths = collect(a.paths) or [input("Enter image path: ").strip().strip('"')]
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"{path}: could not open the image ({e})\n")
            continue
        x = tf(img)
        x = x if x.ndim == 4 else x.unsqueeze(0)
        with torch.no_grad():
            crop_probs = torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy()
        r = analyse(crop_probs, names, real_idx, task)
        print_result(path, r, names, task)
        if a.show:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.title(f"{'AI-GENERATED' if r['p_ai'] >= 0.5 else 'REAL'} - P(AI) {r['p_ai'] * 100:.1f}%")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
