"""Runs Naive, Replay, Replay+KD, EWC and KD one after another (same data, same settings), then compares them."""
import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

import torch

from cl_common import config as cfg
from cl_common.runner import run_experiment
from run_all.compare import compare_results
from run_all.methods import METHODS

if __name__ == "__main__":
    for name, method in METHODS:
        if os.path.exists(os.path.join(cfg.RESULTS_DIR, name, "metrics.json")):
            print(f"\n[{name}] already finished -> skipped (delete {os.path.join(cfg.RESULTS_DIR, name)} to run it again)")
            continue
        run_experiment(method(), name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    compare_results()
