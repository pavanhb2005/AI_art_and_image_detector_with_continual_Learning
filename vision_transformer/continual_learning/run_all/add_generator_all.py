"""Updates EVERY trained method with the new generator(s) you added to cl_common/config.py,
then rebuilds the comparison. (To update just one method use <method>/add_generator.py.)"""
import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

import torch

from cl_common.runner import CLError, add_generator
from run_all.compare import compare_results
from run_all.methods import METHODS

if __name__ == "__main__":
    for name, method in METHODS:
        try:
            add_generator(method(), name)
        except CLError as e:
            print(f"\n[{name}] skipped: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    compare_results()
