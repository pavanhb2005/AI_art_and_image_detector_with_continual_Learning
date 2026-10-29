"""Update the saved ewc model with the new generator(s) you added to cl_common/config.py
(CLASS_TO_IDX, NUM_CLASSES, GENERATORS) - no retraining from scratch."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

from cl_common.runner import add_generator_cli
from ewc.strategy import EWC

if __name__ == "__main__":
    add_generator_cli(EWC(), "ewc")
