import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

from cl_common.runner import run_experiment
from replay_kd.strategy import ReplayKD

if __name__ == "__main__":
    run_experiment(ReplayKD(), "replay_kd")
