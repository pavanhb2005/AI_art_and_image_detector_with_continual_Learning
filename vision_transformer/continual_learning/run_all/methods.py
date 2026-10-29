from ewc.strategy import EWC
from kd.strategy import KD
from naive.strategy import Naive
from replay.strategy import Replay
from replay_kd.strategy import ReplayKD

METHODS = [("naive", Naive), ("replay", Replay), ("replay_kd", ReplayKD), ("ewc", EWC), ("kd", KD)]
