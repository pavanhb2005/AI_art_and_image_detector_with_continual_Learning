"""REPLAY + KD: Replay (old images mixed into every batch) AND distillation from a frozen teacher.

 - Replay anchors the old classes with real images and their labels
 - the teacher (the model as it was before this experience) additionally keeps the model's *whole answer*
   about the old classes unchanged on every image in the batch - memory images and new images alike
This is usually the strongest option when only a small memory of old images is allowed."""
from kd.strategy import KD
from replay.strategy import Replay


class ReplayKD(Replay):
    def __init__(self):
        super().__init__()
        self.kd = KD()

    def loss(self, model, x, y):
        return self.kd.loss(model, x, y)                 # cross-entropy + distillation on the replay-mixed batch

    def after_experience(self, model, dataset):
        super().after_experience(model, dataset)         # replay: update the memory
        self.kd.after_experience(model, dataset)         # kd: freeze a teacher

    def status(self):
        return self.kd.status()
