"""Data for continual learning.
Experience k = all 'real' images + all images of generator k, taken straight from the original
dataset folder (nothing is copied). Labels always use the one global mapping in config.py."""
import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cl_common import config as cfg
from dataset import IMG_EXT, build_transform, find_split_dir   # your existing code (transforms etc.)


def list_images(split, class_name):
    """All image paths of one class in one split (optionally capped by MAX_PER_CLASS)."""
    folder = os.path.join(find_split_dir(cfg.DATA_DIR, split), class_name)
    paths = sorted(e.path for e in os.scandir(folder)
                   if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXT)
    if cfg.MAX_PER_CLASS > 0:
        paths = random.Random(cfg.SEED).sample(paths, min(cfg.MAX_PER_CLASS, len(paths)))
    return paths


def get_items(split, class_names):
    """List of (image_path, global_label) for the given classes."""
    return [(p, cfg.CLASS_TO_IDX[c]) for c in class_names for p in list_images(split, c)]


class ImageDataset(Dataset):
    """Returns (image_tensor, label). A corrupt image is replaced by a random one instead of crashing."""

    def __init__(self, items, train):
        self.items = items
        tta = (not train) and cfg.EVAL_TTA and cfg.PREPROCESS == "crop"      # 5 crops per image when testing
        self.transform = build_transform(cfg.PREPROCESS, cfg.IMAGE_SIZE, train, tta=tta,
                                         jpeg_prob=cfg.JPEG_PROB if train else 0.0,
                                         blur_prob=cfg.BLUR_PROB if train else 0.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        for _ in range(5):
            path, label = self.items[i]
            try:
                with Image.open(path) as im:
                    return self.transform(im.convert("RGB")), label
            except Exception:
                print(f"[warn] unreadable image skipped: {path}")
                i = random.randrange(len(self.items))
        raise RuntimeError("Too many unreadable images")


def experience_dataset(k):
    """Training data of experience k (0-based): real + GENERATORS[k]."""
    return ImageDataset(get_items("train", ["real", cfg.GENERATORS[k]]), train=True)


def seen_test_dataset(k):
    """Evaluation data after experience k: real + every generator learned so far."""
    return ImageDataset(get_items(cfg.EVAL_SPLIT, ["real"] + cfg.GENERATORS[:k + 1]), train=False)


def make_loader(dataset, train):
    workers = cfg.NUM_WORKERS if train else max(0, cfg.NUM_WORKERS // 2)
    batch = cfg.BATCH_SIZE if train or not cfg.EVAL_TTA else max(1, cfg.BATCH_SIZE // 5)   # 5 crops per image
    kwargs = dict(batch_size=batch, shuffle=train, drop_last=train,
                  num_workers=workers, pin_memory=cfg.DEVICE == "cuda")
    if workers > 0:
        kwargs.update(persistent_workers=train, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)
