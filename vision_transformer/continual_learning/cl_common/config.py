"""Every setting for the continual-learning (CL) experiments lives in this one file.
There are no command-line options: to change something, edit it here."""
import os

import torch

# ------------------------------------------------------------------ data
# The ORIGINAL dataset folder (the one that contains train/ validation/ test/).
# No separate dataset_CL copy is needed: the experiences are built on the fly from it.
DATA_DIR = "C:/7th_sem_project/ai_art_and_image_detector_cl/AI_art_and_image_detector_with_continual_Learning/dataset"

# One global label mapping for the whole experiment (output 0 is always "real", ...).
CLASS_TO_IDX = {"real": 0, "dalle3": 1, "midjourney": 2, "sd21": 3, "sd3": 4, "sdxl": 5,  "nanobanana": 6, "gpt_image_2": 7}
NUM_CLASSES = 8
GENERATORS = ["dalle3", "midjourney", "sd21", "sd3", "sdxl", "nanobanana", "gpt_image_2"]   # experience 1..5 (each one also gets "real")

# Quick smoke test: set this to e.g. 200 to use only 200 images per class (0 = use everything).
# Smoke-test results go to a different folder ("results_smoke"), so they never mix with real ones.
MAX_PER_CLASS = 0

# ------------------------------------------------------------------ images (same as your offline model)
PREPROCESS = "crop"      # random 224 crops of the original pixels (keeps the generator fingerprints)
IMAGE_SIZE = 224
JPEG_PROB = 0.3
BLUR_PROB = 0.1

# ------------------------------------------------------------------ training (identical for every method -> fair comparison)
BATCH_SIZE = 32          # lower to 16 if the GPU runs out of memory
NUM_WORKERS = min(4, os.cpu_count() or 2)   # lower to 2 if Windows says "paging file too small" (error 1455)
HEAD_EPOCHS = 1          # experience 1 only: first train just the new head (backbone frozen)
EPOCHS_PER_EXPERIENCE = 5
LR_HEAD = 3e-4
LR = 5e-5                # top layer; lower layers get LR * LAYER_DECAY ** depth
LAYER_DECAY = 0.8
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
SEED = 42

# ------------------------------------------------------------------ the CL methods
REPLAY_IMAGES_PER_GENERATOR = 5000   # images kept from old experiences (try 500 / 1000 / 2000 / 5000 / 10000)
EWC_LAMBDA = 10.0           # how strongly EWC protects old weights (try 1 / 10 / 100; see README)
EWC_FISHER_SAMPLES = 1000   # images used to measure which weights are important
KD_LAMBDA = 2.0             # weight of the distillation loss
KD_TEMPERATURE = 2.0

# ------------------------------------------------------------------ evaluation / output
EVAL_SPLIT = "test"         # use "validation" while you are tuning the numbers above, "test" for final results
EVAL_TTA = True             # test with 5 crops per image (4 corners + centre) and average them; same for every method
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results_smoke" if MAX_PER_CLASS else "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [n for n, _ in sorted(CLASS_TO_IDX.items(), key=lambda kv: kv[1])]
