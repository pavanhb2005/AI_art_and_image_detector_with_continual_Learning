"""The ONE training loop used by every method. A CL method only changes:
   - which batches it trains on        (Strategy.make_loader)   <- Replay
   - what loss it minimises            (Strategy.loss)          <- EWC, KD
   - what it remembers after an experience (Strategy.after_experience)
Everything else (optimizer, learning rate, epochs, ...) is identical, so the comparison is fair."""
import math
import warnings

import torch
from tqdm import tqdm

from cl_common import config as cfg
from cl_common.data import make_loader
from models import layerwise_param_groups, set_head_only   # your existing code


warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")   # harmless fp16 first-step warning


class Strategy:
    """Base class. Naive training = this + a plain cross-entropy loss."""

    def make_loader(self, dataset):
        return make_loader(dataset, train=True)

    def loss(self, model, x, y):
        raise NotImplementedError

    def after_experience(self, model, dataset):
        pass

    def status(self):
        return ""        # optional text shown in the log (e.g. the size of the EWC penalty)


def train_stage(model, loader, strategy, epochs, head_only, desc):
    device = torch.device(cfg.DEVICE)
    amp = device.type == "cuda"
    set_head_only(model, head_only)
    if head_only:
        opt = torch.optim.AdamW(model.heads.parameters(), lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY)
    else:
        opt = torch.optim.AdamW(layerwise_param_groups(model, cfg.LR, cfg.LAYER_DECAY, cfg.WEIGHT_DECAY))

    total = epochs * len(loader)
    warm = max(1, int(total * cfg.WARMUP_RATIO))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    params = [p for p in model.parameters() if p.requires_grad]

    for ep in range(epochs):
        model.train()
        running, n = 0.0, 0
        bar = tqdm(loader, desc=f"{desc} epoch {ep + 1}/{epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = strategy.loss(model, x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
            bar.set_postfix(loss=f"{running / n:.4f}")
        print(f"  {desc} epoch {ep + 1}/{epochs}: loss {running / n:.4f} {strategy.status()}")
