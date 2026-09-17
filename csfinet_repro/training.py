"""Patient-level training, configurable loss/augmentation/schedules, and traceable GPU smoke runs."""

import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .files import read_csv, sha256, write_json
from .models import build_segmentation
from .nifti import load_patient


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # CUDA grid_sample backward is nondeterministic; do not claim bitwise replay.
    torch.use_deterministic_algorithms(False)


LOSSES = ("four_class_cross_entropy", "dice_ce")
UPDATE_POLICIES = ("per_patient", "per_slice_batch")


def dice_ce_loss(logits, targets, smooth=1.0):
    """Cross entropy plus soft Dice averaged over the foreground classes (indices 1..C-1) of one slice batch.

    Softmax and the Dice terms are computed in float32, so the loss is safe under float16 autocast.
    Each class sums intersection and cardinalities over N, H and W; the per-class term is
    ``1 - (2 * intersection + smooth) / (prediction_sum + target_sum + smooth)``, so a class absent
    from both prediction and target contributes exactly 0 and a perfect one-hot prediction gives 0.
    """
    if logits.ndim != 4 or logits.shape[1] < 2 or targets.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("Dice/CE loss expects NCHW logits with at least one foreground class and NHW targets")
    if smooth <= 0:
        raise ValueError("Dice smoothing must be positive")
    logits = logits.float()
    ce = F.cross_entropy(logits, targets)
    probabilities = torch.softmax(logits, dim=1)[:, 1:]
    one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].to(probabilities.dtype)
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    return ce + dice.mean()


def _uniform_range(name, value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not float(value[0]) <= float(value[1]):
        raise ValueError(f"{name} must be a [low, high] range with low <= high")
    return float(value[0]), float(value[1])


def augment_slices(images, targets, rng, settings):
    """Flip and re-scale one slice batch with draws from ``rng`` in a fixed order.

    ``settings`` keys: ``flip_axes`` (negative spatial axes shared by NCHW images and NHW targets;
    one Bernoulli(0.5) draw per axis in the listed order, labels flipped identically),
    ``intensity_scale`` and ``intensity_shift`` (``[low, high]`` ranges; one uniform draw each, in
    that order, applied to every modality of the batch, images only). The output is a pure function
    of the inputs and the generator state, so a resumed epoch replays the same batches. Falsy
    ``settings`` returns the inputs unchanged without consuming any draw (augmentation disabled).
    """
    if not settings:
        return images, targets
    if rng is None:
        raise ValueError("Augmentation requires the checkpointed numpy Generator")
    if not isinstance(settings, dict):
        raise ValueError("Augmentation settings must be a mapping")
    images, targets = np.asarray(images), np.asarray(targets)
    if images.ndim != 4 or targets.ndim != 3 or images.shape[0] != targets.shape[0] or images.shape[2:] != targets.shape[1:]:
        raise ValueError("Augmentation expects NCHW images and NHW targets with matching N, H, W")
    for axis in settings.get("flip_axes") or ():
        axis = int(axis)
        if axis not in (-2, -1):
            raise ValueError("flip_axes must name the shared trailing spatial axes -2 (H) or -1 (W)")
        if rng.random() < 0.5:
            images, targets = np.flip(images, axis=axis), np.flip(targets, axis=axis)
    scale_range, shift_range = settings.get("intensity_scale"), settings.get("intensity_shift")
    scale = rng.uniform(*_uniform_range("intensity_scale", scale_range)) if scale_range is not None else None
    shift = rng.uniform(*_uniform_range("intensity_shift", shift_range)) if shift_range is not None else None
    if scale is not None or shift is not None:
        images = images * np.float32(1.0 if scale is None else scale) + np.float32(0.0 if shift is None else shift)
    return np.ascontiguousarray(images, dtype=np.float32), np.ascontiguousarray(targets)


def epoch_learning_rate(base_lr, epoch, max_epochs, schedule):
    """Learning rate of a 1-based epoch; ``schedule=None`` returns ``base_lr`` unchanged.

    ``{"kind": "cosine", "warmup_epochs": W, "final_fraction": f}``: epochs ``1..W`` use
    ``base_lr * epoch / W``; the first epoch after warmup uses ``base_lr`` and the rate then follows
    a half cosine reaching ``base_lr * f`` exactly at ``max_epochs``. The horizon is the configured
    ``max_epochs`` for every phase, so a refit of ``best_epoch`` epochs repeats the development rates
    of epochs ``1..best_epoch``. It is a pure function: no scheduler object, so checkpoints keep the
    constant-rate checkpoint layout and a resumed run recomputes the rate from the epoch number.
    """
    if schedule is None:
        return base_lr
    if not isinstance(schedule, dict) or schedule.get("kind") != "cosine":
        raise ValueError("Only the cosine learning-rate schedule is defined")
    warmup, final_fraction = int(schedule.get("warmup_epochs", 0)), float(schedule.get("final_fraction", 0.0))
    if not 1 <= int(epoch) <= int(max_epochs) or int(epoch) != epoch or int(max_epochs) != max_epochs:
        raise ValueError("Epoch must be an integer in 1..max_epochs")
    if warmup < 0 or warmup >= max_epochs or not 0.0 <= final_fraction <= 1.0:
        raise ValueError("Warmup must leave at least one cosine epoch and final_fraction must be within [0, 1]")
    if epoch <= warmup:
        return base_lr * epoch / warmup
    span = max_epochs - warmup - 1
    progress = (epoch - warmup - 1) / span if span > 0 else 0.0
    final = base_lr * final_fraction
    return final + (base_lr - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _loss_function(loss_name):
    if loss_name == "four_class_cross_entropy":
        return F.cross_entropy
    if loss_name == "dice_ce":
        return dice_ce_loss
    raise ValueError(f"Unknown segmentation loss: {loss_name}")


def _gradients_finite(model):
    flags = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    return bool(torch.stack(flags).all().item()) if flags else True


def train_patient(model, optimizer, scaler, images, targets, slice_batch, device, amp=True, *,
                  loss_name="four_class_cross_entropy", update_policy="per_patient", augmentation=None,
                  rng=None, stats=None):
    """Train on one patient's slices; the keyword-only options default to cross-entropy with per-patient updates.

    ``per_patient``: exactly one optimizer update for all slices; each slice batch's loss is
    weighted by its slice count (the final short batch included) so the update equals one
    full-patient gradient. A non-finite loss or gradient raises and no update is performed.
    ``per_slice_batch`` (study): one update per slice batch on the unweighted batch loss; the return
    value is the slice-count-weighted mean loss. A non-finite loss still raises; a batch whose
    unscaled gradients are non-finite skips its update and lets ``GradScaler`` halve the scale
    (standard dynamic loss scaling), because roughly eight times more steps per epoch would
    otherwise grow the float16 scale until a single overflow aborted the run.
    ``augmentation`` draws come from ``rng`` per slice batch in a fixed order (see
    ``augment_slices``). ``stats``, when a dict, accumulates ``optimizer_steps`` and
    ``skipped_updates`` in place.
    """
    compute_loss = _loss_function(loss_name)
    if update_policy not in UPDATE_POLICIES:
        raise ValueError(f"Unknown update policy: {update_policy}")
    if augmentation and rng is None:
        raise ValueError("Augmentation requires the checkpointed numpy Generator")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_slices = len(images)
    if total_slices == 0 or total_slices != len(targets) or slice_batch < 1:
        raise ValueError("Invalid patient slices or batch size")
    mean_loss, steps, skipped = 0.0, 0, 0
    for start in range(0, total_slices, slice_batch):
        batch_images, batch_targets = images[start:start + slice_batch], targets[start:start + slice_batch]
        if augmentation:
            batch_images, batch_targets = augment_slices(batch_images, batch_targets, rng, augmentation)
        x = torch.as_tensor(batch_images, device=device, dtype=torch.float32)
        y = torch.as_tensor(batch_targets, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = compute_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite cross entropy" if loss_name == "four_class_cross_entropy"
                                     else f"Non-finite {loss_name} loss")
        fraction = len(x) / total_slices
        mean_loss += float(loss.detach()) * fraction
        if update_policy == "per_patient":
            scaler.scale(loss * fraction).backward()
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if _gradients_finite(model):
            scaler.step(optimizer)
            steps += 1
        else:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
        scaler.update()
    if update_policy == "per_patient":
        scaler.unscale_(optimizer)
        finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            raise FloatingPointError("Non-finite gradients; optimizer update not performed")
        scaler.step(optimizer)
        scaler.update()
        steps = 1
    if stats is not None:
        stats["optimizer_steps"] = stats.get("optimizer_steps", 0) + steps
        stats["skipped_updates"] = stats.get("skipped_updates", 0) + skipped
    return mean_loss


def smoke_segmentation(config_path, patient_csv, root, output, name="csfinet", slice_batch=None, slices=155, steps=2):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    candidates = [r for r in read_csv(patient_csv, ["patient_id", "development_split"])
                  if r["development_split"] == "train"]
    available = [r["patient_id"] for r in candidates
                 if all((Path(root) / r["patient_id"] / f"{r['patient_id']}_{m}.nii.gz").exists()
                        for m in ("flair", "t1", "t1ce", "t2", "seg"))]
    if not available:
        raise FileNotFoundError("No complete development-training patient downloaded yet")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    if not 1 <= slices <= 155 or not 1 <= steps <= 200:
        raise ValueError("Slices must be 1..155 and smoke steps 1..200")
    patient = available[0]
    settings = config["segmentation"]
    batch = settings["slice_batch"] if slice_batch is None else slice_batch
    seed_everything(config["seed"])
    device = torch.device("cuda")
    model = build_segmentation(name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    images, targets = load_patient(root, patient)
    # A shorter smoke uses central slices, never a test patient.
    start = (155 - slices) // 2
    images, targets = images[start:start + slices], targets[start:start + slices]
    before = model.head.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    report = dict(kind="engineering_smoke_not_research_result", model=name, patient_id=patient,
                  patient_role="development_train", slice_count=slices, slice_batch=batch,
                  config_sha256=sha256(config_path), split_sha256=sha256(patient_csv), seed=config["seed"],
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
                  cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  determinism="seeded; CUDA grid_sample backward is not bitwise deterministic")
    try:
        losses = []
        for step in range(steps):
            losses.append(train_patient(model, optimizer, scaler, images, targets, batch, device))
            if steps > 5 and (step + 1) % 10 == 0:
                print(f"Overfit smoke step={step + 1}/{steps} CE={losses[-1]:.6f}", flush=True)
        torch.cuda.synchronize()
        changed = not torch.equal(before, model.head.weight.detach())
        if not changed:
            raise RuntimeError("Optimizer did not change output head")
        report.update(status="passed", mean_cross_entropy_by_step=losses, optimizer_steps=steps, head_weights_changed=changed)
        if steps > 5:
            from .segmentation import original_labels, predict_patient
            from .metrics import region_metrics
            prediction = predict_patient(model, images, batch, device)
            report["training_subset_metrics_not_test"] = region_metrics(original_labels(targets), original_labels(prediction))
    except (torch.cuda.OutOfMemoryError, FloatingPointError, RuntimeError) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output, report)
        print(json.dumps(report, ensure_ascii=True), flush=True)
    return report
