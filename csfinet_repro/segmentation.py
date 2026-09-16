"""Development selection, full-training refit, and patient-volume evaluation."""

import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from .files import read_csv, sha256, write_csv, write_json
from .metrics import region_metrics
from .models import build_segmentation
from .nifti import load_patient
from .training import LOSSES, UPDATE_POLICIES, epoch_learning_rate, seed_everything, train_patient


@torch.inference_mode()
def predict_patient(model, images, slice_batch, device):
    model.eval()
    predictions = []
    for start in range(0, len(images), slice_batch):
        batch = torch.as_tensor(images[start:start + slice_batch], device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            predicted = model(batch).argmax(dim=1)
        predictions.append(predicted.cpu().numpy().astype(np.uint8))
    return np.concatenate(predictions)


def original_labels(indices):
    result = np.asarray(indices, dtype=np.uint8).copy()
    if not np.isin(result, [0, 1, 2, 3]).all():
        raise ValueError("Expected training indices 0..3")
    result[result == 3] = 4
    return result


def checkpoint_payload(model, optimizer, scaler, rng, epoch, best_epoch, best_score, history, identity,
                       stage="epoch_complete", pending_losses=None, pending_stats=None):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   shuffle_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   epoch=epoch, best_epoch=best_epoch, best_score=best_score, history=history, identity=identity,
                   stage=stage, pending_losses=pending_losses)
    if pending_stats is not None:
        # v2 only: per-epoch optimizer step counters completed before validation; payloads without counters keep their keys.
        payload["pending_stats"] = pending_stats
    return payload


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256(path)


def train_segmentation(config_path, patient_csv, root, output, name, phase, epochs=None, selection=None, resume=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = config["segmentation"]
    # Training options are supplied by the experiment configuration.
    loss_name = settings.get("loss", "four_class_cross_entropy")
    update_policy = settings.get("update_policy", "per_patient")
    augmentation = settings.get("augmentation") or None
    lr_schedule = settings.get("lr_schedule") or None
    selection_tolerance = float(settings.get("selection_tolerance", 0.0))
    if loss_name not in LOSSES or update_policy not in UPDATE_POLICIES:
        raise ValueError("Unknown segmentation loss or update policy")
    if augmentation is not None and not isinstance(augmentation, dict):
        raise ValueError("Segmentation augmentation must be false or a mapping of augmentation settings")
    if selection_tolerance < 0:
        raise ValueError("Selection tolerance must be non-negative")
    epoch_learning_rate(settings["learning_rate"], 1, settings["max_epochs"], lr_schedule)
    protocol_v2 = None
    if (loss_name != "four_class_cross_entropy" or update_policy != "per_patient" or augmentation
            or lr_schedule or selection_tolerance):
        protocol_v2 = dict(loss=loss_name, update_policy=update_policy, augmentation=augmentation,
                           lr_schedule=lr_schedule, schedule_horizon_epochs=settings["max_epochs"],
                           base_learning_rate=settings["learning_rate"], selection_tolerance=selection_tolerance,
                           amp_overflow_policy=("skip_update_and_halve_scale" if update_policy == "per_slice_batch"
                                                else "raise_without_update"),
                           train_loss_recorded_as="mean_train_CE")
    loss_label = "CE" if loss_name == "four_class_cross_entropy" else loss_name
    rows = read_csv(patient_csv, ["patient_id", "split", "development_split"])
    if len(rows) != 235 or len({r["patient_id"] for r in rows}) != 235:
        raise ValueError("Expected frozen cohort of 235 unique patients")
    if phase not in {"pilot", "development", "refit"}:
        raise ValueError("Unknown training phase")
    training = [r for r in rows if r["split"] == "train" and (phase == "refit" or r["development_split"] == "train")]
    validation = [r for r in rows if r["development_split"] == "validation"] if phase != "refit" else []
    if len(training) != (188 if phase == "refit" else 150) or len(validation) != (0 if phase == "refit" else 38):
        raise ValueError("Split counts do not match protocol")
    selection_sha = None
    if phase == "pilot":
        training, validation = training[:2], validation[:1]
        budget = 2 if epochs is None else epochs
    elif phase == "development":
        budget = settings["max_epochs"] if epochs is None else epochs
    else:
        if selection is None or epochs is not None:
            raise ValueError("Refit requires a completed development run.json; no arbitrary epoch override")
        selected = json.loads(Path(selection).read_text(encoding="utf-8"))
        if (selected.get("phase") != "development" or selected.get("status") != "completed"
                or selected.get("model") != name or selected.get("config_sha256") != sha256(config_path)
                or selected.get("split_sha256") != sha256(patient_csv)):
            raise ValueError("Selection provenance does not match this completed development protocol")
        budget, selection_sha = selected["best_epoch"], sha256(selection)
    if not isinstance(budget, int) or budget < 1 or budget > settings["max_epochs"]:
        raise ValueError("Epoch budget outside configured range")
    data_hashes = {}
    for row in training + validation:
        for modality in ("flair", "t1", "t1ce", "t2", "seg"):
            path = Path(root) / row["patient_id"] / f"{row['patient_id']}_{modality}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(f"Download incomplete: {path}")
            provenance = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
            digest = sha256(path)
            if provenance["gzip_sha256"] != digest:
                raise ValueError(f"MRI provenance hash mismatch: {path}")
            data_hashes[path.name] = digest
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires CUDA")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Run directory exists; choose a new run or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(model=name, phase=phase, config_sha256=sha256(config_path), split_sha256=sha256(patient_csv),
                    selection_sha256=selection_sha, max_epochs=budget,
                    code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    identity["data_sha256"] = data_hashes
    seed_everything(config["seed"])
    device = torch.device("cuda")
    model = build_segmentation(name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    rng = np.random.default_rng(config["seed"])
    first_epoch, best_epoch, best_score, history = 1, 0, -1.0, []
    resume_stage, pending_losses, pending_stats = "epoch_complete", None, None
    execution = dict(amp="float16", grad_scaler_initial_scale=1024, tf32=False, data_loader_workers=0,
                     cudnn_benchmark=False, cudnn_deterministic=True, activation_checkpoint=settings.get("activation_checkpoint", False))
    if protocol_v2:
        execution["protocol_v2"] = protocol_v2
    manifest = dict(**identity, status="running", seed=config["seed"],
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    training_ids=[r["patient_id"] for r in training], validation_ids=[r["patient_id"] for r in validation],
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    execution=execution,
                    determinism="seeded; CUDA grid_sample backward nondeterministic", research_result=phase != "pilot")
    if resume:
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        pending_path, last_path = output / "pending.pt", output / "last.pt"
        if (pending_path.exists() and previous.get("pending_checkpoint_sha256") == sha256(pending_path)):
            checkpoint_path = pending_path
        elif last_path.exists() and previous.get("last_checkpoint_sha256") == sha256(last_path):
            checkpoint_path = last_path
        else:
            raise ValueError("No provenance-matched pending or completed-epoch checkpoint")
        # Only local checkpoints produced by this application with recorded SHA are accepted.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume requires unchanged config, split, code, and epoch budget")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        rng.bit_generator.state = checkpoint["shuffle_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        resume_stage = checkpoint.get("stage", "epoch_complete")
        if resume_stage not in {"epoch_complete", "validation_pending"}:
            raise ValueError("Unknown segmentation checkpoint stage")
        first_epoch = checkpoint["epoch"] if resume_stage == "validation_pending" else checkpoint["epoch"] + 1
        best_epoch, best_score = checkpoint["best_epoch"], checkpoint["best_score"]
        history = checkpoint["history"]
        pending_losses, pending_stats = checkpoint.get("pending_losses"), checkpoint.get("pending_stats")
        if resume_stage == "validation_pending" and (not pending_losses or not np.isfinite(pending_losses).all()):
            raise ValueError("Validation-pending checkpoint requires finite training losses")
        manifest = previous
        del checkpoint
    write_json(output / "config.json", config)
    write_json(output / "run.json", manifest)
    try:
        for epoch in range(first_epoch, budget + 1):
            started = time.perf_counter()
            if lr_schedule:
                # Recomputed from the epoch number on every (resumed) epoch; an omitted schedule keeps the optimizer's constant rate.
                learning_rate = epoch_learning_rate(settings["learning_rate"], epoch, settings["max_epochs"], lr_schedule)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            epoch_stats = {}
            if resume_stage == "validation_pending" and epoch == first_epoch:
                losses = list(pending_losses)
                epoch_stats = dict(pending_stats or {})
                print(f"Resuming {name}/{phase} epoch={epoch} at validation from verified pending.pt", flush=True)
            else:
                losses = []
                for position, index in enumerate(rng.permutation(len(training)), 1):
                    patient = training[index]["patient_id"]
                    images, targets = load_patient(root, patient)
                    # Augmentation draws follow the permutation on the same checkpointed rng, so --resume replays them.
                    loss = train_patient(model, optimizer, scaler, images, targets, settings["slice_batch"], device,
                                         loss_name=loss_name, update_policy=update_policy, augmentation=augmentation,
                                         rng=rng, stats=epoch_stats if protocol_v2 else None)
                    losses.append(loss)
                    del images, targets
                    print(f"{name}/{phase} epoch={epoch} patient={position}/{len(training)} id={patient} {loss_label}={loss:.6f}", flush=True)
                pending = checkpoint_payload(model, optimizer, scaler, rng, epoch, best_epoch, best_score,
                                             history, identity, stage="validation_pending", pending_losses=losses,
                                             pending_stats=epoch_stats if protocol_v2 else None)
                manifest["pending_checkpoint_sha256"] = atomic_checkpoint(output / "pending.pt", pending)
                manifest.update(pending_epoch=epoch, pending_stage="validation")
                write_json(output / "run.json", manifest)
                del pending
            progress_path = output / "validation-progress.json"
            score_by_patient = {}
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    if (progress.get("epoch") == epoch and progress.get("pending_checkpoint_sha256") ==
                            manifest.get("pending_checkpoint_sha256")):
                        score_by_patient = {item["patient_id"]: float(item["WT_Dice"])
                                            for item in progress.get("scores", [])}
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    score_by_patient = {}
            if not set(score_by_patient).issubset({row["patient_id"] for row in validation}) \
                    or not np.isfinite(list(score_by_patient.values())).all():
                raise ValueError("Invalid validation progress")
            for position, row in enumerate(validation, 1):
                if row["patient_id"] in score_by_patient:
                    print(f"{name}/{phase} validation epoch={epoch} patient={position}/{len(validation)} "
                          f"id={row['patient_id']} (verified existing)", flush=True)
                    continue
                images, targets = load_patient(root, row["patient_id"])
                predicted = predict_patient(model, images, settings["slice_batch"], device)
                score_by_patient[row["patient_id"]] = region_metrics(
                    original_labels(targets), original_labels(predicted))["WT"]["dice"]
                del images, targets, predicted
                write_json(progress_path, dict(epoch=epoch,
                           pending_checkpoint_sha256=manifest.get("pending_checkpoint_sha256"),
                           scores=[dict(patient_id=patient, WT_Dice=value)
                                   for patient, value in sorted(score_by_patient.items())]))
                print(f"{name}/{phase} validation epoch={epoch} patient={position}/{len(validation)} "
                      f"id={row['patient_id']} WT_Dice={score_by_patient[row['patient_id']]:.6f}", flush=True)
            scores = [score_by_patient[row["patient_id"]] for row in validation]
            score = float(np.mean(scores)) if scores else None
            # selection_tolerance defaults to 0.0, which leaves the strict ">" comparison unchanged.
            improved = score is not None and score > best_score + selection_tolerance
            if improved:
                best_score, best_epoch = score, epoch
                manifest["best_checkpoint_sha256"] = atomic_checkpoint(output / "best.pt", model.state_dict())
            entry = dict(epoch=epoch, mean_train_CE=float(np.mean(losses)), validation_WT_Dice=score,
                         elapsed_seconds=time.perf_counter() - started)
            if protocol_v2:
                entry.update(learning_rate=optimizer.param_groups[0]["lr"], **epoch_stats)
            history.append(entry)
            payload = checkpoint_payload(model, optimizer, scaler, rng, epoch, best_epoch, best_score, history,
                                         identity, stage="epoch_complete")
            manifest["last_checkpoint_sha256"] = atomic_checkpoint(output / "last.pt", payload)
            manifest.update(last_epoch=epoch, best_epoch=best_epoch, best_validation_WT_Dice=best_score if validation else None,
                            history=history)
            write_json(output / "run.json", manifest)
            for path in (output / "pending.pt", progress_path):
                if path.exists():
                    path.unlink()
            manifest.pop("pending_checkpoint_sha256", None)
            manifest.pop("pending_epoch", None)
            manifest.pop("pending_stage", None)
            write_json(output / "run.json", manifest)
            print(f"Epoch {epoch}: train {loss_label}={np.mean(losses):.6f}, validation WT Dice={score}", flush=True)
            resume_stage, pending_losses, pending_stats = "epoch_complete", None, None
            if validation and epoch - best_epoch >= settings["patience"]:
                break
        if phase == "refit":
            manifest["final_checkpoint_sha256"] = atomic_checkpoint(output / "final.pt", model.state_dict())
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        write_json(output / "run.json", manifest)
    return manifest
