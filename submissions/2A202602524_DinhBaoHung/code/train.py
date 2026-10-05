"""Reusable DeepWeeds training loop for backbone, recipe, and final experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints, Union
import types

import numpy as np
import pandas as pd


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    inference_method: str = "single"    # single | hflip_prob | hflip_logit | amp
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    momentum: float = 0.9
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split phải là 'val' hoặc 'test'")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed phải >= 0")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model, cfg: Config):
    import torch
    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum)
    raise ValueError(f"optimizer không hỗ trợ: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    import torch

    if cfg.epochs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs và steps_per_epoch phải >= 1")
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = min(total_steps, max(0, int(cfg.warmup_epochs * steps_per_epoch)))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """EMA of trainable parameters; copy BN buffers from the live model."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay phải thuộc (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.buffer_names = {name for name, _ in model.named_buffers()}

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for name, value in model.state_dict().items():
                if name in self.buffer_names or not value.is_floating_point():
                    self.shadow[name].copy_(value.detach())
                else:
                    self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        if self.shadow.keys() != state.keys():
            raise ValueError("EMA state_dict không khớp model")
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def average_parameters(self, model):
        return _AveragedParameters(self, model)


class _AveragedParameters:
    def __init__(self, ema: EMA, model):
        self.ema = ema
        self.model = model
        self.backup = None

    def __enter__(self):
        self.backup = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.load_state_dict(self.backup, strict=True)
        return False


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    import torch
    from losses import mix_batch, mixed_loss

    device = torch.device(device)
    model.train()
    if getattr(model, "_backbone_frozen", False):
        for module in model.modules():
            if module is not model and all(not p.requires_grad for p in module.parameters()):
                module.eval()
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_total, sample_total, epoch_lrs = 0.0, 0, []
    total_steps = len(loader)
    report_every = max(1, total_steps // 20)
    epoch_start = time.perf_counter()
    for step, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed_targets = None
        if cfg.mix:
            images, mixed_targets = mix_batch(images, targets, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = mixed_loss(criterion, logits, mixed_targets) if mixed_targets else criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        count = images.shape[0]
        loss_total += float(loss.detach()) * count
        sample_total += count
        epoch_lrs.append(float(optimizer.param_groups[0]["lr"]))
        if step % report_every == 0 or step == total_steps:
            elapsed = time.perf_counter() - epoch_start
            eta = elapsed * (total_steps - step) / step
            print(
                f"train batch {step}/{total_steps} "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True,
            )
    if sample_total == 0:
        raise ValueError("Training loader không có batch")
    return {
        "train_loss": loss_total / sample_total,
        "lr": float(np.mean(epoch_lrs)),
        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
    }


def evaluate(model, loader, criterion, device, inference_method: str = "single"):
    import torch

    if inference_method not in {"single", "hflip_prob", "hflip_logit", "amp"}:
        raise ValueError(f"inference_method không hỗ trợ: {inference_method}")
    model.eval()
    filenames, labels, logits = [], [], []
    total_loss, total = 0.0, 0
    device = torch.device(device)
    with torch.inference_mode():
        for images, targets, names in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            use_amp = inference_method == "amp" and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(images)
                if inference_method in {"hflip_prob", "hflip_logit"}:
                    flipped_output = model(torch.flip(images, dims=(-1,)))
            if inference_method == "hflip_prob":
                probabilities = (
                    torch.softmax(output.float(), dim=1)
                    + torch.softmax(flipped_output.float(), dim=1)
                ) / 2
                output = probabilities.clamp_min(1e-12).log()
            elif inference_method == "hflip_logit":
                output = (output.float() + flipped_output.float()) / 2
            loss = criterion(output, targets)
            count = len(targets)
            total_loss += float(loss) * count
            total += count
            logits.append(output.float().cpu().numpy())
            labels.append(targets.cpu().numpy())
            filenames.extend(list(names))
    if total == 0:
        raise ValueError("Evaluation loader không có mẫu")
    return filenames, np.concatenate(labels), np.concatenate(logits), total_loss / total


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Không có history để vẽ")
    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    loss_axis.plot(epochs, [row["val_loss"] for row in history], label="val")
    loss_axis.set(title="Cross-entropy", xlabel="Epoch", ylabel="Loss")
    loss_axis.legend()
    metric_axis.plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1")
    metric_axis.plot(epochs, [row["val_top1"] for row in history], label="val top-1")
    metric_axis.set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    metric_axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _metric_record(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    from eval import CLASS_NAMES

    return {
        key: float(metrics[key])
        for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll")
    } | {
        "per_class": {
            name: {
                "precision": float(metrics["precision"][index]),
                "recall": float(metrics["recall"][index]),
                "f1": float(metrics["f1"][index]),
                "support": int(metrics["support"][index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion": metrics["confusion"].tolist(),
    }


def run(cfg: Config) -> dict:
    import torch
    from dataset import build_transforms, check_split, load_split, make_loader
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params

    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs và batch_size phải >= 1")
    set_seed(cfg.seed)
    folder = run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    test_file = pred_path(cfg, "test")
    if cfg.save_test_predictions and test_file.exists():
        raise FileExistsError(f"Refusing to overwrite test predictions: {test_file}")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_stats = check_split(train_df, val_df, test_df, cfg.images_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg.backbone, pretrained=cfg.init != "scratch", num_classes=9,
        drop_rate=cfg.drop_rate, init=cfg.init,
    ).to(device)
    counts = train_df["Label"].value_counts().reindex(range(9), fill_value=0).to_numpy()
    weights = None
    if cfg.class_weight_beta is not None:
        weights = class_weights(counts, cfg.class_weight_beta).to(device)
    elif cfg.loss == "ce_weighted":
        weights = class_weights(counts, 0.0).to(device)
    criterion = build_criterion(
        cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
        weight=weights, alpha=weights if cfg.loss == "focal" else None,
    ).to(device)
    train_loader = make_loader(
        train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, sampler=cfg.sampler, num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        val_df, cfg.images_dir, build_transforms(False, cfg.img_size),
        cfg.batch_size, False, num_workers=cfg.num_workers,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    import timm
    import torchvision

    metadata = {
        "config": asdict(cfg),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pretrained_tag": getattr(model, "pretrained_tag", "unknown"),
        "split": split_stats,
    }
    (folder / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    history, epoch_seconds = [], []
    best_f1, best_epoch = -math.inf, None
    best_path = folder / "best_checkpoint.pt"
    from eval import compute_metrics

    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema,
        )
        if ema is not None:
            with ema.average_parameters(model):
                _, y_val, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        else:
            _, y_val, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        probabilities = _softmax(val_logits)
        metrics = compute_metrics(y_val, probabilities.argmax(1), probabilities)
        duration = time.perf_counter() - start
        epoch_seconds.append(duration)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_top1": metrics["top1"],
            "val_balanced_acc": metrics["balanced_acc"],
            "lr": train_metrics["lr"],
            "seconds": duration,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_macro_f1": best_f1,
            },
            folder / "last_checkpoint.pt",
        )
        print(
            f"{cfg.exp_id} epoch {epoch}/{cfg.epochs}: "
            f"train_loss={record['train_loss']:.4f}, "
            f"val_macro_f1={record['val_macro_f1']:.4f}, seconds={duration:.1f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": copy.deepcopy(model.state_dict()),
                    "ema": ema.state_dict() if ema is not None else None,
                    "macro_f1": best_f1,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if ema is not None and checkpoint["ema"] is not None:
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    filenames, y_val, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
    val_probs = _softmax(val_logits)
    val_metrics = compute_metrics(y_val, val_probs.argmax(1), val_probs)
    np.save(folder / "val_logits.npy", val_logits)
    from eval import save_predictions

    save_predictions(pred_path(cfg, "val"), filenames, y_val, val_probs)

    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = make_loader(
            test_df, cfg.images_dir, build_transforms(False, cfg.img_size),
            cfg.batch_size, False, num_workers=cfg.num_workers,
        )
        test_names, y_test, test_logits, _ = evaluate(
            model, test_loader, criterion, device, cfg.inference_method
        )
        test_probs = _softmax(test_logits)
        test_metrics = compute_metrics(y_test, test_probs.argmax(1), test_probs)
        np.save(folder / "test_logits.npy", test_logits)
        save_predictions(test_file, test_names, y_test, test_probs)

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} — {cfg.backbone} — seed {cfg.seed}")
    result = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "pretrained_tag": metadata["pretrained_tag"],
        "inference_method": cfg.inference_method,
        "params_m": count_params(model),
        "gmacs": count_gmacs(model, cfg.img_size),
        "best_epoch": best_epoch,
        "macro_f1_val": val_metrics["macro_f1"],
        "top1_val": val_metrics["top1"],
        "val_loss": val_loss,
        "seconds_per_epoch": float(np.mean(epoch_seconds)),
        "seconds_total": float(sum(epoch_seconds)),
        "test_metrics": _metric_record(test_metrics),
        "checkpoint": str(best_path),
        "curve": str(curve_path),
    }
    (folder / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_overrides(pairs: list[str]) -> dict:
    annotations = get_type_hints(Config)
    output = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override phải có dạng KEY=VALUE, nhận {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in Config.__dataclass_fields__:
            raise ValueError(f"Config không có trường {key!r}")
        if raw.lower() in {"none", "null"}:
            value = None
            output[key] = value
            continue
        target = annotations[key]
        choices = get_args(target)
        if get_origin(target) in (Union, types.UnionType):
            choices = tuple(choice for choice in choices if choice is not type(None))
            target = choices[0] if len(choices) == 1 else target
        if target is bool:
            if raw.lower() not in {"true", "false"}:
                raise ValueError(f"{key} phải là true hoặc false")
            value = raw.lower() == "true"
        elif target is int:
            value = int(raw)
        elif target is float:
            value = float(raw)
        else:
            value = raw
        output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepWeeds experiments")
    parser.add_argument("--set", nargs="+", required=True, metavar="KEY=VALUE")
    args = parser.parse_args()
    config = Config(**parse_overrides(args.set))
    print(json.dumps(run(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
