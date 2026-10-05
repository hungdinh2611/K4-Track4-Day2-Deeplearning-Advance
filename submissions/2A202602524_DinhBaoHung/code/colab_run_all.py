#!/usr/bin/env python3
"""One-click runner for the DeepWeeds submission in Colab.

This script intentionally mirrors the notebook workflow so the same repo can be
replayed end-to-end in a single command on a Colab GPU.

Example:
    python submissions/2A202602524_DinhBaoHung/code/colab_run_all.py --mode all
    python submissions/2A202602524_DinhBaoHung/code/colab_run_all.py --mode smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "submissions" / "2A202602524_DinhBaoHung" / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def ensure_repo_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / "eval.py").exists():
        root = root.parent
    if not (root / "eval.py").exists():
        return ROOT
    return root


def install_requirements() -> None:
    requirement_file = CODE_DIR / "requirements.txt"
    if requirement_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirement_file)],
            check=False,
        )


def ensure_data_dirs() -> None:
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "labels").mkdir(parents=True, exist_ok=True)
    for folder in [
        ROOT / "submissions" / "2A202602524_DinhBaoHung" / "curves",
        ROOT / "submissions" / "2A202602524_DinhBaoHung" / "predictions",
        ROOT / "submissions" / "2A202602524_DinhBaoHung" / "runs",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def prepare_data() -> None:
    labels_dir = ROOT / "data" / "labels"
    for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
        target = labels_dir / f"{name}.csv"
        if not target.exists():
            import urllib.request
            url = f"https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv"
            urllib.request.urlretrieve(url, target)

    if not (ROOT / "data" / "images").exists():
        image_count = len(list((ROOT / "data").rglob("*.jpg")))
        if image_count == 0:
            print("[INFO] Data directory is empty. Mount your dataset or place it under data/ before running full experiments.")


def src_cfg(exp_id: str, backbone: str, **overrides):
    from train import Config

    base = {
        "exp_id": exp_id,
        "seed": 0,
        "fold": 0,
        "backbone": backbone,
        "init": "finetune",
        "drop_rate": 0.0,
        "img_size": 224,
        "aug": "basic",
        "sampler": None,
        "mix": None,
        "mix_alpha": 1.0,
        "loss": "ce",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
        "class_weight_beta": None,
        "inference_method": "single",
        "epochs": 12,
        "batch_size": 8,
        "lr_backbone": 1e-4,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "optimizer": "adamw",
        "momentum": 0.9,
        "warmup_epochs": 1.0,
        "ema_decay": None,
        "amp": True,
        "num_workers": 0,
        "images_dir": "data",
        "labels_dir": "data/labels",
        "out_dir": str(ROOT / "submissions" / "2A202602524_DinhBaoHung" / "runs"),
        "pred_dir": str(ROOT / "submissions" / "2A202602524_DinhBaoHung" / "predictions"),
        "curves_dir": str(ROOT / "submissions" / "2A202602524_DinhBaoHung" / "curves"),
        "save_test_predictions": False,
    }
    base.update(overrides)
    return Config(**base)


def run_exp(cfg) -> dict:
    from train import run

    print(f"[RUN] {cfg.exp_id} backbone={cfg.backbone} seed={cfg.seed}")
    result = run(cfg)
    print(f"[OK] {cfg.exp_id}: val macro-F1={result.get('macro_f1_val')}")
    return result


def run_smoke() -> dict:
    cfg = src_cfg("SMOKE", "resnet50", epochs=1, batch_size=8, save_test_predictions=False)
    return run_exp(cfg)


def run_backbones() -> list[dict]:
    backbones = [
        "resnet50",
        "resnext50_32x4d",
        "convnext_tiny",
        "deit_small_patch16_224",
        "mobilenetv3_large_100",
    ]
    return [run_exp(src_cfg(f"B{idx:02d}", name, seed=0)) for idx, name in enumerate(backbones, start=1)]


def run_training_ablation(selected_backbone: str) -> list[dict]:
    configs = [
        ("T00", {}),
        ("T01", {"init": "frozen"}),
        ("T02", {"init": "scratch"}),
        ("T03", {"aug": "color"}),
        ("T04", {"mix": "cutmix"}),
        ("T05", {"loss": "ls", "label_smoothing": 0.1}),
        ("T06", {"loss": "ce_weighted"}),
        ("T07", {"sampler": "balanced"}),
        ("T08", {"ema_decay": 0.999}),
        ("T09", {"mix": "cutmix", "loss": "ls", "label_smoothing": 0.1, "ema_decay": 0.999}),
    ]
    results = []
    for exp_id, overrides in configs:
        cfg = src_cfg(exp_id, selected_backbone, **overrides)
        results.append(run_exp(cfg))
    return results


def run_final_seeds(selected_backbone: str, final_exp_id: str) -> list[dict]:
    results = []
    for seed in [0, 1, 2]:
        cfg = src_cfg(final_exp_id, selected_backbone, seed=seed, save_test_predictions=True)
        results.append(run_exp(cfg))
    return results


def generate_workbook() -> None:
    submission_dir = ROOT / "submissions" / "2A202602524_DinhBaoHung"
    if not (submission_dir / "runs").exists():
        return
    from submission_tools import build_results_workbook

    workbook = build_results_workbook(submission_dir)
    print(f"[WORKBOOK] created {workbook}")


def run_eval() -> None:
    subprocess.run([sys.executable, str(ROOT / "eval.py"), "score"], check=False)
    subprocess.run([sys.executable, str(ROOT / "eval.py"), "grade"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepWeeds submission pipeline in Colab.")
    parser.add_argument("--mode", choices=["smoke", "backbones", "training", "final", "all"], default="smoke")
    args = parser.parse_args()

    os.chdir(ROOT)
    install_requirements()
    ensure_data_dirs()
    prepare_data()

    if args.mode == "smoke":
        run_smoke()
        return
    if args.mode == "backbones":
        run_backbones()
        return
    if args.mode == "training":
        selected = run_backbones()[0]["backbone"]
        run_training_ablation(selected)
        return
    if args.mode == "final":
        selected = run_backbones()[0]["backbone"]
        run_final_seeds(selected, "F01")
        return

    smoke = run_smoke()
    print("[INFO] Smoke run complete.")
    backbones = run_backbones()
    selected_backbone = sorted(backbones, key=lambda row: row.get("macro_f1_val", -1), reverse=True)[0]["backbone"]
    training = run_training_ablation(selected_backbone)
    final = run_final_seeds(selected_backbone, "F01")
    generate_workbook()
    run_eval()
    print("[DONE] Full Colab pipeline finished. Check submission outputs under submissions/2A202602524_DinhBaoHung/")


if __name__ == "__main__":
    main()
