#!/usr/bin/env python3
"""Run and resume the complete DeepWeeds submission pipeline on Colab."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = ROOT / "submissions" / "2A202602524_DinhBaoHung"
CODE_DIR = SUBMISSION_DIR / "code"
DATA_IMAGES_DIR = ROOT / "data"
EPOCHS = 3
BATCH_SIZE = 8
SEEDS = (0, 1, 2)

for directory in (ROOT, CODE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def prepare_data() -> None:
    global DATA_IMAGES_DIR
    labels_dir = ROOT / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
        target = labels_dir / f"{name}.csv"
        if not target.exists():
            url = f"https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv"
            urllib.request.urlretrieve(url, target)

    drive_data = ROOT / "data"
    local_data = Path("/content/deepweeds_data") if Path("/content").is_dir() else drive_data
    local_data.mkdir(parents=True, exist_ok=True)
    count = len(list(local_data.rglob("*.jpg")))
    if count != 17509:
        if count:
            print(f"[DATA] Resuming extraction: found {count}/17509 images.", flush=True)
        archive = local_data / "images.zip"
        if not archive.exists():
            drive_archive = drive_data / "images.zip"
            if local_data != drive_data and drive_archive.exists():
                print("[DATA] Copying dataset archive from Drive to Colab local disk.", flush=True)
                shutil.copy2(drive_archive, archive)
            else:
                url = "https://zenodo.org/records/7939060/files/images.zip?download=1"
                print(f"[DATA] Downloading images from {url}", flush=True)
                urllib.request.urlretrieve(url, archive)
        digest = hashlib.md5()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != "b7b30f96d466fba86016aa5a26606e0f":
            raise RuntimeError(f"DeepWeeds archive MD5 mismatch: {digest.hexdigest()}")
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(local_data)
        count = len(list(local_data.rglob("*.jpg")))
        if count != 17509:
            raise RuntimeError(f"Expected 17,509 extracted images, found {count}.")
    DATA_IMAGES_DIR = local_data
    print(f"[DATA] Verified {count} images. Training image path: {DATA_IMAGES_DIR}", flush=True)


def make_config(exp_id: str, backbone: str, *, seed: int = 0, **overrides):
    from train import Config

    values = {
        "exp_id": exp_id,
        "seed": seed,
        "fold": 0,
        "backbone": backbone,
        "init": "finetune",
        "img_size": 224,
        "aug": "basic",
        "loss": "ce",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_backbone": 1e-4,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "warmup_epochs": 1.0,
        "amp": True,
        "num_workers": 0,
        "images_dir": str(DATA_IMAGES_DIR),
        "labels_dir": str(ROOT / "data" / "labels"),
        "out_dir": str(SUBMISSION_DIR / "runs"),
        "pred_dir": str(SUBMISSION_DIR / "predictions"),
        "curves_dir": str(SUBMISSION_DIR / "curves"),
        "save_test_predictions": False,
    }
    values.update(overrides)
    return Config(**values)


def run_exp(cfg):
    from train import run, run_dir, pred_path

    folder = run_dir(cfg)
    result_path = folder / "result.json"
    config_path = folder / "config.json"
    test_path = pred_path(cfg, "test")
    if result_path.exists() and config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8")).get("config")
        if old_config == asdict(cfg) and (not cfg.save_test_predictions or test_path.exists()):
            print(f"[RESUME] Reusing completed {cfg.exp_id}/seed{cfg.seed}", flush=True)
            return json.loads(result_path.read_text(encoding="utf-8"))
        if cfg.save_test_predictions and test_path.exists():
            raise FileExistsError(
                f"{test_path} already exists but this run has a different configuration; "
                "do not overwrite a test prediction."
            )
    elif cfg.save_test_predictions and test_path.exists():
        raise FileExistsError(f"Refusing to overwrite test predictions: {test_path}")
    print(f"[RUN] {cfg.exp_id} / {cfg.backbone} / seed {cfg.seed} / {cfg.epochs} epochs", flush=True)
    result = run(cfg)
    print(f"[OK] {cfg.exp_id}/seed{cfg.seed}: val macro-F1={result['macro_f1_val']:.4f}", flush=True)
    return result


def run_smoke() -> None:
    from train import run_dir

    cfg = make_config("SMOKE", "resnet50", epochs=1)
    result_file = run_dir(cfg) / "result.json"
    if result_file.exists():
        print("[RESUME] A completed smoke run already exists; not repeating it.", flush=True)
        result = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        result = run_exp(cfg)
    print(f"[SMOKE OK] Validation macro-F1={result['macro_f1_val']:.4f}", flush=True)


def run_backbone_comparison() -> list[dict]:
    import torch
    from benchmark import latency_report
    from model import build_model

    backbones = (
        "resnet50",
        "resnext50_32x4d",
        "convnext_tiny",
        "deit_small_patch16_224",
        "mobilenetv3_large_100",
    )
    results = []
    for index, name in enumerate(backbones, start=1):
        result = run_exp(make_config(f"B{index:02d}", name))
        model = build_model(name, pretrained=True).cuda().eval()
        checkpoint = torch.load(result["checkpoint"], map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        latency = latency_report(model, 1, 224, "fp32", "cuda", warmup=10, iters=50)
        result["latency_batch1_p50_ms"] = latency["p50"]
        (Path(result["checkpoint"]).parent / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results.sort(key=lambda row: row["macro_f1_val"], reverse=True)
    (SUBMISSION_DIR / "backbone_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


TRAINING_CONFIGS = (
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
)


def run_training_comparison(backbone: str) -> list[dict]:
    results = []
    for exp_id, overrides in TRAINING_CONFIGS:
        result = run_exp(make_config(exp_id, backbone, **overrides))
        result["changed_from_T00"] = json.dumps(overrides, sort_keys=True) or "{}"
        results.append(result)
        (SUBMISSION_DIR / "training_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    baseline = next(row["macro_f1_val"] for row in results if row["exp_id"] == "T00")
    for row in results:
        row["delta_vs_T00"] = row["macro_f1_val"] - baseline
        row["interpretation"] = "Một seed; delta mô tả, chưa ước lượng nhiễu giữa seed."
    (SUBMISSION_DIR / "training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def evaluate_inference(backbone: str, recipe: dict) -> tuple[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from benchmark import latency_report, tta_latency
    from dataset import build_transforms, load_split, make_loader
    from eval import compute_metrics
    from inference import (
        aggregate_views,
        apply_temperature,
        fit_temperature,
        predict_logits,
        view_hflip,
    )
    from model import build_model

    overrides = json.loads(recipe["changed_from_T00"])
    model = build_model(
        backbone, pretrained=overrides.get("init", "finetune") != "scratch",
        init=overrides.get("init", "finetune"),
    ).cuda().eval()
    checkpoint_path = SUBMISSION_DIR / "runs" / recipe["exp_id"] / "seed0" / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])

    _, val_df, _ = load_split(str(ROOT / "data" / "labels"), 0)
    loader = make_loader(val_df, str(DATA_IMAGES_DIR), build_transforms(False, 224), BATCH_SIZE, False, num_workers=0)
    filenames, labels, logits = predict_logits(model, loader, "cuda")
    flip_names, flip_labels, flip_logits = predict_logits(model, loader, "cuda", view_hflip)
    amp_names, amp_labels, amp_logits = predict_logits(model, loader, "cuda", amp=True)
    if filenames != flip_names or filenames != amp_names or not (
        np.array_equal(labels, flip_labels) and np.array_equal(labels, amp_labels)
    ):
        raise ValueError("Inference methods returned different validation ordering.")

    temperature = fit_temperature(logits, labels)
    inference_rows = []
    methods = (
        ("I00", "single center view", aggregate_views([logits], "prob"), 1),
        ("I02", "flip TTA: probability mean", aggregate_views([logits, flip_logits], "prob"), 2),
        ("I03", "flip TTA: logit mean", aggregate_views([logits, flip_logits], "logit"), 2),
        ("I07", "temperature scaling", apply_temperature(logits, temperature), 1),
        ("I08", "AMP inference", aggregate_views([amp_logits], "prob"), 1),
    )
    for exp_id, name, probabilities, views in methods:
        metrics = compute_metrics(labels, probabilities.argmax(1), probabilities)
        inference_rows.append({
            "exp_id": exp_id,
            "method": name,
            "checkpoint": f"{recipe['exp_id']}/seed0",
            "views": views,
            "macro_f1_val": float(metrics["macro_f1"]),
            "top1_val": float(metrics["top1"]),
            "ece_val": float(metrics["ece"]),
            "temperature": temperature if exp_id == "I07" else None,
        })

    latency_rows = [
        {"exp_id": "I00", **latency_report(model, 1, 224, "fp32", "cuda", warmup=10, iters=50)},
        {"exp_id": "I08", **latency_report(model, 1, 224, "amp", "cuda", warmup=10, iters=50)},
        {"exp_id": "I07", "method": "temperature scaling", **latency_report(
            model, 1, 224, "fp32", "cuda", warmup=10, iters=50
        )},
        {"exp_id": "I00_batch8", **latency_report(model, 8, 224, "fp32", "cuda", warmup=10, iters=50)},
        {"exp_id": "I02", "method": "flip probability TTA", **tta_latency(
            model, 2, batch_size=1, img_size=224, device="cuda", warmup=10, iters=50
        )},
        {"exp_id": "I03", "method": "flip logit TTA", **tta_latency(
            model, 2, batch_size=1, img_size=224, device="cuda", warmup=10, iters=50
        )},
    ]
    latency_by_id = {row["exp_id"]: row for row in latency_rows}
    inference_by_id = {row["exp_id"]: row for row in inference_rows}
    candidates = [
        (inference_by_id[key], latency_by_id[latency_key])
        for key, latency_key in (("I00", "I00"), ("I02", "I02"), ("I03", "I03"), ("I07", "I00"), ("I08", "I08"))
        if latency_by_id[latency_key]["p95"] <= 100
    ]
    if candidates:
        selected, latency = min(candidates, key=lambda pair: (-pair[0]["macro_f1_val"], pair[1]["p95"]))
    else:
        selected = max(inference_rows, key=lambda row: row["macro_f1_val"])
        latency = latency_by_id["I00"]
        print("[WARN] No inference method meets the 100 ms p95 budget; selecting best validation macro-F1.", flush=True)
    method_map = {"I00": "single", "I02": "hflip_prob", "I03": "hflip_logit", "I07": "single", "I08": "amp"}
    selected_method = method_map[selected["exp_id"]]
    (SUBMISSION_DIR / "inference_results.json").write_text(
        json.dumps(inference_rows, indent=2), encoding="utf-8"
    )
    (SUBMISSION_DIR / "latency_results.json").write_text(
        json.dumps(latency_rows, indent=2), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(7, 5))
    for row in inference_rows:
        latency_row = latency_by_id.get(row["exp_id"], latency_by_id["I00"])
        axis.scatter(latency_row["p50"], row["macro_f1_val"])
        axis.annotate(row["exp_id"], (latency_row["p50"], row["macro_f1_val"]))
    axis.set(xlabel="Latency p50, batch 1 (ms)", ylabel="Validation macro-F1",
             title="Accuracy-latency trade-off")
    fig.tight_layout()
    fig.savefig(SUBMISSION_DIR / "curves" / "inference_accuracy_latency.png", dpi=160)
    plt.close(fig)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[SELECTED] recipe={recipe['exp_id']} inference={selected['exp_id']} "
        f"method={selected_method} p95={latency['p95']:.2f}ms T={temperature:.4f}",
        flush=True,
    )
    return selected_method, float(latency["p95"])


def run_final(backbone: str, recipe: dict, inference_method: str) -> None:
    import numpy as np
    import torch
    from dataset import build_transforms, load_split, make_loader
    from eval import save_predictions
    from inference import apply_temperature
    from train import run

    labels_dir = ROOT / "data" / "labels"
    _, val_df, _ = load_split(str(labels_dir), 0)
    val_loader = make_loader(val_df, str(DATA_IMAGES_DIR), build_transforms(False, 224), BATCH_SIZE, False, num_workers=0)
    for seed in SEEDS:
        print(f"[FINAL] seed={seed}: train calibrated final, uncalibrated copy, and baseline.", flush=True)
        final_cfg = make_config(
            "F01uncal", backbone, seed=seed,
            **json.loads(recipe["changed_from_T00"]),
            inference_method=inference_method,
            save_test_predictions=True,
        )
        run_exp(final_cfg)
        baseline_cfg = make_config(
            "T00", backbone, seed=seed, save_test_predictions=True,
        )
        run_exp(baseline_cfg)

        # Refit calibration on this seed's validation logits; reuse its one test pass.
        from model import build_model
        from inference import fit_temperature
        model = build_model(
            backbone, pretrained=final_cfg.init != "scratch", init=final_cfg.init
        ).cuda().eval()
        folder = SUBMISSION_DIR / "runs" / "F01uncal" / f"seed{seed}"
        checkpoint = torch.load(folder / "best_checkpoint.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
        val_names, y_val, val_logits, _ = __import__("train").evaluate(
            model, val_loader, __import__("losses").build_criterion("ce"), "cuda",
            inference_method=inference_method,
        )
        temperature = fit_temperature(val_logits, y_val)
        save_predictions(
            SUBMISSION_DIR / "predictions" / f"F01_seed{seed}_val.csv",
            val_names, y_val, apply_temperature(val_logits, temperature),
        )

        raw_test_path = SUBMISSION_DIR / "predictions" / f"F01uncal_seed{seed}_test.csv"
        raw_prediction = __import__("eval").read_pred(str(raw_test_path))
        test_logits = np.load(folder / "test_logits.npy")
        if len(raw_prediction.filenames) != len(test_logits):
            raise ValueError(f"Test logits and prediction count disagree for seed {seed}.")
        save_predictions(
            SUBMISSION_DIR / "predictions" / f"F01_seed{seed}_test.csv",
            raw_prediction.filenames,
            raw_prediction.y_true,
            apply_temperature(test_logits, temperature),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_eval(latency_p95: float) -> None:
    eval_file = ROOT / "eval.py"
    predictions = SUBMISSION_DIR / "predictions"
    labels = ROOT / "data" / "labels"
    final = sorted(str(path) for path in predictions.glob("F01_seed*_test.csv"))
    baseline = sorted(str(path) for path in predictions.glob("T00_seed*_test.csv"))
    uncalibrated = sorted(str(path) for path in predictions.glob("F01uncal_seed*_test.csv"))
    final_val = sorted(str(path) for path in predictions.glob("F01_seed*_val.csv"))
    common = [
        "--test-csv", str(labels / "test_subset0.csv"),
        "--labels", str(labels / "labels.csv"),
        "--out", str(SUBMISSION_DIR / "eval_out"),
    ]
    subprocess.run(
        [sys.executable, str(eval_file), "score", "--pred", *final,
         "--tag", "F01", *common],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [sys.executable, str(eval_file), "score", "--pred", *baseline,
         "--tag", "T00", *common],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [sys.executable, str(eval_file), "score", "--pred", *uncalibrated,
         "--tag", "F01uncal", *common],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [sys.executable, str(eval_file), "grade",
         "--final", *final, "--baseline", *baseline, "--uncal", *uncalibrated,
         "--final-val", *final_val,
         "--latency-p95-ms", str(latency_p95), "--latency-method", "proper",
         "--val-csv", str(labels / "val_subset0.csv"), *common],
        cwd=ROOT, check=True,
    )


def run_all() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU is not enabled. Select Runtime > Change runtime type > T4 GPU.")

    run_smoke()
    backbones = run_backbone_comparison()
    selected_backbone = backbones[0]["backbone"]
    training = run_training_comparison(selected_backbone)
    recipe = max(training, key=lambda row: row["macro_f1_val"])
    method, latency_p95 = evaluate_inference(selected_backbone, recipe)
    run_final(selected_backbone, recipe, method)
    run_eval(latency_p95)

    from submission_tools import build_results_workbook, validate_submission, write_submission_docs
    build_results_workbook(SUBMISSION_DIR)
    write_submission_docs(SUBMISSION_DIR)
    validation = validate_submission(SUBMISSION_DIR, require_predictions=True)
    print(f"[DONE] All requested outputs generated and validated: {validation}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/resume all DeepWeeds submission experiments in Colab.")
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="epochs per full experiment (default: 3); smoke test always uses one epoch",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    global EPOCHS
    EPOCHS = args.epochs

    os.chdir(ROOT)
    requirements = CODE_DIR / "requirements.txt"
    if requirements.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            check=True,
        )
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    for folder in ("curves", "predictions", "runs"):
        (SUBMISSION_DIR / folder).mkdir(parents=True, exist_ok=True)
    prepare_data()
    if args.mode == "smoke":
        run_smoke()
    else:
        run_all()


if __name__ == "__main__":
    main()
