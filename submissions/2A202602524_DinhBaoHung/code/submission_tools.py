"""Build and validate the small deliverables from recorded experiment outputs."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


SHEET_COLUMNS = {
    "Backbones": [
        "exp_id", "backbone", "tag trọng số", "tham số (M)", "GMAC", "độ phân giải",
        "epoch", "seed", "macro-F1 val", "top-1 val", "giây/epoch", "độ trễ batch-1 (ms)", "ghi chú",
    ],
    "Training": [
        "exp_id", "backbone", "trục thay đổi (A-G)", "khác T00 ở điểm nào", "seed",
        "macro-F1 val", "top-1 val", "delta so với T00", "ghi chú",
    ],
    "Inference": [
        "exp_id", "phương pháp", "mô hình/checkpoint", "K", "macro-F1 val", "top-1 val",
        "ECE val", "p50 (ms)", "p95 (ms)", "p99 (ms)", "ảnh/s", "chi phí tương đối",
    ],
    "Final": [
        "exp_id", "cấu hình", "seed", "macro-F1 val", "macro-F1 test", "top-1 test", "ECE test",
        "macro-F1 test mean", "macro-F1 test std", "top-1 test mean", "top-1 test std",
    ],
    "PerClass": ["exp_id", "seed", "lớp", "số ảnh", "precision", "recall", "F1"],
    "Latency": [
        "exp_id", "GPU", "dtype", "batch", "độ phân giải", "gộp BN", "p50 (ms)", "p95 (ms)",
        "p99 (ms)", "ảnh/s", "PyTorch", "preprocessing",
    ],
    "Summary": [
        "hạng", "exp_id", "backbone", "macro-F1 val", "top-1 val", "tham số (M)", "GMAC",
        "giây/epoch", "ghi chú",
    ],
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_rows(submission_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((submission_dir / "runs").glob("*/*/result.json")):
        result = _read_json(path)
        config_path = path.parent / "config.json"
        result["_metadata"] = _read_json(config_path) if config_path.exists() else {}
        result["_config"] = result["_metadata"].get("config", {})
        rows.append(result)
    return rows


def _final_prediction_rows(submission_dir: Path) -> list[dict]:
    from eval import compute_metrics, read_pred

    rows = []
    for prediction_path in sorted((submission_dir / "predictions").glob("*_seed*_test.csv")):
        exp_id = prediction_path.name.split("_seed", 1)[0]
        if not (exp_id.startswith("F") or exp_id == "T00"):
            continue
        seed = read_pred(str(prediction_path)).seed
        run_path = submission_dir / "runs" / exp_id / f"seed{seed}" / "result.json"
        if not run_path.exists() and exp_id == "F01":
            run_path = submission_dir / "runs" / "F01uncal" / f"seed{seed}" / "result.json"
        metadata = _read_json(run_path) if run_path.exists() else {}
        prediction = read_pred(str(prediction_path))
        metrics = compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)
        metrics_record = {
            key: float(metrics[key])
            for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll")
        }
        metrics_record["per_class"] = {
            name: {
                "precision": float(metrics["precision"][index]),
                "recall": float(metrics["recall"][index]),
                "f1": float(metrics["f1"][index]),
                "support": int(metrics["support"][index]),
            }
            for index, name in enumerate(__import__("eval").CLASS_NAMES)
        }
        metrics_record["confusion"] = metrics["confusion"].tolist()
        row = dict(metadata)
        row.update({
            "exp_id": exp_id,
            "seed": seed,
            "test_metrics": metrics_record,
            "test_prediction": prediction_path.name,
        })
        val_path = submission_dir / "predictions" / f"{exp_id}_seed{seed}_val.csv"
        if val_path.exists():
            val_prediction = read_pred(str(val_path))
            val_metrics = compute_metrics(
                val_prediction.y_true, val_prediction.y_pred, val_prediction.probs
            )
            row["macro_f1_val"] = float(val_metrics["macro_f1"])
            row["top1_val"] = float(val_metrics["top1"])
        rows.append(row)
    return rows


def build_results_workbook(submission_dir: str | Path) -> Path:
    submission_dir = Path(submission_dir)
    results = _run_rows(submission_dir)
    if not results:
        raise FileNotFoundError(
            f"Không có runs/*/seed*/result.json dưới {submission_dir}; "
            "không thể tạo bảng số liệu nếu chưa chạy thí nghiệm thật."
        )
    backbone_results = [row for row in results if row["exp_id"].startswith("B")]
    training_results = [row for row in results if row["exp_id"].startswith("T")]
    final_results = _final_prediction_rows(submission_dir)
    backbone_table = [{
        "exp_id": row["exp_id"], "backbone": row["backbone"], "tag trọng số": row["pretrained_tag"],
        "tham số (M)": row["params_m"], "GMAC": row["gmacs"], "độ phân giải": row["_config"].get("img_size"),
        "epoch": row["best_epoch"], "seed": row["seed"], "macro-F1 val": row["macro_f1_val"],
        "top-1 val": row["top1_val"], "giây/epoch": row["seconds_per_epoch"],
        "độ trễ batch-1 (ms)": row.get("latency_batch1_p50_ms"), "ghi chú": "",
    } for row in backbone_results]
    training_by_id = {
        row["exp_id"]: row
        for row in (_read_json(submission_dir / "training_results.json")
                    if (submission_dir / "training_results.json").exists() else [])
    }
    training_table = []
    for row in training_results:
        notebook_row = training_by_id.get(row["exp_id"], {})
        change = notebook_row.get("changed_from_T00", "{}")
        if isinstance(change, str):
            try:
                change = json.loads(change)
            except json.JSONDecodeError:
                change = {"note": change}
        axis = "baseline" if not change else ", ".join(
            key for key in ("init", "aug", "mix", "loss", "sampler", "optimizer", "ema_decay")
            if key in change
        )
        training_table.append({
            "exp_id": row["exp_id"], "backbone": row["backbone"],
            "trục thay đổi (A-G)": axis,
            "khác T00 ở điểm nào": json.dumps(change, ensure_ascii=False, sort_keys=True),
            "seed": row["seed"], "macro-F1 val": row["macro_f1_val"],
            "top-1 val": row["top1_val"],
            "delta so với T00": notebook_row.get("delta_vs_T00"),
            "ghi chú": notebook_row.get("interpretation", ""),
        })
    inference_json = submission_dir / "inference_results.json"
    inference_rows = _read_json(inference_json) if inference_json.exists() else []
    latency_json = submission_dir / "latency_results.json"
    latency_rows = _read_json(latency_json) if latency_json.exists() else []
    latency_by_id = {row.get("exp_id"): row for row in latency_rows}
    inference_table = []
    for row in inference_rows:
        latency = latency_by_id.get(row["exp_id"], {})
        inference_table.append({
            "exp_id": row["exp_id"], "phương pháp": row.get("method"),
            "mô hình/checkpoint": row.get("checkpoint", ""), "K": row.get("views"),
            "macro-F1 val": row.get("macro_f1_val"), "top-1 val": row.get("top1_val"),
            "ECE val": row.get("ece_val"), "p50 (ms)": latency.get("p50"),
            "p95 (ms)": latency.get("p95"), "p99 (ms)": latency.get("p99"),
            "ảnh/s": latency.get("images_per_s"), "chi phí tương đối": None,
        })
    final_table = []
    for row in final_results:
        metrics = row.get("test_metrics") or {}
        final_table.append({
            "exp_id": row["exp_id"], "cấu hình": row["backbone"], "seed": row["seed"],
            "macro-F1 val": row["macro_f1_val"], "macro-F1 test": metrics.get("macro_f1"),
            "top-1 test": metrics.get("top1"), "ECE test": metrics.get("ece"),
        })
    for exp_id in {row["exp_id"] for row in final_table}:
        group = [row for row in final_table if row["exp_id"] == exp_id]
        f1 = [row["macro-F1 test"] for row in group if row["macro-F1 test"] is not None]
        top1 = [row["top-1 test"] for row in group if row["top-1 test"] is not None]
        import numpy as np

        summary = {
            "macro-F1 test mean": float(np.mean(f1)) if f1 else None,
            "macro-F1 test std": float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
            "top-1 test mean": float(np.mean(top1)) if top1 else None,
            "top-1 test std": float(np.std(top1, ddof=1)) if len(top1) > 1 else None,
        }
        group[-1].update(summary)
    per_class_rows = []
    for row in final_results:
        metrics = row.get("test_metrics") or {}
        classes = metrics.get("per_class", {})
        for class_name, values in classes.items():
            per_class_rows.append({
                "exp_id": row["exp_id"], "seed": row["seed"], "lớp": class_name,
                "số ảnh": values.get("support"), "precision": values.get("precision"),
                "recall": values.get("recall"), "F1": values.get("f1"),
            })
    latency_table = [{
        "exp_id": row.get("exp_id"), "GPU": row.get("gpu"), "dtype": row.get("dtype"),
        "batch": row.get("batch"), "độ phân giải": row.get("img_size"), "gộp BN": row.get("bn_fused"),
        "p50 (ms)": row.get("p50"), "p95 (ms)": row.get("p95"), "p99 (ms)": row.get("p99"),
        "ảnh/s": row.get("images_per_s"), "PyTorch": row.get("torch"),
        "preprocessing": row.get("preprocessing_included"),
    } for row in latency_rows]
    summary_table = [{
        "exp_id": row["exp_id"], "backbone": row["backbone"],
        "macro-F1 val": row["macro_f1_val"], "top-1 val": row["top1_val"],
        "tham số (M)": row["params_m"], "GMAC": row["gmacs"],
        "giây/epoch": row["seconds_per_epoch"], "ghi chú": "",
    } for row in sorted(results, key=lambda item: item["macro_f1_val"], reverse=True)[:10]]
    for rank, row in enumerate(summary_table, start=1):
        row["hạng"] = rank

    frames = {
        "Backbones": pd.DataFrame(backbone_table),
        "Training": pd.DataFrame(training_table),
        "Inference": pd.DataFrame(inference_table),
        "Final": pd.DataFrame(final_table),
        "PerClass": pd.DataFrame(per_class_rows),
        "Latency": pd.DataFrame(latency_table),
        "Summary": pd.DataFrame(summary_table),
    }
    output = submission_dir / "results.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, columns in SHEET_COLUMNS.items():
            frame = frames[sheet]
            for column in columns:
                if column not in frame.columns:
                    frame[column] = None
            frame[columns].to_excel(writer, sheet_name=sheet, index=False, freeze_panes=(1, 0))
            worksheet = writer.sheets[sheet]
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.columns:
                width = min(42, max(12, max(len(str(cell.value or "")) for cell in cells) + 2))
                worksheet.column_dimensions[cells[0].column_letter].width = width
    return output


def write_submission_docs(submission_dir: str | Path) -> tuple[Path, Path]:
    submission_dir = Path(submission_dir)
    results = _run_rows(submission_dir)
    if not results:
        raise FileNotFoundError("Cần có kết quả chạy thật trước khi sinh báo cáo.")
    final_runs = [
        row for row in _final_prediction_rows(submission_dir)
        if row["exp_id"].startswith("F") and not row["exp_id"].endswith("uncal")
    ]
    if not final_runs:
        raise ValueError("Chưa có các lần chạy chung kết Fxx; không thể kết luận kết quả cuối.")
    best_val = max(results, key=lambda row: row["macro_f1_val"])
    final_test = [row["test_metrics"]["macro_f1"] for row in final_runs
                  if row.get("test_metrics") and row["test_metrics"].get("macro_f1") is not None]
    if not final_test:
        raise ValueError("Chưa có metric test từ lần chạy chung kết.")
    import numpy as np

    test_mean = float(np.mean(final_test))
    test_std = float(np.std(final_test, ddof=1)) if len(final_test) > 1 else float("nan")
    config = best_val["_config"]
    readme = submission_dir / "README.md"
    readme.write_text(
        "# Bài nộp Lab Day 2\n\n"
        "Notebook chạy: `code/lab_day2.ipynb` (mở trong repo clone hoặc tải lên Colab/Kaggle).\n\n"
        "## Môi trường và chạy lại\n"
        f"- Python/PyTorch/torchvision/timm: "
        f"{best_val['_metadata'].get('python_version', 'n/a')} / "
        f"{best_val['_metadata'].get('torch_version', 'n/a')} / "
        f"{best_val['_metadata'].get('torchvision_version', 'n/a')} / "
        f"{best_val['_metadata'].get('timm_version', 'n/a')}.\n"
        "- Thư viện: PyTorch, torchvision, timm, fvcore, numpy, pandas, matplotlib, openpyxl.\n"
        "- Dữ liệu: `data/` và `data/labels/`; fold 0 nguyên bản.\n"
        "- Chạy notebook theo thứ tự từ trên xuống. Mỗi lần chạy ghi log/checkpoint dưới `runs/`, "
        "biểu đồ dưới `curves/`, dự đoán dưới `predictions/`.\n"
        "- Seeds vòng chung kết: 0, 1, 2. Test chỉ chạy một lần mỗi seed.\n"
        "- Chạy lại CLI: `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0`.\n"
        "- Hardware và phiên bản đầy đủ được ghi trong `runs/<exp_id>/seed<seed>/config.json`.\n",
        encoding="utf-8",
    )
    report = submission_dir / "report.md"
    report.write_text(
        "# Báo cáo Lab Day 2 — DeepWeeds\n\n"
        "## Tóm tắt\n\n"
        f"Cấu hình có macro-F1 validation cao nhất là **{best_val['exp_id']} / {best_val['backbone']}**, "
        f"macro-F1 val {best_val['macro_f1_val']:.4f}, top-1 val {best_val['top1_val']:.4f}. "
        f"Chung kết có macro-F1 test {test_mean:.4f} ± {test_std:.4f} qua {len(final_test)} seed. "
        "Đây là số liệu được tổng hợp từ các file result.json sinh bởi các lần chạy trong repo.\n\n"
        "## Thiết lập và dữ liệu\n\n"
        f"- Fold: {config.get('fold', 0)}. Cấu hình chạy chọn theo validation; không gộp val vào train.\n"
        "- Phân bố và kiểm tra giao split lưu trong `runs/<exp_id>/seed<seed>/config.json`.\n"
        "- Cấu hình chi tiết, seed, tag trọng số, thời gian và chỉ số được lưu trong `results.xlsx`.\n\n"
        "## So sánh và phân tích\n\n"
        "Xem bảng Backbones, Training, Inference, Final và Latency trong `results.xlsx`; "
        "biểu đồ theo epoch nằm trong `curves/`. Chỉ diễn giải khác biệt giữa thí nghiệm có "
        "kiểm soát một yếu tố và đối chiếu độ lệch chuẩn giữa các seed.\n\n"
        "## Kết luận, hạn chế\n\n"
        "Kết quả chỉ phản ánh fold 0, một GPU và tập dữ liệu DeepWeeds hiện tại. Chia ngẫu nhiên "
        "có thể lạc quan do ảnh cùng địa điểm/mùa; số seed và ngân sách tính toán hữu hạn. "
        "Không suy rộng kết luận ngoài miền dữ liệu này.\n",
        encoding="utf-8",
    )
    return readme, report


def validate_submission(submission_dir: str | Path, require_predictions: bool = True) -> dict:
    from eval import read_pred

    submission_dir = Path(submission_dir)
    required = [
        submission_dir / "README.md",
        submission_dir / "results.xlsx",
        submission_dir / "report.md",
        submission_dir / "code",
        submission_dir / "curves",
        submission_dir / "predictions",
    ]
    missing = [str(path) for path in required if not path.exists()]
    python_files = list((submission_dir / "code").glob("*.py"))
    unfinished = []
    marker = "NotImplemented" + "Error"
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == marker
            for node in ast.walk(tree)
        ):
            unfinished.append(str(path))
    if unfinished:
        missing.append(f"Unimplemented stubs in: {unfinished}")
    if not python_files:
        missing.append("No Python source files in code/")
    result_files = list((submission_dir / "runs").glob("*/*/result.json"))
    if not result_files:
        missing.append("No recorded experiment results")
    run_rows = [_read_json(path) for path in result_files]
    backbone_ids = {row["exp_id"] for row in run_rows if row["exp_id"].startswith("B")}
    training_ids = {row["exp_id"] for row in run_rows if row["exp_id"].startswith("T")}
    if len(backbone_ids) < 5:
        missing.append(f"Need at least five backbone experiments; found {len(backbone_ids)}")
    if len(training_ids) < 4:
        missing.append(f"Need a T00 baseline and at least three training experiments; found {len(training_ids)}")
    curves = list((submission_dir / "curves").glob("*.png"))
    if result_files and not curves:
        missing.append("No training curves")
    curve_names = {path.name for path in curves}
    for exp_id in backbone_ids | training_ids | {
        row["exp_id"] for row in run_rows if row["exp_id"].startswith("F")
    }:
        if not any(path.startswith(f"{exp_id}_") for path in curve_names):
            missing.append(f"No curve for {exp_id}")
    inference_path = submission_dir / "inference_results.json"
    if inference_path.exists():
        inference_ids = {row.get("exp_id") for row in _read_json(inference_path)}
        if len(inference_ids - {"I00"}) < 4:
            missing.append(f"Need four inference methods besides I00; found {len(inference_ids - {'I00'})}")
    else:
        missing.append("No inference_results.json")
    predictions = list((submission_dir / "predictions").glob("*_test.csv"))
    if require_predictions and not predictions:
        missing.append("No test prediction files")
    if require_predictions:
        expected = {
            f"{exp_id}_seed{seed}_test.csv"
            for exp_id in ("F01", "T00")
            for seed in (0, 1, 2)
        }
        actual = {path.name for path in predictions}
        absent = sorted(expected - actual)
        if absent:
            missing.append(f"Missing final/baseline seed predictions: {absent}")
    if missing:
        raise ValueError("Submission validation failed:\n- " + "\n- ".join(missing))
    prediction_rows = []
    for path in predictions:
        pred = read_pred(str(path))
        prediction_rows.append({"file": path.name, "n": len(pred.y_true), "seed": pred.seed})
    return {
        "experiments": len(result_files),
        "curves": len(curves),
        "test_prediction_files": prediction_rows,
    }
