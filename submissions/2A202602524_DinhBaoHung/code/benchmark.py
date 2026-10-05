"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    import time
    import numpy as np

    if warmup < 10:
        raise ValueError("Benchmark cần ít nhất 10 lượt warmup")
    if iters < 50:
        raise ValueError("Benchmark cần ít nhất 50 lượt đo")
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "p99": float(np.percentile(timings, 99)),
        "mean": float(np.mean(timings)),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    Reports synchronized latency quantiles, throughput, device, dtype, and library version.
    """
    import torch

    device_obj = torch.device(device)
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là 'fp32', 'amp' hoặc 'fp16'")
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, nhưng CUDA không khả dụng")
    if dtype in {"amp", "fp16"} and device_obj.type != "cuda":
        raise ValueError(f"dtype={dtype} chỉ được benchmark trên CUDA")

    model = model.to(device_obj).eval()
    if dtype == "fp16":
        model = model.half()
    parameter = next(model.parameters(), None)
    input_dtype = torch.float16 if dtype == "fp16" else (
        parameter.dtype if parameter is not None else torch.float32
    )
    inputs = torch.randn(batch_size, 3, img_size, img_size, device=device_obj, dtype=input_dtype)
    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None

    def forward():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return model(inputs)
            return model(inputs)

    timing = bench(forward, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU"
    return {
        "gpu": gpu,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        **timing,
        "images_per_s": batch_size / (timing["p50"] / 1000.0),
        "torch": torch.__version__,
        "preprocessing_included": False,
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo độ trễ của K forward views và báo thời gian xấp xỉ mỗi view."""
    import torch

    device = kw.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng để đo latency TTA")
    model = model.to(device).eval()
    inputs = torch.randn(
        batch_size := kw.get("batch_size", 1),
        3,
        kw.get("img_size", 224),
        kw.get("img_size", 224),
        device=device,
    )
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else None

    def forward_views():
        with torch.inference_mode():
            for _ in range(k_views):
                model(inputs)

    timing = bench(
        forward_views,
        warmup=kw.get("warmup", 10),
        iters=kw.get("iters", 100),
        sync=sync,
    )
    return {
        "batch": batch_size,
        "k_views": k_views,
        **timing,
        "approx_single_view_p50_ms": timing["p50"] / k_views,
    }
