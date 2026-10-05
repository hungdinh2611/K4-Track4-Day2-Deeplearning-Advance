"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Inference uses eval mode and inference_mode; returned logits are NumPy arrays.
    """
    import numpy as np
    import torch

    model = model.to(device).eval()
    use_amp = bool(amp and torch.device(device).type == "cuda")
    filenames, labels, logits = [], [], []
    with torch.inference_mode():
        for images, targets, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(device_type=torch.device(device).type,
                                dtype=torch.float16, enabled=use_amp):
                output = model(images)
            logits.append(output.float().cpu().numpy())
            labels.append(targets.numpy())
            filenames.extend(list(names))
    if not logits:
        raise ValueError("Loader không có dữ liệu")
    return filenames, np.concatenate(labels), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    import torch

    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Trả về 5 crop góc/giữa và các bản lật ngang tương ứng."""
    import torch

    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError(f"crop phải thuộc [1, {min(height, width)}], nhận {crop}")
    offsets = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    crops = [x[:, :, top:top + crop, left:left + crop] for top, left in offsets]
    return crops + [torch.flip(crop_batch, dims=(-1,)) for crop_batch in crops]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    import torch.nn.functional as functional

    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes phải là dãy số nguyên dương")
    return [
        functional.interpolate(
            x, size=(int(size), int(size)), mode="bilinear", align_corners=False,
            antialias=True,
        )
        for size in sizes
    ]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    Returns normalized probabilities.
    """
    import numpy as np
    import torch

    if space not in {"prob", "logit"}:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    if not logits_per_view:
        raise ValueError("Cần ít nhất một view để gộp")
    if isinstance(logits_per_view[0], torch.Tensor):
        stacked = torch.stack(logits_per_view)
        if space == "prob":
            return torch.softmax(stacked, dim=-1).mean(dim=0)
        return torch.softmax(stacked.mean(dim=0), dim=-1)
    stacked = np.stack(logits_per_view)
    if space == "prob":
        shifted = stacked - stacked.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return probabilities.mean(axis=0)
    mean_logits = stacked.mean(axis=0)
    mean_logits -= mean_logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(mean_logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    import numpy as np

    if not list_of_probs:
        raise ValueError("Cần ít nhất một bộ xác suất")
    arrays = [np.asarray(probabilities, dtype=np.float64) for probabilities in list_of_probs]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Các bộ xác suất ensemble phải cùng shape và thứ tự ảnh")
    result = np.mean(arrays, axis=0)
    row_sums = result.sum(axis=-1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Tổng xác suất ensemble không dương")
    return result / row_sums


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Temperature is fitted by minimizing validation negative log likelihood.
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional

    logits = torch.as_tensor(val_logits, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(val_labels, dtype=torch.long, device="cpu")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel():
        raise ValueError("val_logits phải (N,K), val_labels phải (N,) và cùng N")
    if not torch.isfinite(logits).all() or labels.numel() == 0:
        raise ValueError("Validation logits/labels rỗng hoặc không hữu hạn")

    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(min=np.log(0.05), max=np.log(20.0)).exp()
        loss = functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(min=np.log(0.05), max=np.log(20.0)).exp())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    import numpy as np
    import torch

    if not np.isfinite(T) or T <= 0:
        raise ValueError("T phải là số hữu hạn dương")
    if isinstance(logits, torch.Tensor):
        return torch.softmax(logits / T, dim=-1)
    logits = np.asarray(logits, dtype=np.float64) / T
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    For each adjacent Conv2d/BatchNorm2d pair, replace the pair with its fused convolution
    and verify output error on a representative input.
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    import copy
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    fused_model = copy.deepcopy(model).eval()
    maximum_error = 0.0
    fused_count = 0

    def fuse_children(module):
        nonlocal maximum_error, fused_count
        children = list(module.named_children())
        for index, (name, child) in enumerate(children):
            fuse_children(child)
            if index + 1 >= len(children):
                continue
            next_name, next_child = children[index + 1]
            if isinstance(child, torch.nn.Conv2d) and isinstance(next_child, torch.nn.BatchNorm2d):
                sample = torch.randn(
                    1, child.in_channels, 32, 32,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                with torch.inference_mode():
                    expected = next_child(child(sample))
                    fused_conv = fuse_conv_bn_eval(child, next_child)
                    actual = fused_conv(sample)
                maximum_error = max(maximum_error, float((expected - actual).abs().max().item()))
                setattr(module, name, fused_conv)
                setattr(module, next_name, torch.nn.Identity())
                fused_count += 1

    fuse_children(fused_model)
    fused_model.eval()
    if fused_count:
        parameter = next(fused_model.parameters())
        sample = torch.randn(
            1, 3, 224, 224, device=parameter.device, dtype=parameter.dtype,
        )
        with torch.inference_mode():
            expected = model.eval()(sample)
            actual = fused_model(sample)
        maximum_error = max(maximum_error, float((expected - actual).abs().max().item()))
        if maximum_error > 1e-5:
            raise RuntimeError(f"Conv-BN fusion sai khác quá lớn: {maximum_error:.3e}")
    print(f"Fused {fused_count} Conv-BN pairs; max output error: {maximum_error:.3e}")
    return fused_model
