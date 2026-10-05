"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    Returns a callable criterion matching the requested loss.
    """
    import torch.nn as nn

    if kind == "ce":
        return nn.CrossEntropyLoss(
            weight=kw.get("weight"),
            label_smoothing=float(kw.get("smoothing", 0.0)),
        )
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted yêu cầu truyền `weight` từ class_weights()")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Loss không hỗ trợ: {kind}")


import torch
import torch.nn as nn


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Uses PyTorch cross-entropy with its built-in label smoothing.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải thuộc [0, 1)")
        self.smoothing = float(smoothing)
        self.loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits, target):
        return self.loss(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma phải >= 0")
        self.gamma = float(gamma)
        self.register_buffer(
            "alpha",
            None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32),
        )

    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).pow(self.gamma) * log_pt)
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target.long()]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    Returns normalized class weights. Use only training counts.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() == 0 or (counts <= 0).any():
        raise ValueError("counts phải là vector 1-D có mọi giá trị dương")
    if beta < 0 or beta >= 1:
        raise ValueError("beta phải thuộc [0, 1)")
    if beta == 0:
        weights = counts.reciprocal()
    else:
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    Returns a mixed batch and its paired targets with the realized CutMix area ratio.
    """
    if alpha <= 0:
        raise ValueError("alpha phải > 0")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode phải là 'mixup' hoặc 'cutmix'")
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("Mixup/CutMix cần ít nhất hai ảnh và nhãn cùng batch")
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[permutation], (y_a, y_b, lam)

    height, width = x.shape[-2:]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + (cut_h + 1) // 2, height)
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + (cut_w + 1) // 2, width)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Accuracy on a mixed batch is not directly meaningful; evaluate on validation.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
