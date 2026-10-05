"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Module đã triển khai các thao tác backbone, tối ưu hoá và ước lượng MAC.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    `init` controls whether the backbone is pretrained and/or frozen; the
    selected timm weight identifier is retained as `model.pretrained_tag`.
    """
    import timm

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"init phải là scratch, frozen hoặc finetune; nhận {init!r}")
    use_pretrained = bool(pretrained and init != "scratch")
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    cfg = getattr(model, "pretrained_cfg", None) or {}
    model.pretrained_tag = cfg.get("tag") or cfg.get("hf_hub_id") or (
        "pretrained-unspecified" if use_pretrained else "scratch"
    )
    model.backbone_name = name
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    Frozen backbone modules, including their BatchNorm layers, remain in eval mode
    when the training loop re-enters train mode.
    """
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    if not head_ids:
        raise ValueError("Không tìm thấy tham số của classifier để đóng băng backbone")
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    for module in model.modules():
        if module is not model and all(not parameter.requires_grad for parameter in module.parameters()):
            module.eval()
    model._backbone_frozen = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    Frozen parameters are excluded, and norm/bias parameters avoid weight decay.
    """
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    groups: dict[tuple[float, float], list] = {}
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_head = id(parameter) in head_ids
        lr = lr_head if is_head else lr_backbone
        decay = weight_decay if is_head or parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    if not groups:
        raise ValueError("Model không có tham số trainable")
    return [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in groups.items()
    ]


def count_params(model) -> float:
    """Số tham số tính theo triệu, gồm cả tham số bị đóng băng."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    fvcore is preferred; thop is used as an explicit fallback if fvcore is absent.
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        parameter = next(model.parameters())
        device = parameter.device
        was_training = model.training
        model.eval()
        try:
            sample = __import__("torch").zeros(1, 3, img_size, img_size, device=device)
            # fvcore's convolution handlers count one multiply-accumulate as one FLOP.
            return float(FlopCountAnalysis(model, sample).total() / 1e9)
        finally:
            model.train(was_training)
    except ImportError:
        try:
            from thop import profile
        except ImportError as error:
            raise RuntimeError(
                "Đếm GMAC cần fvcore hoặc thop; cài một trong hai thư viện."
            ) from error
        import torch

        parameter = next(model.parameters())
        was_training = model.training
        model.eval()
        try:
            sample = torch.zeros(1, 3, img_size, img_size, device=parameter.device)
            macs, _ = profile(model, inputs=(sample,), verbose=False)
            return float(macs / 1e9)
        finally:
            model.train(was_training)
