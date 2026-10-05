"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Split files are read unchanged; the official fold CSVs may omit `Species`, which
is joined in memory from `labels.csv` when available.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import random
import numpy as np

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_image_index(images_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in images_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def _resolve_image(images_dir: Path, filename: str,
                   index: dict[str, list[Path]] | None = None) -> Path | None:
    candidate = images_dir / filename
    if candidate.is_file():
        return candidate
    basename = Path(filename).name
    direct_basename = images_dir / basename
    if direct_basename.is_file():
        return direct_basename
    matches = (index if index is not None else _build_image_index(images_dir)).get(basename, [])
    if len(matches) > 1:
        raise ValueError(f"Filename không duy nhất dưới {images_dir}: {filename}")
    return matches[0] if matches else None


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    Split CSVs are never filtered, rewritten, or re-partitioned.
    """
    if not 0 <= fold <= 4:
        raise ValueError(f"fold phải thuộc 0..4, nhận {fold}")
    labels_dir = Path(labels_dir)
    species_by_label = {index: name for index, name in enumerate(CLASS_NAMES)}
    labels_path = labels_dir / "labels.csv"
    if labels_path.is_file():
        full_labels = pd.read_csv(labels_path)
        if not {"Label", "Species"}.issubset(full_labels.columns):
            raise ValueError(f"{labels_path}: cần cột Label và Species")
        species_rows = full_labels[["Label", "Species"]].drop_duplicates()
        if species_rows["Label"].duplicated().any():
            raise ValueError(f"{labels_path}: mỗi Label phải ánh xạ tới đúng một Species")
        species_by_label = dict(zip(
            species_rows["Label"].astype(int), species_rows["Species"].astype(str)
        ))
    frames = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy split CSV: {path}")
        frame = pd.read_csv(path)
        required = {"Filename", "Label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: thiếu cột {sorted(required - set(frame.columns))}")
        if frame.empty:
            raise ValueError(f"{path}: split rỗng")
        if frame["Filename"].isna().any() or frame["Filename"].duplicated().any():
            raise ValueError(f"{path}: Filename rỗng hoặc trùng")
        frame = frame.copy()
        frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
        if not frame["Label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{path}: Label phải nằm trong khoảng 0..{NUM_CLASSES - 1}")
        if "Species" not in frame.columns:
            frame["Species"] = frame["Label"].map(species_by_label)
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Checks pairwise disjointness, the expected dataset cardinality, and that
    every listed image is available below `images_dir`.
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {name: set(frame["Filename"].astype(str)) for name, frame in frames.items()}
    overlaps = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    nonempty = {key: value for key, value in overlaps.items() if value}
    if nonempty:
        detail = {key: len(value) for key, value in nonempty.items()}
        raise ValueError(f"Các split có Filename giao nhau: {detail}")

    all_names = set.union(*names.values())
    if len(all_names) != 17_509:
        raise ValueError(f"Hợp các split phải có đúng 17.509 ảnh, hiện có {len(all_names)}")

    images_dir = Path(images_dir)
    image_index = _build_image_index(images_dir)
    missing = [
        name for name in all_names
        if _resolve_image(images_dir, name, image_index) is None
    ]
    if missing:
        sample = missing[:10]
        raise FileNotFoundError(
            f"Thiếu {len(missing)} ảnh trong {images_dir}; ví dụ: {sample}"
        )

    per_class = {
        split: {
            CLASS_NAMES[label]: int(count)
            for label, count in frame["Label"].value_counts().reindex(
                range(NUM_CLASSES), fill_value=0
            ).items()
        }
        for split, frame in frames.items()
    }
    result = {
        "n": {split: len(frame) for split, frame in frames.items()},
        "per_class": per_class,
        "overlap": {key: len(value) for key, value in overlaps.items()},
        "union": len(all_names),
        "images_dir": str(images_dir.resolve()),
    }
    print(result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    Vertical flipping is intentionally excluded for field images.
    """
    from torchvision import transforms

    if img_size < 32:
        raise ValueError(f"img_size phải >= 32, nhận {img_size}")
    if train:
        operations = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
        if aug == "color":
            operations.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            operations.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            operations.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug != "basic":
            raise ValueError(f"Augmentation không hỗ trợ: {aug}")
    else:
        operations = [
            transforms.Resize(max(256, img_size)),
            transforms.CenterCrop(img_size),
        ]
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)


from torch.utils.data import Dataset


def _seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    Image paths resolve directly or recursively by basename when the CSV stores a path.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.image_index = _build_image_index(self.images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = _resolve_image(self.images_dir, filename, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh {filename} dưới {self.images_dir}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    Validation and test preserve DataFrame order; balanced sampling is train-only.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler phải None hoặc 'balanced', nhận {sampler!r}")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size phải >= 1 và num_workers phải >= 0")
    if sampler is not None and not train:
        raise ValueError("Sampler chỉ được dùng cho tập train")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    sampler_obj = None
    shuffle = bool(train)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        sample_weights = df["Label"].map(lambda label: 1.0 / counts[label]).to_numpy()
        sampler_obj = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train and len(dataset) >= batch_size),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
