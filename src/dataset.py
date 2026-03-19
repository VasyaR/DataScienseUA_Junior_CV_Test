"""CelebA dataset loading and preprocessing with disk-backed tensor caching.

First run builds .pt cache files (~10 min one-time cost).
Subsequent runs memory-map from disk — fast random access without loading into RAM.
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def get_transforms(crop_size: int = 148, image_size: int = 64):
    """CelebA preprocessing: center crop to remove background, resize to target.

    Args:
        crop_size: center crop size (148 keeps face, removes background)
        image_size: final resize target (64x64 for our architecture)
    """
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(image_size),
        transforms.ToTensor(),  # scales to [0, 1]
    ])


class CelebADataset(datasets.CelebA):
    """CelebA with integrity check bypassed for Kaggle-sourced data."""

    def _check_integrity(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))


class CachedDataset(Dataset):
    """Memory-mapped tensor dataset. Data lives on disk, OS pages in on demand."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx], 0  # 0 = dummy label


def _build_cache(config: dict, split: str) -> str:
    """Build .pt cache file for a split. Returns cache path."""
    cache_dir = os.path.join(config["data_dir"], "celeba_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}.pt")

    if os.path.exists(cache_path):
        return cache_path

    print(f"  Building cache for {split} split (one-time cost)...")
    transform = get_transforms(config["crop_size"], config["image_size"])
    dataset = CelebADataset(
        root=config["data_dir"], split=split,
        transform=transform, download=False,
    )

    # Pre-allocate and fill one image at a time (low peak RAM)
    n = len(dataset)
    c, h, w = config["channels"], config["image_size"], config["image_size"]
    images = torch.empty(n, c, h, w)
    for i in tqdm(range(n), desc=f"  Caching {split}"):
        images[i] = dataset[i][0]

    torch.save(images, cache_path)
    size_gb = images.element_size() * images.nelement() / 1e9
    print(f"  Saved {cache_path} ({images.shape}, {size_gb:.1f} GB)")

    # Free RAM immediately
    del images
    return cache_path


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for CelebA.

    First run caches all images as .pt files (~2.4 GB total on disk).
    All runs memory-map from disk — near-RAM speed, minimal RAM usage.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("Preparing datasets...")

    # Build caches if needed (one-time)
    train_path = _build_cache(config, "train")
    val_path = _build_cache(config, "valid")
    test_path = _build_cache(config, "test")

    # Load as memory-mapped: data stays on disk, OS pages in on access
    print("  Memory-mapping cached tensors...")
    train_data = torch.load(train_path, weights_only=True, mmap=True)
    val_data = torch.load(val_path, weights_only=True, mmap=True)
    test_data = torch.load(test_path, weights_only=True, mmap=True)

    train_dataset = CachedDataset(train_data)
    val_dataset = CachedDataset(val_data)
    test_dataset = CachedDataset(test_data)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
