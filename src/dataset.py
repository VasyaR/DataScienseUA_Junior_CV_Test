"""CelebA dataset loading and preprocessing."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for CelebA.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    transform = get_transforms(config["crop_size"], config["image_size"])

    train_dataset = datasets.CelebA(
        root=config["data_dir"], split="train",
        transform=transform, download=True,
    )
    val_dataset = datasets.CelebA(
        root=config["data_dir"], split="valid",
        transform=transform, download=True,
    )
    test_dataset = datasets.CelebA(
        root=config["data_dir"], split="test",
        transform=transform, download=True,
    )

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
