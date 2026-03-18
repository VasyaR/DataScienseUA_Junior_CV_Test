"""Utility functions: seed setting, image grid saving, training curve plotting."""

import os
import random
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, title: str = None):
    """Save a batch of images as a grid.

    Args:
        images: [N, C, H, W] tensor in [0, 1]
        path: output file path
        nrow: images per row
        title: optional title above the grid
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(images.cpu().detach(), nrow=nrow, padding=2, normalize=False)
    fig, ax = plt.subplots(1, 1, figsize=(nrow * 1.5, (len(images) // nrow + 1) * 1.5))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_reconstruction_grid(originals: torch.Tensor, reconstructions: torch.Tensor,
                              path: str, n: int = 8):
    """Save side-by-side: originals (top row) vs reconstructions (bottom row)."""
    originals = originals[:n].cpu().detach()
    reconstructions = reconstructions[:n].cpu().detach()
    comparison = torch.cat([originals, reconstructions], dim=0)
    save_image_grid(comparison, path, nrow=n, title="Top: Original  |  Bottom: Reconstruction")


def save_interpolation(model, img1: torch.Tensor, img2: torch.Tensor,
                        path: str, steps: int = 10, device: torch.device = "cpu"):
    """Interpolate between two images in latent space."""
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encoder(img1.unsqueeze(0).to(device))
        mu2, _ = model.encoder(img2.unsqueeze(0).to(device))

        alphas = torch.linspace(0, 1, steps).to(device)
        interpolations = []
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            img = model.decoder(z)
            interpolations.append(img.squeeze(0))

        interpolations = torch.stack(interpolations)
    save_image_grid(interpolations, path, nrow=steps, title="Latent Space Interpolation")


def plot_training_curves(history: dict, output_dir: str):
    """Plot and save training curves: total loss, reconstruction, KL."""
    curves_dir = os.path.join(output_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Total loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Recon + beta * KL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(curves_dir, "total_loss.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Reconstruction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_recon"], label="Train")
    ax.plot(epochs, history["val_recon"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss (BCE)")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(curves_dir, "recon_loss.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # KL divergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_kl"], label="Train KL")
    ax.plot(epochs, history["val_kl"], label="Val KL")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["beta"], "k--", alpha=0.5, label="beta")
    ax2.set_ylabel("beta")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence + beta Schedule")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(curves_dir, "kl_divergence.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_history(history: dict, path: str):
    """Save training history to JSON for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
