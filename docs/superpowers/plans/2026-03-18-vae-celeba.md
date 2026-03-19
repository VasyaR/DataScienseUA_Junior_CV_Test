# VAE for CelebA Image Generation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a convolutional VAE that reconstructs and generates CelebA face images, with full evaluation and debugging narrative.

**Architecture:** Symmetric conv encoder-decoder (~8-9M params), latent dim 256, ELBO loss with BCE reconstruction + KL divergence. KL warm-up + gradient clipping for stability. AdamW + CosineAnnealingLR.

**Tech Stack:** PyTorch 2.6, torchvision (CelebA loader), matplotlib, numpy, PyYAML, tqdm

**Spec:** `docs/superpowers/specs/2026-03-18-vae-celeba-design.md`

**Python interpreter:** `/home/beav3r/DataScienceUa/Test/env/bin/python`

---

## Chunk 1: Project Scaffolding + Dataset

### Task 1: Create project directories and config

**Files:**
- Create: `configs/default.yaml`
- Create: `src/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /home/beav3r/DataScienceUa/Test
mkdir -p configs src outputs/checkpoints outputs/images outputs/curves
```

- [ ] **Step 2: Write config file**

Create `configs/default.yaml`:

```yaml
# Dataset
dataset: celeba
data_dir: data
image_size: 64
crop_size: 148
channels: 3

# Model
latent_dim: 256
encoder_channels: [64, 128, 256, 512]
decoder_channels: [512, 256, 128, 64]

# Training
batch_size: 128
epochs: 50
lr: 3.0e-4
weight_decay: 1.0e-5
optimizer: adamw
scheduler: cosine

# Stability
kl_warmup_epochs: 20
grad_clip_max_norm: 5.0

# Reproducibility
seed: 42
num_workers: 4

# Outputs
output_dir: outputs
checkpoint_every: 10
sample_every: 5
```

- [ ] **Step 3: Create empty `src/__init__.py`**

- [ ] **Step 4: Commit**

```bash
git add configs/ src/__init__.py
git commit -m "scaffold: project structure and default config"
```

---

### Task 2: Dataset loading module

**Files:**
- Create: `src/dataset.py`

- [ ] **Step 1: Write `src/dataset.py`**

This module handles CelebA download, center-crop, resize, and DataLoader creation.

```python
"""CelebA dataset loading and preprocessing."""

import torch
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
```

- [ ] **Step 2: Test the dataset loads**

```bash
env/bin/python -c "
import yaml
from src.dataset import get_dataloaders
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
train_loader, val_loader, test_loader = get_dataloaders(config)
batch, _ = next(iter(train_loader))
print(f'Train: {len(train_loader.dataset)} images')
print(f'Val: {len(val_loader.dataset)} images')
print(f'Test: {len(test_loader.dataset)} images')
print(f'Batch shape: {batch.shape}')
print(f'Pixel range: [{batch.min():.3f}, {batch.max():.3f}]')
"
```

Expected output:
```
Train: 162770 images
Val: 19867 images
Test: 19962 images
Batch shape: torch.Size([128, 3, 64, 64])
Pixel range: [0.000, 1.000]
```

If torchvision CelebA download fails (Google Drive quota), manually download:
1. Download `img_align_celeba.zip`, `list_eval_partition.txt`, `identity_CelebA.txt`, `list_attr_celeba.txt`, `list_bbox_celeba.txt`, `list_landmarks_align_celeba.txt`
2. Place in `data/celeba/`
3. Set `download=False` in the code

- [ ] **Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: CelebA dataset loader with center crop + resize preprocessing"
```

---

## Chunk 2: Model

### Task 3: VAE model (Encoder + Decoder + Reparameterization)

**Files:**
- Create: `src/model.py`

- [ ] **Step 1: Write `src/model.py`**

```python
"""Convolutional VAE: Encoder, Decoder, and full VAE with reparameterization."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional encoder: image -> (mu, log_var).

    4 conv layers with stride 2 downsample 64x64 -> 4x4,
    then flatten and project to latent parameters.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 hidden_channels: list[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]

        layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

        flat_dim = hidden_channels[-1] * 4 * 4

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Convolutional decoder: z -> reconstructed image.

    Linear projection -> unflatten -> 4 transposed conv layers
    upsample 4x4 -> 64x64. Sigmoid output for [0,1] pixel range.
    """

    def __init__(self, out_channels: int = 3, latent_dim: int = 256,
                 hidden_channels: list[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [512, 256, 128, 64]

        self.first_channels = hidden_channels[0]
        flat_dim = hidden_channels[0] * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1],
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.extend([
            nn.ConvTranspose2d(hidden_channels[-1], out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.first_channels, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick.

    Combines Encoder and Decoder. Forward pass returns
    (reconstruction, mu, log_var) for loss computation.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 encoder_channels: list[int] = None,
                 decoder_channels: list[int] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim, encoder_channels)
        self.decoder = Decoder(in_channels, latent_dim, decoder_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon.

        Enables backpropagation through the sampling operation by
        expressing z as a deterministic function of (mu, log_var, epsilon).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate images by sampling z ~ N(0, I) and decoding."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
```

- [ ] **Step 2: Verify model shapes and parameter count**

```bash
env/bin/python -c "
import torch
from src.model import VAE

model = VAE(in_channels=3, latent_dim=256)
x = torch.randn(2, 3, 64, 64)
x_recon, mu, log_var = model(x)
print(f'Input:   {x.shape}')
print(f'Recon:   {x_recon.shape}')
print(f'mu:      {mu.shape}')
print(f'log_var: {log_var.shape}')
print(f'Recon range: [{x_recon.min():.3f}, {x_recon.max():.3f}]')

total = sum(p.numel() for p in model.parameters())
enc = sum(p.numel() for p in model.encoder.parameters())
dec = sum(p.numel() for p in model.decoder.parameters())
print(f'Params total: {total:,}, encoder: {enc:,}, decoder: {dec:,}')

samples = model.sample(4, device='cpu')
print(f'Samples: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}]')
"
```

Expected: shapes match, recon in [0,1], ~8-9M total params.

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: convolutional VAE with encoder, decoder, reparameterization"
```

---

## Chunk 3: Loss Function

### Task 4: ELBO loss with KL warm-up

**Files:**
- Create: `src/loss.py`

- [ ] **Step 1: Write `src/loss.py`**

```python
"""ELBO loss for VAE training with KL warm-up support."""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy reconstruction loss, summed over pixels.

    Treats each pixel as independent Bernoulli variable.
    Sum over pixels (C*H*W), mean over batch.

    ELBO derivation:
        Recon = -E_q(z|x)[log p(x|z)]
             = -sum [x*log(x_hat) + (1-x)*log(1-x_hat)]  (Bernoulli log-likelihood)
             = BCE(x_hat, x) summed over pixels
    """
    return F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL divergence between q(z|x) = N(mu, sigma^2 I) and p(z) = N(0, I).

    Closed-form solution:
        KL = -0.5 * sum [1 + log(sigma^2) - mu^2 - sigma^2]
           = -0.5 * sum [1 + log_var - mu^2 - exp(log_var)]

    Sum over latent dims, mean over batch.
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)


def elbo_loss(x_recon: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, log_var: torch.Tensor,
              beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full ELBO loss = Reconstruction + beta * KL.

    Args:
        x_recon: decoder output [B, C, H, W]
        x: input images [B, C, H, W]
        mu: encoder mean [B, latent_dim]
        log_var: encoder log-variance [B, latent_dim]
        beta: KL weight (0 to 1 during warm-up)

    Returns:
        (total_loss, recon_loss, kl_loss) as scalars
    """
    recon = reconstruction_loss(x_recon, x)
    kl = kl_divergence(mu, log_var)
    total = recon + beta * kl
    return total, recon, kl


def get_beta(epoch: int, warmup_epochs: int) -> float:
    """Linear KL warm-up schedule: beta = min(1, epoch / warmup_epochs).

    During early training the decoder is weak. Without warm-up, the model
    minimizes loss by collapsing q(z|x) to the prior (posterior collapse).
    Warm-up suppresses KL initially, letting the decoder learn first.
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)
```

- [ ] **Step 2: Verify loss computation**

```bash
env/bin/python -c "
import torch
from src.model import VAE
from src.loss import elbo_loss, get_beta

model = VAE(in_channels=3, latent_dim=256)
x = torch.rand(4, 3, 64, 64)
x_recon, mu, log_var = model(x)

total, recon, kl = elbo_loss(x_recon, x, mu, log_var, beta=1.0)
print(f'Total: {total.item():.1f}, Recon: {recon.item():.1f}, KL: {kl.item():.1f}')

for epoch in [0, 5, 10, 15, 20, 25]:
    print(f'Epoch {epoch:2d}: beta = {get_beta(epoch, 20):.2f}')
"
```

Expected: losses are positive scalars; beta ramps 0.00, 0.25, 0.50, 0.75, 1.00, 1.00

- [ ] **Step 3: Commit**

```bash
git add src/loss.py
git commit -m "feat: ELBO loss with BCE reconstruction, KL divergence, warm-up schedule"
```

---

## Chunk 4: Utilities

### Task 5: Visualization and logging utilities

**Files:**
- Create: `src/utils.py`

- [ ] **Step 1: Write `src/utils.py`**

```python
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
    model_eval = model
    model_eval.eval()
    with torch.no_grad():
        mu1, _ = model_eval.encoder(img1.unsqueeze(0).to(device))
        mu2, _ = model_eval.encoder(img2.unsqueeze(0).to(device))

        alphas = torch.linspace(0, 1, steps).to(device)
        interpolations = []
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            img = model_eval.decoder(z)
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
```

- [ ] **Step 2: Quick smoke test**

```bash
env/bin/python -c "
import torch
from src.utils import save_image_grid, set_seed
set_seed(42)
fake_images = torch.rand(16, 3, 64, 64)
save_image_grid(fake_images, 'outputs/images/test_grid.png', nrow=8, title='Test')
print('Saved test grid')
import os; os.remove('outputs/images/test_grid.png'); print('Cleaned up')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/utils.py
git commit -m "feat: utilities for seed, image grids, interpolation, training curves"
```

---

## Chunk 5: Training Script

### Task 6: train.py -- end-to-end training

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write `train.py`**

```python
"""End-to-end VAE training on CelebA.

Usage:
    env/bin/python train.py                          # use default config
    env/bin/python train.py --config configs/my.yaml # custom config
"""

import argparse
import os
import time

import yaml
import torch
from tqdm import tqdm

from src.model import VAE
from src.loss import elbo_loss, get_beta
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, save_image_grid, save_reconstruction_grid,
    plot_training_curves, save_history,
)


def train_one_epoch(model, loader, optimizer, beta, grad_clip, device):
    """Train for one epoch. Returns average (loss, recon, kl)."""
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_batches = 0

    for batch, _ in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        x_recon, mu, log_var = model(batch)
        loss, recon, kl = elbo_loss(x_recon, batch, mu, log_var, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


@torch.no_grad()
def validate(model, loader, beta, device):
    """Validate. Returns average (loss, recon, kl)."""
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_batches = 0

    for batch, _ in tqdm(loader, desc="Validating", leave=False):
        batch = batch.to(device)
        x_recon, mu, log_var = model(batch)
        loss, recon, kl = elbo_loss(x_recon, batch, mu, log_var, beta=beta)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train VAE on CelebA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = config["output_dir"]
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "curves"), exist_ok=True)

    # Save config snapshot alongside outputs
    with open(os.path.join(output_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    print("Loading CelebA...")
    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = VAE(
        in_channels=config["channels"],
        latent_dim=config["latent_dim"],
        encoder_channels=config["encoder_channels"],
        decoder_channels=config["decoder_channels"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # Fixed batch for reconstruction visualization across epochs
    fixed_batch, _ = next(iter(val_loader))
    fixed_batch = fixed_batch[:16].to(device)

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [],
        "beta": [], "lr": [],
    }

    print(f"\nTraining for {config['epochs']} epochs...")
    for epoch in range(1, config["epochs"] + 1):
        beta = get_beta(epoch, config["kl_warmup_epochs"])
        t0 = time.time()

        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, beta, config["grad_clip_max_norm"], device
        )
        val_loss, val_recon, val_kl = validate(model, val_loader, beta, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["beta"].append(beta)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"beta={beta:.2f} | lr={lr:.2e} | "
            f"Train: {train_loss:.1f} (R:{train_recon:.1f} K:{train_kl:.1f}) | "
            f"Val: {val_loss:.1f} (R:{val_recon:.1f} K:{val_kl:.1f}) | "
            f"{elapsed:.1f}s"
        )

        # Save sample images periodically
        if epoch % config["sample_every"] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                recon, _, _ = model(fixed_batch)
                save_reconstruction_grid(
                    fixed_batch, recon,
                    os.path.join(output_dir, "images", f"recon_epoch_{epoch:03d}.png"),
                )
                samples = model.sample(64, device)
                save_image_grid(
                    samples,
                    os.path.join(output_dir, "images", f"samples_epoch_{epoch:03d}.png"),
                    nrow=8, title=f"Samples  Epoch {epoch}",
                )

        # Save checkpoint periodically
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
            }, os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch:03d}.pt"))

    save_history(history, os.path.join(output_dir, "history.json"))
    plot_training_curves(history, output_dir)
    print(f"\nTraining complete. Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run test (1 training step on GPU)**

```bash
env/bin/python -c "
import torch
from src.model import VAE
from src.loss import elbo_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(in_channels=3, latent_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.rand(8, 3, 64, 64).to(device)
x_recon, mu, log_var = model(x)
loss, recon, kl = elbo_loss(x_recon, x, mu, log_var, beta=0.05)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
print(f'1 step OK  loss: {loss.item():.1f}, recon: {recon.item():.1f}, kl: {kl.item():.1f}')
print(f'GPU mem: {torch.cuda.memory_allocated()/1e6:.0f} MB')
"
```

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: end-to-end training script with logging, checkpoints, sample visualization"
```

---

## Chunk 6: Evaluation Script

### Task 7: evaluate.py -- metrics + image grids + failure gallery

**Files:**
- Create: `evaluate.py`

- [ ] **Step 1: Write `evaluate.py`**

```python
"""Evaluate trained VAE: metrics, reconstruction grids, samples, failure gallery.

Usage:
    env/bin/python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_050.pt
"""

import argparse
import os
import json

import torch
from tqdm import tqdm

from src.model import VAE
from src.loss import reconstruction_loss, kl_divergence
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, save_image_grid, save_reconstruction_grid,
    save_interpolation, plot_training_curves,
)


def compute_ssim_batch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute mean SSIM between two batches of images.

    Simplified SSIM using 8x8 sliding window via average pooling.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    pool = torch.nn.AvgPool2d(kernel_size=8, stride=4, padding=0)

    mu1, mu2 = pool(img1), pool(img2)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = pool(img1 ** 2) - mu1_sq
    sigma2_sq = pool(img2 ** 2) - mu2_sq
    sigma12 = pool(img1 * img2) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Compute test metrics: average recon loss, KL, SSIM."""
    model.eval()
    total_recon, total_kl, total_ssim = 0.0, 0.0, 0.0
    n_batches = 0

    all_losses = []

    for batch, _ in tqdm(test_loader, desc="Evaluating"):
        batch = batch.to(device)
        x_recon, mu, log_var = model(batch)

        recon = reconstruction_loss(x_recon, batch)
        kl = kl_divergence(mu, log_var)
        ssim = compute_ssim_batch(x_recon, batch)

        per_image_recon = torch.nn.functional.binary_cross_entropy(
            x_recon, batch, reduction="none"
        ).sum(dim=(1, 2, 3))
        all_losses.append((per_image_recon.cpu(), batch.cpu(), x_recon.cpu()))

        total_recon += recon.item()
        total_kl += kl.item()
        total_ssim += ssim
        n_batches += 1

    metrics = {
        "test_recon_loss": total_recon / n_batches,
        "test_kl_divergence": total_kl / n_batches,
        "test_ssim": total_ssim / n_batches,
    }

    return metrics, all_losses


def build_failure_gallery(all_losses: list, output_dir: str, n_failures: int = 20):
    """Find the N worst reconstructions and save them with hypotheses."""
    losses = torch.cat([item[0] for item in all_losses])
    originals = torch.cat([item[1] for item in all_losses])
    reconstructions = torch.cat([item[2] for item in all_losses])

    worst_indices = torch.argsort(losses, descending=True)[:n_failures]

    worst_originals = originals[worst_indices]
    worst_recons = reconstructions[worst_indices]
    worst_losses = losses[worst_indices]

    comparison = torch.cat([worst_originals, worst_recons], dim=0)
    save_image_grid(
        comparison,
        os.path.join(output_dir, "images", "failure_gallery.png"),
        nrow=n_failures,
        title=f"Failure Gallery  Top {n_failures} worst reconstructions\n"
              f"Top: Original | Bottom: Reconstruction\n"
              f"Loss range: [{worst_losses[-1]:.0f}, {worst_losses[0]:.0f}]",
    )

    print(f"\nFailure gallery saved ({n_failures} worst reconstructions)")
    print(f"Loss range: {worst_losses[-1]:.0f} to {worst_losses[0]:.0f}")
    print(f"Mean loss of worst: {worst_losses.mean():.0f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = config["output_dir"]
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    model = VAE(
        in_channels=config["channels"],
        latent_dim=config["latent_dim"],
        encoder_channels=config["encoder_channels"],
        decoder_channels=config["decoder_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    _, _, test_loader = get_dataloaders(config)
    print(f"Test set: {len(test_loader.dataset)} images")

    # Quantitative
    print("\n--- Quantitative Metrics ---")
    metrics, all_losses = evaluate_model(model, test_loader, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Qualitative
    print("\n--- Qualitative Evaluation ---")

    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:16].to(device)
    with torch.no_grad():
        recon, _, _ = model(test_batch)
    save_reconstruction_grid(
        test_batch, recon,
        os.path.join(output_dir, "images", "test_reconstructions.png"),
        n=16,
    )
    print("Saved: test_reconstructions.png")

    with torch.no_grad():
        samples = model.sample(64, device)
    save_image_grid(
        samples,
        os.path.join(output_dir, "images", "random_samples.png"),
        nrow=8, title="Random Samples from z ~ N(0, I)",
    )
    print("Saved: random_samples.png")

    save_interpolation(
        model, test_batch[0].cpu(), test_batch[1].cpu(),
        os.path.join(output_dir, "images", "interpolation.png"),
        steps=10, device=device,
    )
    print("Saved: interpolation.png")

    build_failure_gallery(all_losses, output_dir, n_failures=20)

    if "history" in checkpoint:
        plot_training_curves(checkpoint["history"], output_dir)
        print("Saved: training curves")

    print(f"\nAll evaluation artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: evaluation script with metrics, grids, interpolation, failure gallery"
```

---

## Chunk 7: Training Run + Debugging + README

### Task 8: Download CelebA dataset

- [ ] **Step 1: Attempt torchvision download**

```bash
env/bin/python -c "
from torchvision import datasets
ds = datasets.CelebA(root='data', split='train', download=True)
print(f'Dataset size: {len(ds)}')
"
```

If Google Drive fails, manually download and place files in `data/celeba/`.

- [ ] **Step 2: Verify dataset loads end-to-end**

```bash
env/bin/python -c "
import yaml
from src.dataset import get_dataloaders
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
train_loader, _, _ = get_dataloaders(config)
batch, _ = next(iter(train_loader))
print(f'OK: {batch.shape}, range [{batch.min():.3f}, {batch.max():.3f}]')
"
```

### Task 9: Run full training

- [ ] **Step 1: Launch training**

```bash
env/bin/python train.py --config configs/default.yaml
```

Expected: ~50 epochs, ~2-2.5 hours. Monitor:
- Reconstruction loss decreases steadily
- KL stays near 0 during warm-up, then rises
- No NaN or exploding losses

- [ ] **Step 2: Inspect intermediate outputs**

Check `outputs/images/` for reconstruction and sample grids after a few epochs.

- [ ] **Step 3: Commit artifacts**

```bash
git add outputs/
git commit -m "artifact: training run  50 epochs, checkpoints, samples, curves"
```

### Task 10: Run evaluation

- [ ] **Step 1: Run evaluate.py**

```bash
env/bin/python evaluate.py --checkpoint outputs/checkpoints/checkpoint_epoch_050.pt
```

- [ ] **Step 2: Review all outputs in outputs/**

- [ ] **Step 3: Commit evaluation artifacts**

```bash
git add outputs/
git commit -m "artifact: evaluation  metrics, grids, failure gallery, curves"
```

### Task 11: Debugging narrative

- [ ] **Step 1: Analyze training curves for issues**

Look for: posterior collapse, mode collapse, instability, overfitting.

- [ ] **Step 2: If issues found, retrain with fixes (save as configs/tuned.yaml)**

- [ ] **Step 3: Document findings in docs/debugging_notes.md**

### Task 12: Write README.md report

- [ ] **Step 1: Write README.md (1-2 pages) with all required sections**

Sections:
1. Dataset and preprocessing
2. Architecture summary + parameter counts
3. Loss design + why
4. Metrics + image grids (embedded)
5. Top issues encountered and fixes
6. Next steps (improve sharpness/diversity)

- [ ] **Step 2: Commit**

```bash
git add README.md docs/
git commit -m "docs: README report with architecture, results, debugging narrative"
```
