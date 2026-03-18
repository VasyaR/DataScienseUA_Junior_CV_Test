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
                    nrow=8, title=f"Samples - Epoch {epoch}",
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
