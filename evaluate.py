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
        title=f"Failure Gallery - Top {n_failures} worst reconstructions\n"
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
