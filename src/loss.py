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
