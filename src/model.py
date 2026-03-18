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
