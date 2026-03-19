# Architecture Design

## Overview
Convolutional VAE for CelebA 64×64 RGB images. Symmetric encoder-decoder with reparameterization trick. ~8-9M parameters.

## Encoder
Compresses 64×64×3 image into latent vector (mean + log-variance).

```
Input: [B, 3, 64, 64]
  │
  ├─ Conv2d(3→64, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU   → [B, 64, 32, 32]
  ├─ Conv2d(64→128, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU → [B, 128, 16, 16]
  ├─ Conv2d(128→256, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU→ [B, 256, 8, 8]
  ├─ Conv2d(256→512, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU→ [B, 512, 4, 4]
  │
  ├─ Flatten → [B, 8192]
  ├─ Linear(8192 → 256) → mu
  └─ Linear(8192 → 256) → log_var
```

### Design choices
- **4×4 kernels, stride 2, padding 1** — standard downsampling. Each layer halves spatial dims: 64→32→16→8→4.
- **Channels 64→128→256→512** — doubles per layer. More channels as spatial dims shrink keeps information capacity roughly constant.
- **BatchNorm** — stabilizes training by normalizing activations. Important in VAEs where the loss landscape is tricky.
- **LeakyReLU (slope=0.2)** — unlike ReLU, doesn't kill gradients for negative inputs. Common in generative models.
- **Latent dim 256** — large enough for face details (identity, expression, lighting), small enough for KL to regularize.

## Reparameterization Trick
```
z = mu + exp(0.5 * log_var) * epsilon,  where epsilon ~ N(0, I)
```
Can't backpropagate through sampling. This rewrites the random sample as a deterministic function of (mu, log_var, epsilon), making it differentiable.

## Decoder
Reconstructs 64×64×3 image from latent vector z. Mirrors encoder exactly.

```
Input: z [B, 256]
  │
  ├─ Linear(256 → 8192) + LeakyReLU
  ├─ Unflatten → [B, 512, 4, 4]
  │
  ├─ ConvTranspose2d(512→256, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU → [B, 256, 8, 8]
  ├─ ConvTranspose2d(256→128, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU → [B, 128, 16, 16]
  ├─ ConvTranspose2d(128→64, 4×4, stride=2, pad=1) + BatchNorm + LeakyReLU  → [B, 64, 32, 32]
  ├─ ConvTranspose2d(64→3, 4×4, stride=2, pad=1) + Sigmoid                  → [B, 3, 64, 64]
```

### Design choices
- **Symmetric to encoder** — easier to debug, ensures decoder has enough capacity.
- **ConvTranspose2d** — learned upsampling. Each layer doubles spatial dims: 4→8→16→32→64.
- **Sigmoid output** — pixels in [0,1], matching input normalization. Required for BCE loss.
- **No BatchNorm on last layer** — BatchNorm before Sigmoid would fight with it.

## Parameter Count Estimate
| Component | Params |
|-----------|--------|
| Encoder convs | ~1.2M |
| Encoder linear (mu + logvar) | ~4.2M |
| Decoder linear | ~2.1M |
| Decoder convs | ~1.2M |
| **Total** | **~8-9M** |

Fits in ~2-3 GB VRAM at batch size 128.
