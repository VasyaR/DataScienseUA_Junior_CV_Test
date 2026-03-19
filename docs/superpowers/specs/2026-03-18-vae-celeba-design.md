# Design Spec: Convolutional VAE for CelebA Image Generation

## Goal
Train a convolutional VAE that reconstructs CelebA face images and generates plausible new faces by sampling from the latent space. Demonstrate debugging maturity with quantitative + qualitative evidence.

## Dataset
- **CelebA** — 202,599 celebrity face images, RGB
- **Preprocessing:** center crop 148×148 → resize to 64×64, normalize to [0,1]
- **Why CelebA:** Faces are low-variance and structured — VAEs produce the best visual results here. Still RGB 64×64, not trivially easy. Rich failure modes for the debugging narrative.
- **Loader:** `torchvision.datasets.CelebA` with `gdown` for download

## Architecture (~8-9M params)

### Encoder
```
[B,3,64,64] → Conv(3→64,4×4,s2,p1)+BN+LReLU → [B,64,32,32]
            → Conv(64→128,4×4,s2,p1)+BN+LReLU → [B,128,16,16]
            → Conv(128→256,4×4,s2,p1)+BN+LReLU → [B,256,8,8]
            → Conv(256→512,4×4,s2,p1)+BN+LReLU → [B,512,4,4]
            → Flatten → [B,8192]
            → Linear(8192→256) → mu
            → Linear(8192→256) → log_var
```

### Reparameterization
```
z = mu + exp(0.5 * log_var) * epsilon,  epsilon ~ N(0,I)
```

### Decoder
```
[B,256] → Linear(256→8192)+LReLU → Unflatten → [B,512,4,4]
        → ConvT(512→256,4×4,s2,p1)+BN+LReLU → [B,256,8,8]
        → ConvT(256→128,4×4,s2,p1)+BN+LReLU → [B,128,16,16]
        → ConvT(128→64,4×4,s2,p1)+BN+LReLU  → [B,64,32,32]
        → ConvT(64→3,4×4,s2,p1)+Sigmoid      → [B,3,64,64]
```

### Key choices
- LeakyReLU(0.2): no dead gradients
- BatchNorm: stabilizes VAE training
- Sigmoid output: pixels in [0,1] for BCE loss
- Latent dim 256: enough for face details, regularizable

## Loss: ELBO

```
Loss = Recon + β(epoch) · KL
```

- **Reconstruction:** BCE = -Σ[x·log(x̂) + (1-x)·log(1-x̂)], summed over pixels
- **KL:** -0.5·Σ[1 + log_var - mu² - exp(log_var)], summed over latent dims
- **Why BCE:** standard for [0,1] outputs; penalizes confident wrong predictions more than MSE

## Training Stability (2 techniques)

### 1. KL Warm-up
β(epoch) = min(1, epoch / 20). Prevents posterior collapse by suppressing KL while decoder learns.

### 2. Gradient Clipping
max_norm=5.0. Safety net against gradient spikes during warm-up transition.

### Why not others
- β-VAE: conflicts with warm-up (both control KL weight)
- Free bits: harder to tune threshold
- Better decoder likelihood: variance head can collapse/explode

## Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| batch_size | 128 | Fits in 6GB VRAM |
| epochs | 50 | Solid convergence |
| lr | 3e-4 | Standard for Adam-family |
| optimizer | AdamW | Decoupled weight decay; clean with LR scheduling |
| weight_decay | 1e-5 | Minimal; KL already regularizes encoder |
| scheduler | CosineAnnealingLR | Smooth decay, no extra hyperparams |
| kl_warmup_epochs | 20 | ~40% of training |
| grad_clip | 5.0 | Conservative |
| seed | 42 | Reproducibility |

## Evaluation

### Quantitative
- Reconstruction loss (BCE) on test set
- KL divergence on test set
- SSIM (structural similarity)
- FID if time permits

### Qualitative
1. Reconstruction grid: 8×2 (originals vs reconstructions)
2. Random samples: 8×8 from z ~ N(0,I)
3. Latent interpolation: 2 faces, 10 steps
4. Failure gallery: 10-20 worst reconstructions with hypotheses

### Training curves
1. Total loss per epoch
2. Reconstruction loss per epoch
3. KL divergence per epoch

## Project Structure
```
Test/
├── configs/default.yaml
├── src/
│   ├── __init__.py
│   ├── model.py        # Encoder, Decoder, VAE
│   ├── loss.py         # ELBO, KL warm-up
│   ├── dataset.py      # CelebA loading + preprocessing
│   └── utils.py        # Image grids, training curves
├── outputs/
│   ├── checkpoints/
│   ├── images/
│   └── curves/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Engineering Deliverables Checklist
- [ ] train.py runs end-to-end
- [ ] evaluate.py produces metrics + image grids
- [ ] configs/default.yaml with all hyperparams
- [ ] Saved checkpoints
- [ ] Generated images (grids, samples, interpolations, failure gallery)
- [ ] Training curves (total loss, recon, KL)
- [ ] README.md report (1-2 pages)

## Report Sections (README.md)
1. Dataset & preprocessing
2. Architecture summary + parameter counts
3. Loss design + why
4. Metrics + grids
5. Top issues encountered and fixes
6. Next steps (improve sharpness/diversity)
