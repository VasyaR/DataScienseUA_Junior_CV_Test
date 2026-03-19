# Configuration & Reproducibility

## Requirement
> Reproducible config (JSON/YAML/etc.)

## Format: YAML
Single file `configs/default.yaml` with all hyperparameters. Loaded at runtime by both `train.py` and `evaluate.py`.

## Hyperparameters & Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| dataset | celeba | Best visual results for VAE (structured face domain) |
| image_size | 64 | Standard CelebA benchmark resolution |
| crop_size | 148 | Center crop to remove background before resize |
| channels | 3 | RGB |
| latent_dim | 256 | Enough for face details, regularizable by KL |
| batch_size | 128 | Fits in 6GB VRAM with ~2-3GB usage |
| epochs | 50 | Solid convergence; can extend to 100 if needed |
| lr | 3e-4 | Standard starting point for Adam-family optimizers |
| optimizer | adamw | Decoupled weight decay; modern default. Small weight decay (1e-5) for clean regularization without conflicting with KL |
| weight_decay | 1e-5 | Light regularization; KL already regularizes the encoder, so we keep this minimal |
| scheduler | cosine | CosineAnnealingLR — smooth decay from 3e-4 to near 0 over training. No extra hyperparams to tune |
| kl_warmup_epochs | 20 | ~40% of training; enough for decoder to learn before KL kicks in |
| grad_clip_max_norm | 5.0 | Conservative; only activates on spikes |
| seed | 42 | Reproducibility |

## Why AdamW over Adam
- Adam bakes weight decay into the gradient update (L2 reg), coupling it to the learning rate
- AdamW decouples weight decay — cleaner regularization, especially with LR scheduling
- KL already regularizes the encoder, so weight decay is kept minimal (1e-5) to avoid double-penalizing

## Why CosineAnnealingLR
- Fixed LR causes oscillation around the optimum late in training
- Cosine schedule decays smoothly — no step-size or gamma hyperparams to tune
- Natural fit for fixed epoch count (50 epochs)
- Widely used in modern training pipelines

## Reproducibility Guarantees
- Fixed random seed for Python, NumPy, PyTorch, CUDA
- Config file checked into git
- `torch.backends.cudnn.deterministic = True`
- All outputs saved with config snapshot
