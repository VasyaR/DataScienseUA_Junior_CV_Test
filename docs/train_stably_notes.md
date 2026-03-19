# Training Stability: Technique Choices

## Requirement
> Add **at least two** of the following (and explain why you chose them):
> KL warm-up / annealing, β-VAE (β ≠ 1), Gradient clipping, Free bits, Better decoder likelihood

## Chosen: KL Warm-up + Gradient Clipping

### 1. KL Warm-up / Annealing Schedule
**What:** Linearly ramp the KL weight β from 0 → 1 over the first 20 epochs.

**Why we chose it:**
- The #1 VAE failure mode is **posterior collapse**: the encoder outputs q(z|x) ≈ N(0,I) regardless of input, making KL ≈ 0 but producing meaningless latents.
- This happens because early in training the decoder is weak. The easiest way to reduce total loss is to minimize KL (push q toward the prior) rather than improve reconstruction.
- Warm-up suppresses KL early, forcing the encoder to encode useful information while the decoder catches up.
- Gives us a clear **debugging narrative**: we can show training curves with and without warm-up, demonstrating posterior collapse and its fix.

### 2. Gradient Clipping (max_norm=5.0)
**What:** Clip the global gradient norm before each optimizer step.

**Why we chose it:**
- During the warm-up transition (β increasing), reconstruction and KL gradients can spike as they compete.
- Clipping is a safety net: does nothing when gradients are healthy, prevents catastrophic updates when they spike.
- Simple, robust, no hyperparameter sensitivity (5.0 is a safe default).

## Why NOT the others

| Option | Reason for skipping |
|--------|-------------------|
| **β-VAE (β ≠ 1)** | Conflicts with KL warm-up — both control the KL weight. Using both adds confusing interactions. Pick one or the other. |
| **Free bits** | Sets a minimum KL per latent dimension (λ). Effective but harder to tune (what threshold?) and harder to explain concisely in a 1–2 page report. |
| **Better decoder likelihood** | Learned Gaussian variance per pixel. Most principled theoretically, but the variance head can collapse (log_var → -∞) or explode — adds debugging risk under time pressure. |

## Summary
KL warm-up is **high-impact** (directly fixes the main failure mode) and gradient clipping is **low-risk** (pure safety net). Together they address training stability without adding complexity.
