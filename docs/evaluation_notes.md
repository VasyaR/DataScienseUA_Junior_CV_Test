# Evaluation Plan

## Requirement
> Provide qualitative and quantitative evaluations.
> Include a "failure gallery", showing 10–20 bad samples/reconstructions and hypothesize why.

## Quantitative Metrics

| Metric | What it measures | How |
|--------|-----------------|-----|
| Reconstruction loss (BCE) | How well the model reconstructs test images | Average BCE over test set |
| KL divergence | How close the encoder posterior is to the prior | Average KL over test set |
| SSIM | Perceptual similarity between input and reconstruction | `torchmetrics` or manual implementation |
| FID (if time permits) | Quality/diversity of generated samples vs real data | Inception network feature comparison |

## Qualitative Outputs (Image Grids)

1. **Reconstruction grid** — 8×2: top row originals, bottom row reconstructions. Shows reconstruction quality.
2. **Random samples** — 8×8 grid from z ~ N(0,I). Shows generative quality.
3. **Latent interpolation** — two faces, linear interpolation in z-space, ~10 steps. Shows latent space smoothness.
4. **Failure gallery** — 10-20 worst reconstructions by loss.

## Failure Gallery Hypotheses
Common reasons VAE reconstructions fail on CelebA:
- **Rare accessories:** unusual hats, sunglasses, jewelry — underrepresented in training data
- **Unusual angles:** extreme head poses the model hasn't seen often
- **Occlusions:** hands covering face, overlapping objects
- **Fine details:** text on clothing, complex earrings — high-frequency info lost by VAE bottleneck
- **Lighting extremes:** very dark or overexposed images

## Training Curves (3 plots)
1. Total loss per epoch (train + val)
2. Reconstruction loss per epoch
3. KL divergence per epoch

These show the warm-up dynamics: KL should be suppressed early, rise during warm-up, then stabilize.

## evaluate.py Outputs
- All metrics printed to console and saved to `outputs/metrics.json`
- All image grids saved to `outputs/images/`
- Training curves saved to `outputs/curves/`
