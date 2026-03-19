# Loss Design: ELBO

## Requirement
> You must write down and implement the **ELBO loss**.

## The Math

The **Evidence Lower Bound (ELBO)** is what we maximize:

```
ELBO = E_q(z|x)[ log p(x|z) ] - KL( q(z|x) || p(z) )
         ↑ reconstruction          ↑ regularization
```

Since we minimize loss, we negate it:

```
Loss = -ELBO = -E_q(z|x)[ log p(x|z) ] + KL( q(z|x) || p(z) )
     = Reconstruction Loss + KL Loss
```

## Reconstruction Term: Bernoulli Likelihood (BCE)

```
Recon = -Σ[ x·log(x̂) + (1-x)·log(1-x̂) ]
```

Summed over all pixels (C × H × W).

**Why BCE over MSE:**
- Images normalized to [0,1], decoder outputs Sigmoid → [0,1].
- BCE treats each pixel as independent Bernoulli. Not perfect (pixels are continuous), but standard.
- MSE (= Gaussian likelihood with fixed variance) tends to produce even blurrier outputs because it penalizes all errors equally, while BCE penalizes confident wrong predictions more heavily.

## KL Term: Closed-Form Gaussian

With q(z|x) = N(mu, diag(σ²)) and p(z) = N(0, I):

```
KL = -0.5 * Σ[ 1 + log_var - mu² - exp(log_var) ]
```

Summed over all latent dimensions (256).

**Intuition:** Penalizes the encoder for producing distributions far from the standard normal. This regularizes the latent space so random samples from N(0,I) produce plausible images.

## Training Loss with KL Warm-up

```
Loss = Recon + β(epoch) · KL
```

Where β(epoch) = min(1, epoch / warmup_epochs), with warmup_epochs = 20.

## What We Log
Three separate curves for the report:
1. **Total loss** — should decrease overall
2. **Reconstruction loss** — should decrease as decoder improves
3. **KL loss** — should increase during warm-up (encoder learns structure), then stabilize
