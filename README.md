# 🧠 Denoising Diffusion Probabilistic Model (DDPM) on MNIST

> A from-scratch implementation of [Ho et al., 2020](https://arxiv.org/abs/2006.11239) in a Google Colab notebook — covering the full diffusion pipeline, U-Net training, noise schedule ablation, and DDIM sampling.

<p align="center">
  <a href="https://colab.research.google.com/github/TN108/DDPM/blob/main/DDPM.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Dataset-MNIST-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Open in Colab](#open-in-colab)
- [Notebook Structure](#notebook-structure)
- [Results](#results)
- [How It Works](#how-it-works)
- [Ablation Study](#ablation-study)
- [Extension: DDIM Sampling](#extension-ddim-sampling)
- [Evaluation](#evaluation)
- [References](#references)

---

## Overview

Diffusion models learn to generate images by reversing a gradual noising process. This notebook implements everything from scratch:

- ✅ **Forward diffusion** — closed-form noisy image generation at any timestep
- ✅ **Reverse diffusion** — U-Net noise predictor with sinusoidal timestep embeddings
- ✅ **Training pipeline** — AdamW optimizer, MSE loss on predicted noise
- ✅ **Noise schedule ablation** — linear vs cosine schedule comparison
- ✅ **DDIM sampler** — deterministic, fast generation extension

---

## Open in Colab

Click the badge below to run the notebook directly in Google Colab (free GPU recommended):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TN108/DDPM/blob/main/DDPM.ipynb)

> **Tip:** Go to `Runtime → Change runtime type → T4 GPU` for faster training.

---

## Notebook Structure

The notebook is organized into self-contained sections — run them top to bottom:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Setup & Imports** | Install dependencies, import libraries |
| 2 | **Dataset** | Load and preprocess MNIST, visualize samples |
| 3 | **Noise Schedules** | Define linear & cosine `β` schedules, plot `ᾱ_t` and SNR |
| 4 | **Forward Diffusion** | Add noise, visualize noising progression |
| 5 | **U-Net Architecture** | Residual blocks, timestep embedding, encoder-decoder |
| 6 | **Training Loop** | AdamW, MSE loss, loss curve, gradient norm monitoring |
| 7 | **Sampling (DDPM)** | Reverse diffusion, generate 64-image grid |
| 8 | **Ablation Study** | Linear vs cosine — loss & gradient comparison |
| 9 | **DDIM Sampling** | Deterministic sampler, fast inference |
| 10 | **Evaluation** | Noise MSE, sample diversity metrics |

---

## Results

### Noising Progression

The digit "3" at timesteps `t = 0, 50, 200, 500, 800, 999`:

![Noising progression](assets/noising_progression.png)

### Denoising Step at t = 500

`x0 | x_t | x0_hat | x_{t-1} (teacher) | x_{t-1} (from ε)`

![Denoising step](assets/denoise_step.png)

### Training Curves

| Training Loss (MSE) | Gradient Norm |
|:---:|:---:|
| ![Loss](assets/train_loss.png) | ![Grad norm](assets/grad_norm.png) |

---

## How It Works

### Forward Diffusion

Noise is added to a clean image `x₀` over `T = 1000` timesteps using a closed-form equation:

```
x_t = sqrt(ᾱ_t) · x₀  +  sqrt(1 - ᾱ_t) · ε,    ε ~ N(0, I)
```

`ᾱ_t` is the cumulative product of `(1 − β_t)`. As `t → T`, the image becomes pure Gaussian noise.

### Reverse Diffusion

A U-Net `ε_θ` is trained to predict the noise given the noisy image and timestep:

```
ε_hat = ε_θ(x_t, t)
```

**Training loss:**
```
L = || ε − ε_hat ||²
```

At inference, starting from `x_T ~ N(0, I)`, the model denoises step-by-step:
```
x_T → x_{T-1} → ... → x_0
```

### U-Net Architecture

| Component | Details |
|-----------|---------|
| Input | Noisy image `x_t` + timestep `t` |
| Timestep embedding | Sinusoidal → MLP → injected into each residual block |
| Encoder | Conv blocks + downsampling, increasing channels |
| Decoder | Conv blocks + upsampling + skip connections |
| Activation | SiLU |
| Normalization | Group Normalization |

---

## Ablation Study

### Linear vs Cosine Noise Schedule

| Property | Linear | Cosine |
|----------|--------|--------|
| `ᾱ_t` shape | Uniform S-curve | Symmetric S-curve |
| Early timesteps | Faster signal loss | Slower signal loss |
| Training loss | ✅ Slightly lower & stable | Slightly higher |
| Gradient norms | ✅ More stable | Larger fluctuations |
| Sample quality | Visually similar | Visually similar |

| Linear vs Cosine — Loss | Linear vs Cosine — Grad Norm |
|:---:|:---:|
| ![Loss comparison](assets/loss_comparison.png) | ![Grad comparison](assets/grad_comparison.png) |

**Conclusion:** Both schedules work well on MNIST. The cosine schedule's advantage (preserving signal longer) becomes more significant on complex datasets like CIFAR-10.

---

## Extension: DDIM Sampling

A **DDIM (Denoising Diffusion Implicit Models)** deterministic sampler was implemented alongside the standard DDPM sampler.

| | DDPM | DDIM |
|--|------|------|
| Stochastic | ✅ | ❌ |
| Steps needed | ~1000 | ~50 |
| Retraining needed | — | ❌ uses same checkpoint |
| Latent interpolation | ❌ | ✅ |
| Speed | Slower | ~20× faster |

```python
# Switch sampler in the notebook — no retraining needed
samples = ddim_sample(model, n_steps=50)
```

---

## Evaluation

| Metric | Description |
|--------|-------------|
| **Noise Prediction MSE** | How accurately the network predicts noise at each timestep |
| **Sample Diversity** | Pixel variance across generated samples — checks for mode collapse |

FID was not used since MNIST is small and grayscale — standard perceptual metrics are less meaningful at this scale.

---

## References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Song et al. (2020) — [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Nichol & Dhariwal (2021) — [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

---

<p align="center">
  Made with ❤️ &nbsp;·&nbsp; <a href="https://github.com/TN108/DDPM">github.com/TN108/DDPM</a>
</p>
