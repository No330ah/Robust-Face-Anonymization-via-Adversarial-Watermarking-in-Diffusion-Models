
# Robust Face Anonymization via Adversarial Watermarking in Diffusion Models

> **A unified framework for identity-oblivious face anonymization using VAE + DiT + ArcFace + Mamba + Watermark.**

---

## 🔧 Architecture Overview

### 1️⃣ Overall Training Framework

![Overview](pic\framework.png)

---

### 2️⃣ DiT-based Multi-modal Backbone

![MM-DiT](pic\Backbone.png)

---

### 3️⃣ Qualitative Comparison

![Qualitative](pic\Quantitative.png)

---


## 📊 Experimental Results

| Method | SSIM ↑ | PSNR ↑ | Identity Recovery ↑ |
|--------|--------|--------|----------------------|
| Ours   | 0.872  | 31.91  | 1.00                 |
| RIDDLE | 0.793  | 31.36  | 0.80                 |
| FIT    | 0.775  | 32.30  | 0.95                 |

---
## 📊 Full image quality and identity preservation
| Method       | SSIM ↑ | PSNR ↑ | LPIPS ↓ | CLIP ↑ | MSSIM ↑ | FID ↓  | Identity Recovery ↑ |
|--------------|--------|--------|---------|--------|----------|--------|----------------------|
| Ours         | 0.854  | 28.90  | 0.037   | 0.384  | 0.868    | 25.65  | 0.99                 |
| RIDDLE       | 0.762  | 19.49  | 0.192   | 0.365  | 0.792    | 26.85  | 0.80                 |
| CIAGAN       | 0.494  | 18.90  | 0.450   | 0.355  | 0.501    | 25.45  | 0.78                 |
| FIT          | 0.775  | 28.30  | 0.045   | 0.328  | 0.798    | 23.40  | 0.95                 |
| DeepPrivacy  | 0.745  | 19.93  | 0.110   | 0.378  | 0.784    | 22.37  | 0.88                 |


---

## 📦 Installation

```bash
conda env create -f env.yaml
conda activate env
```

---

## 📜 Citation

> Coming soon after paper submission.

---
