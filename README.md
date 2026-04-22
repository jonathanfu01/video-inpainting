# Video Inpainting Pipeline (Unconditional Prior + Conditional Guidance)

This repository is for a project on video inpainting using:

1. an **unconditional generative prior** over realistic videos, and  
2. a **conditional inpainting model** guided by that prior.

The goal is to reconstruct masked regions while preserving spatial quality and temporal consistency.

---

## 1) Project Objectives

Implement and evaluate the following stages:

- **Task I:** Unconditional video prior (`p_theta(V)`) using a pretrained model.
- **Task II:** Conditional inpainting baseline (no prior guidance).
- **Task III:** Prior-guided conditional inpainting with a guidance strength `lambda`.
- **Task IV:** Analysis across `lambda` values (quality, realism, temporal consistency, diversity).

---

## 2) Suggested Repository Structure

Create the project in this structure:

```text
.
├── configs/
│   ├── data.yaml
│   ├── prior.yaml
│   ├── baseline.yaml
│   └── guided.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── masks/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── masks.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── unconditional_prior.py
│   │   ├── conditional_inpainter.py
│   │   ├── temporal_module.py
│   │   └── discriminator.py
│   ├── losses/
│   │   ├── reconstruction.py
│   │   ├── prior_guidance.py
│   │   └── temporal.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_guided.py
│   │   └── sweep_lambda.py
│   ├── eval/
│   │   ├── metrics_spatial.py
│   │   ├── metrics_temporal.py
│   │   ├── metrics_diversity.py
│   │   └── evaluate.py
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── visualize.py
├── experiments/
│   ├── baseline/
│   └── guided/
└── README.md
```

---

## 3) Environment Setup

### Option A: conda

```bash
conda create -n vid-inpaint python=3.10 -y
conda activate vid-inpaint
pip install torch torchvision torchaudio
pip install opencv-python imageio einops tqdm pyyaml matplotlib
pip install scikit-image lpips
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio opencv-python imageio einops tqdm pyyaml matplotlib scikit-image lpips
```

---

## 4) Data Preparation

### 4.1 Choose dataset

Use one video dataset (example choices):

- DAVIS
- YouTube-VOS subset
- UCF101 subset (converted to clips)

### 4.2 Preprocess videos

Standardize all clips to:

- same frame size (for example, `256x256`)
- same clip length (for example, `T=16` frames)
- normalized pixel range (`[-1, 1]` or `[0, 1]`, be consistent)

### 4.3 Build masks

Generate training masks per frame:

- random brush or blob masks
- moving object-like masks
- mixed mask ratios (for example, 10%, 20%, 40%)

Corruption rule:

`v_t_tilde = (1 - m_t) * v_t + m_t * c_t`

where `c_t` is fill value/noise in masked regions.

---

## 5) Stage I: Unconditional Prior Model

Use a pretrained model as the video prior (`p_theta(V)`), such as:

- pretrained video autoencoder/VAE
- pretrained GAN/video discriminator features
- pretrained latent video diffusion encoder (if available)

Minimum deliverables:

- document which pretrained model is used
- define what distribution it approximates
- verify it can score/represent realistic video structure in your pipeline

You are not required to train this model from scratch.

---

## 6) Stage II: Conditional Baseline (No Guidance)

Train a conditional inpainting model with masked video input:

- **Input:** corrupted clip + mask
- **Output:** reconstructed clip
- **Core loss:** masked L1/L2 reconstruction

Example:

`L_rec = (1/T) * sum_t || m_t * (v_hat_t - v_t) ||_1`

Optional:

- lightweight temporal module (ConvLSTM or temporal attention)

Baseline command (target script):

```bash
python -m src.train.train_baseline --config configs/baseline.yaml
```

Save:

- checkpoints
- tensorboard/logs
- validation preview videos

---

## 7) Stage III: Add Prior Guidance

Train guided model with:

`L_total = L_rec + lambda * L_prior`

Choose at least one guidance method:

1. **Latent prior guidance** (KL/L2 to unconditional latent space)
2. **Discriminator guidance** (frame-wise or clip-level realism term)
3. **Energy-based prior term**

Run sweep over guidance strengths:

`lambda in {0.0, 0.01, 0.1, 0.5, 1.0}`

Guided training command:

```bash
python -m src.train.train_guided --config configs/guided.yaml --lambda 0.1
```

Sweep command:

```bash
python -m src.train.sweep_lambda --config configs/guided.yaml --lambdas 0.0 0.01 0.1 0.5 1.0
```

---

## 8) Stage IV: Evaluation and Analysis

Compare baseline vs guided models across `lambda`.

### 8.1 Spatial quality metrics

- PSNR
- SSIM
- LPIPS

### 8.2 Temporal consistency metrics (pick at least one)

- warping error (optical-flow-based)
- frame-difference consistency
- perceptual temporal coherence score

### 8.3 Diversity (recommended)

- multi-sample diversity in masked regions (distance in feature/LPIPS space)

Evaluation command:

```bash
python -m src.eval.evaluate --config configs/guided.yaml --checkpoint path/to/ckpt.pt
```

---

## 9) Expected Experiments

Minimum experiment table:

- Baseline (`lambda=0`)
- Guided (`lambda=0.01`)
- Guided (`lambda=0.1`)
- Guided (`lambda=0.5`)
- Guided (`lambda=1.0`)

For each setting, report:

- PSNR / SSIM / LPIPS
- at least one temporal metric
- qualitative video examples (success + failure cases)

Key analysis questions:

- Does increasing `lambda` improve realism?
- At what point does over-smoothing appear?
- Does temporal stability improve with guidance?
- What is the best trade-off setting?

---

## 10) Training Checklist

- [ ] Data pipeline produces `(clean_clip, mask_clip, corrupted_clip)` correctly
- [ ] Baseline converges and reconstructs masked regions
- [ ] Prior-guided loss is implemented and tunable via `lambda`
- [ ] Lambda sweep completed
- [ ] Metrics script runs on validation/test
- [ ] Plots/tables generated for report
- [ ] Qualitative videos exported

---

## 11) Suggested Milestones

### Week 1

- finalize dataset and mask generation
- run baseline training sanity check

### Week 2

- implement prior guidance loss
- run first `lambda` sweep

### Week 3

- full evaluation and qualitative comparisons
- prepare report figures/tables

---

## 12) Reproducibility Notes

- fix random seeds in all scripts
- log exact config + commit hash for each run
- keep train/val/test splits fixed
- report hardware and runtime

---

## 13) Report Deliverables

Your final report should include:

- chosen unconditional prior and justification
- baseline architecture and losses
- guidance mechanism and objective
- quantitative results vs `lambda`
- qualitative frames/video strips
- discussion of failure cases and trade-offs

This README is intended as the implementation roadmap for completing the full project pipeline.
