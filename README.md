# RL-Guided 3D Diffusion for Point Cloud Completion

This repository implements a complete, reproducible project on **3D point cloud completion with diffusion models**, where we further fine-tune a lightweight **RL scheduler** to accelerate sampling.

We train and evaluate on the official **ModelNet40** benchmark (12,311 CAD models, 9,843 train / 2,468 test, 40 classes). The pipeline is:

1. Train a **3D diffusion model** to complete partial point clouds.
2. Train an **RL scheduler** that learns a dynamic timestep schedule (how many diffusion steps to skip at each state).
3. At inference, use the RL scheduler to reduce the number of denoising steps per sample while preserving or slightly improving reconstruction quality.

On the official ModelNet40 test split, the learned scheduler reduces the average number of denoising steps from **1000 → ~250** per sample, with a small improvement in Chamfer distance.

---

## 1. Project structure

```text
RL_3D_Completion_Github/
├── data/
│   └── processed/
│       └── modelnet40_completion_official/
│           ├── train.npz     # official train split (80%) – used for diffusion + RL training
│           ├── val.npz       # 20% of official train – used for model selection
│           └── test.npz      # official test split (2,468 shapes)
├── runs/
│   ├── diffusion_baseline_500/   # diffusion training logs & checkpoints
│   └── rl_scheduler_500_10000/   # RL scheduler training logs & checkpoints
├── src/
│   ├── data/
│   │   ├── download_modelnet40_hf_datasets.py
│   │   └── preprocess_modelnet40_from_hf.py
│   ├── models/
│   │   ├── diffusion_completion.py
│   │   └── rl_scheduler.py
│   ├── utils/
│   │   ├── chamfer.py
│   │   ├── device.py
│   │   └── seed.py
│   ├── train_diffusion.py
│   ├── train_rl_scheduler.py
│   └── eval_rl_scheduler.py
├── train_500_10000.sh
├── requirements.txt
└── README.md
```

---

## 2. Environment setup

Tested with **Python 3.10** and a single NVIDIA GPU (16 GB, e.g. RTX 4070 Ti Super).

```bash
# Create and activate conda environment
conda create -n rl3d python=3.10 -y
conda activate rl3d

# Install Python dependencies
pip install -r requirements.txt
```

---

## 3. Dataset: ModelNet40 (HF + official split)

We use the compressed **ModelNet40** point cloud dataset from Hugging Face (`Pointcept/modelnet40_normal_resampled-compressed`), which contains pre-sampled point clouds together with the official train/test lists and class names.

### 3.1 Download raw HF dataset

From repo root:

```bash
cd src

python src/data/download_modelnet40_hf_datasets.py
```

This script downloads the ModelNet40 dataset via `datasets.load_dataset` and caches it locally.

### 3.2 Preprocess into completion npz files

We normalize each shape to the unit sphere and generate synthetic **partial point clouds** by occluding part of the full point cloud.  
We follow the **official ModelNet40 split** (9,843 train / 2,468 test) and further split the train set into **train / val = 80% / 20%**.

From repo root:

```bash
python src/data/preprocess_modelnet40_from_hf.py   --out_dir data/processed/modelnet40_completion_official   --n_full 2048   --n_partial 1024   --val_ratio 0.2
```

This creates:

- `train.npz`  – full/partial pairs for diffusion + RL training  
- `val.npz`    – held-out validation for hyperparameter tuning & model selection  
- `test.npz`   – official ModelNet40 test set for final reporting

Each `.npz` file contains:

- `full_points`    – `(N, 2048, 3)` normalized full point clouds  
- `partial_points` – `(N, 1024, 3)` occluded partial point clouds  
- `labels`         – `(N,)` class indices (0–39)  
- `class_names`    – `(40,)` list of class names

---

## 4. Training

All training commands are wrapped in `train_500_10000.sh`.

```bash
sh train_500_10000.sh
```

The script runs two stages:

### 4.1 Diffusion training

```bash
python train_diffusion.py   --train_npz data/processed/modelnet40_completion_official/train.npz   --val_npz   data/processed/modelnet40_completion_official/val.npz   --workdir   runs/diffusion_baseline_500   --epochs    500   --batch_size 512   --lr        2e-4   --num_steps 1000
```

- Learns a conditional diffusion model that maps a **partial point cloud** + noise to a completed point cloud.
- Uses Chamfer distance between predicted full cloud and ground truth as the main reconstruction loss.
- `num_steps=1000` defines the diffusion time horizon (T=1000).

The best diffusion checkpoint is saved to:

```text
runs/diffusion_baseline_500/checkpoints/best.pt
```

### 4.2 RL scheduler training

```bash
python train_rl_scheduler.py   --ckpt      runs/diffusion_baseline_500/checkpoints/best.pt   --train_npz data/processed/modelnet40_completion_official/train.npz   --val_npz   data/processed/modelnet40_completion_official/val.npz   --workdir   runs/rl_scheduler_500_10000   --episodes  10000   --lambda_steps 0.002   --max_steps 500   --num_steps 1000   --actions   1 2 4   --lr        1e-4   --log_interval 50   --eval_interval 200
```

- The diffusion model is **frozen**; only the **RLScheduler** policy is trained.
- The policy observes a compact state derived from the current noisy point cloud and the timestep, and chooses a **skip size** (1, 2, or 4 diffusion steps).  
- Reward is defined as:

  \[
  r = -\text{ChamferDistance}(\hat{X}, X_{\text{gt}}) - \lambda_{\text{steps}} \cdot \text{num_steps}
  \]

  which encourages both **better completion** and **fewer denoising steps**.

- `max_steps=500` caps the number of denoising steps per sample for the RL policy (vs. the baseline 1000-step schedule).

The best policy checkpoint is saved to:

```text
runs/rl_scheduler_500_10000/checkpoints/best_policy.pt
```

---

## 5. Evaluation

After training, you can evaluate on the official ModelNet40 test set using:

```bash
python src/eval_rl_scheduler.py   --ckpt        runs/diffusion_baseline_500/checkpoints/best.pt   --policy_ckpt runs/rl_scheduler_500_10000/checkpoints/best_policy.pt   --test_npz    data/processed/modelnet40_completion_official/test.npz   --num_steps   1000   --actions     1 2 4   --lambda_steps 0.002   --max_steps   500
```

`eval_rl_scheduler.py` reports metrics for:

- **Baseline diffusion** – fixed schedule, 1 step per denoising iteration (T = 1000)  
- **RL scheduler** – learned skip schedule, up to `max_steps` per sample

On the official ModelNet40 test split, we obtain:

```text
[INFO] Evaluating baseline diffusion...
[RESULT][Baseline] CD: 0.016851 ± 0.009623, Steps: 1000.00, Time: 9:03

[INFO] Evaluating RL scheduler...
[RESULT][RL] CD: 0.016484 ± 0.009436, Steps: 250.01, Reward: -0.516508, Time: 4:52

[RESULT][Delta RL - Baseline] ΔCD=-0.000368 (negative = RL better), ΔSteps=-749.99
```

**Key takeaways:**

- The RL scheduler reduces the **average number of denoising steps by ~75%** (1000 → ~250 steps per sample), the inferencing time for the testing dataset is roughly **2x faster** (9:03 to 4:52).
- Despite using far fewer steps, the RL-guided schedule achieves a slightly **better Chamfer distance** (0.01685 → 0.01648), indicating that **speed is gained without sacrificing reconstruction quality**.

---

## 6. Model components

### 6.1 DiffusionCompletionModel (`src/models/diffusion_completion.py`)

- Point-cloud diffusion model with:
  - **Encoder**: encodes partial point clouds into a conditioning representation.
  - **Denoiser**: U-Net-like network on point features, conditioned on time embeddings and the encoder output.
- Uses a standard noise schedule (`alphas_cumprod`) and DDPM/DDIM-style updates during sampling.

### 6.2 RLScheduler (`src/models/rl_scheduler.py`)

- Small MLP policy that takes a **64-dim state**:
  - normalized timestep  
  - global mean and variance of the current noisy point cloud  
  - zero-padded to a fixed dimension  
- Outputs a categorical distribution over skip actions (e.g., {1, 2, 4}).  
- Trained with REINFORCE and a moving baseline; reward combines reconstruction quality and step efficiency.

---

## 7. Reproducibility notes

- Random seeds are controlled via `src/utils/seed.py`.
- All splits follow the **official ModelNet40** train/test lists; we only split the official train set into train/val (80%/20%).  
- Evaluation on `test.npz` is performed only once for final reporting.

---

## 8. Possible extensions

- Plug in stronger point-cloud backbones (PointNet++, DGCNN, PointMLP, etc.) for the denoiser.
- Explore richer RL state representations (e.g., class-aware scheduling, attention-based global features).
- Apply this RL-guided scheduling strategy to other 3D generative tasks (shape generation, volumetric completion in medical imaging, etc.).
