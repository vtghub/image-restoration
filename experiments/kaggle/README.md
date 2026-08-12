# Kaggle GPU iteration

This folder runs the same candidate matrix on a Kaggle P100 or T4 notebook. The runner trains each configuration, restores the recorded validation split, scores PSNR/SSIM, and writes one sortable scoreboard.

## One-time Kaggle setup

1. Create a private notebook, enable an accelerator (P100 or T4), and turn Internet on while cloning, installing dependencies, and downloading the public official archive.
2. Clone this repository, then use `kaggle_gpu_iteration.ipynb` as the executable notebook source. `bootstrap_data.py` downloads and extracts the archive beneath `/kaggle/working/data/train`.

The notebook writes all result files below `/kaggle/working/runs`. Save a version after each candidate matrix so checkpoints and the `scoreboard.json` are retained as notebook output.

## Candidate profiles

- `smoke` — 5 epochs per candidate; validates data layout, GPU availability, and run artifacts.
- `full` — 70–90 epochs per candidate; the intended P100/T4 comparison.

Run a single profile manually:

```bash
python experiments/kaggle/run_experiments.py \
  --data-root /kaggle/input/<dataset-slug>/train \
  --runs-root /kaggle/working/runs \
  --profile full
```

## Promotion rule

Promote a candidate only when its recorded held-out PSNR and SSIM improve on `final_cpu` (27.19862 / 0.7129819), visual evidence shows no invented defect structure, and GPU inference remains acceptable. Run optional LPIPS on the finalist before changing the selected checkpoint.
