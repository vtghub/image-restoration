# Feature map

Last updated: 2026-08-12

```mermaid
mindmap
  root((Semiconductor Image Restoration))
    Data reliability
      Official NoisyLR/GT layout
      Generic degraded/ground_truth layout
      Pair discovery
      2× scale audit
      Intensity-range report
    Restoration model
      Grayscale input/output
      Robust median/MAD normalization
      Bicubic base
      Residual blocks
      PixelShuffle 2× head
    Training
      Paired crop alignment
      Flip augmentation
      Blur/downsample synthesis
      Speckle and Gaussian noise
      Charbonnier plus gradient loss
      Deterministic validation split
    Evaluation
      Directory-to-directory inference
      Automatic CPU/CUDA selection
      PSNR
      SSIM
      Optional LPIPS
      Comparison panels
    Delivery
      Requirements lockfile
      Submission notes
      Official-template deck draft
      Feature/develop/main workflow
    Planned scaling
      CUDA/H100 candidates
      OOD/source-held-out checks
      Published weights
      Final PDF and demo
```

## Ownership map

| Capability | Primary files |
| --- | --- |
| Data integrity | `audit.py`, `restoration/data.py`, `restoration/io.py` |
| Model topology | `restoration/model.py` |
| Optimization | `train.py` |
| Evaluator-facing restoration | `evaluate.py` |
| Metrics and evidence | `score.py`, `visualize_results.py` |
| Submission readiness | `README.md`, `submission/`, `docs/` |

Update this document whenever a user-visible capability is added, retired, or materially changes scope.
