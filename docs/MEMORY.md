# Project memory

Last updated: 2026-08-12

## Current state

- Problem: restore degraded grayscale semiconductor-inspection imagery with noise removal and 2× super-resolution.
- Data: 3,200 paired `NoisyLR` (128×128) → `GT` (256×256) training images; 400 test `NoisyLR` images.
- Model: compact 1-channel residual network, bicubic skip path, and PixelShuffle 2× output head.
- Normalization: per-image median/MAD, without clipping noisy input values.
- Final CPU checkpoint: `artifacts/final_cpu/best.pt` (not committed; data and artifacts are intentionally ignored).
- Held-out validation: 480 images; PSNR 27.19862, SSIM 0.7129819, LPIPS 0.3698113.
- Test inference: 400 outputs in `outputs/final_cpu_test/`; 18.16 ms/image mean CPU model inference.
- Git workflow: `feature/* → develop → main`; baseline currently promoted to `main`.

## Decisions

| Decision | Why | Consequence |
| --- | --- | --- |
| Use grayscale-only restoration | Task data is grayscale and channel expansion adds cost without task value. | Model input/output stay one channel. |
| Add bicubic base plus learned residual | Preserves a stable upsampled signal while learning degradation-specific corrections. | Output is `bicubic(low) + residual`. |
| Use robust un-clipped normalization | Speckle can push input values beyond clean-target range. | Statistics are restored after inference; values are not pre-clipped. |
| Avoid adversarial loss | Inspection workflows must avoid hallucinated defect features. | Fidelity losses and visual checks are preferred. |
| Keep generated data/weights out of Git | Dataset and model artifacts are large and reproduce from documented sources. | Public release must provide a model-download path. |

## Validation evidence

| Run | Split / input | PSNR | SSIM | LPIPS | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_cpu` | 480 held-out images | 26.5384 | 0.67859 | 0.38675 | 5-epoch compact baseline. |
| `final_cpu` | 480 held-out images | 27.19862 | 0.7129819 | 0.3698113 | Best robust CPU checkpoint; selected result. |
| `compact_smoke` | 480 held-out images | 27.20927 | 0.7120021 | — | T4, 5 epochs; 10.48 ms/image GPU model inference. PSNR is slightly above `final_cpu`; SSIM is slightly below, so it is not promoted. |
| `wide_smoke` | 480 held-out images | 27.50970 | 0.7262243 | — | T4, 5 epochs, width 64 / 14 blocks; 20.17 ms/image GPU model inference. Smoke winner, +0.31108 PSNR and +0.01324 SSIM over `final_cpu`. |

## Open items

1. Publish a checkpoint distribution location and verify a clean clone can run inference.
2. Fill the official deck's identity and publication fields, then export PDF.
3. Complete the active T4 full candidate profile, then record LPIPS and visual-finalist evidence before replacing the CPU checkpoint.
4. Kaggle execution is authenticated and uses a private Internet-enabled notebook. The public official archive bootstrap completed successfully.
5. The current Kaggle PyTorch build cannot execute on a P100 (`sm_60`): it raises `CUDA error: no kernel image is available for execution on the device`. The notebook has been moved to the supported T4 (`sm_75`) runtime.

## Change log

| Date | Change | Evidence / impact |
| --- | --- | --- |
| 2026-08-11 | Audited official paired training data. | 3,200 pairs; zero pairing and 2× scale failures. |
| 2026-08-11 | Completed robust CPU training and test inference. | Selected 480-image validation metrics and 400 restored test outputs. |
| 2026-08-11 | Created submission notes and official-template deck draft. | Deck contains measured evidence; placeholders remain only for team/link data. |
| 2026-08-11 | Promoted baseline through GitHub `feature → develop → main`. | Feature `c791439`, develop promotion `239425c`, main promotion `bb6cc20`. |
| 2026-08-12 | Established living planning, memory, and diagram documentation. | This document and the three Mermaid diagrams are now the incremental-update baseline. |
| 2026-08-12 | Added Kaggle P100/T4 experiment runner, candidate profiles, and notebook source. | GPU execution is prepared; Kaggle sign-in was subsequently verified. |
| 2026-08-12 | Added public-archive bootstrap to the Kaggle notebook flow. | The private notebook can download/extract official training data without a manual Kaggle dataset upload. |
| 2026-08-12 | Verified the live Kaggle data and accelerator gates. | Public archive download completed; audit reported 3,200 pairs, zero pairing/scale failures, 128×128 LR to 256×256 GT. P100 failed because the current PyTorch image excludes `sm_60`; T4 smoke training is active. |
| 2026-08-12 | Completed T4 smoke matrix and started the T4 full profile. | `wide_smoke` won at PSNR 27.50970 / SSIM 0.7262243 on 480 images, ahead of the CPU baseline by +0.31108 / +0.01324. The full profile is now running `compact_70e`, `wide_90e`, and `wide_robust_90e`. |
