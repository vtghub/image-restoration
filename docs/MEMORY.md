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

## Open items

1. Publish a checkpoint distribution location and verify a clean clone can run inference.
2. Fill the official deck's identity and publication fields, then export PDF.
3. Run CUDA/H100 candidate experiments and record results before replacing the CPU checkpoint.

## Change log

| Date | Change | Evidence / impact |
| --- | --- | --- |
| 2026-08-11 | Audited official paired training data. | 3,200 pairs; zero pairing and 2× scale failures. |
| 2026-08-11 | Completed robust CPU training and test inference. | Selected 480-image validation metrics and 400 restored test outputs. |
| 2026-08-11 | Created submission notes and official-template deck draft. | Deck contains measured evidence; placeholders remain only for team/link data. |
| 2026-08-11 | Promoted baseline through GitHub `feature → develop → main`. | Feature `c791439`, develop promotion `239425c`, main promotion `bb6cc20`. |
| 2026-08-12 | Established living planning, memory, and diagram documentation. | This document and the three Mermaid diagrams are now the incremental-update baseline. |
