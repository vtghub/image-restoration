# Delivery plan

Last updated: 2026-08-12

## Objective

Deliver a reproducible, standalone image-restoration solution for the i4C/KLA semiconductor-inspection problem: grayscale denoising plus 2× super-resolution, assessed with PSNR, SSIM, LPIPS, robustness, and inference speed.

## Current status

| Phase | Status | Acceptance evidence |
| --- | --- | --- |
| 1. Data audit and baseline | Complete | 3,200 valid paired images; zero pairing and scale failures. |
| 2. Robust training and ablations | Complete | Robust configuration improved matched validation results. |
| 3. Held-out validation and test inference | Complete | 480-image held-out metrics and 400 official test outputs. |
| 4. Submission package | In progress | Code, lockfile, notes, and template deck exist; publication fields remain. |
| 5. Performance scaling | In progress | Kaggle P100/T4 runner and candidate matrix are ready; GPU results pending. |

## Next work

1. Publish the repository and a checkpoint-download location that inference can obtain automatically.
2. Replace bracketed team, contact, GitHub, and optional-demo fields in the official deck; export the final PDF.
3. Run clean-environment inference using the public repository instructions.
4. Attach the official data to Kaggle, run the `smoke` profile on P100/T4, then run `full` after the GPU and data gate pass.
5. Promote only a measured improvement and record it in project memory.

## Kaggle P100/T4 candidate matrix

| Candidate | Change | Keep only if |
| --- | --- | --- |
| Wider residual model | Increase width/blocks while retaining 1-channel residual output | Held-out PSNR/SSIM improve without material latency regression. |
| Stronger degradation mix | Tune blur, speckle, and Gaussian ranges | Source/OOD performance improves without degrading paired validation. |
| Loss ablation | Compare Charbonnier, gradient, and perceptual terms | Improves target metrics without hallucinated features. |
| Ensemble-free selection | Compare individual checkpoints | One checkpoint dominates the baseline on quality and speed. |

## Quality gates

- Pair audit reports zero missing, mismatched, or non-2× samples.
- Evaluation accepts an input directory and writes one output per input without source edits.
- All metrics are reported on an immutable, recorded validation split.
- A promoted model has recorded configuration, weights location, metrics, latency, and visual comparison.
- The Kaggle `smoke` profile detects CUDA and produces a complete `scoreboard.json` before the full profile is run.
- Submission metadata contains no bracketed placeholders.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Synthetic training degrades real inspection fidelity | Compare paired, source-held-out, and visual defect-preservation evidence. |
| Larger model is too slow | Benchmark warm CPU/GPU inference and retain a compact fallback. |
| Unpublished checkpoint breaks evaluator use | Publish the model location and test a fresh setup before submission. |
| Deck is submitted with incomplete identity/link data | Use the submission checklist and final PDF review. |
