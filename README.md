# Defect-Faithful Semiconductor Image Restoration

An efficient PyTorch baseline for the KLA **AI-Based Restoration of Degraded Images for Semiconductor Inspection** challenge. It jointly denoises and performs 2x super-resolution of grayscale inspection imagery.

## Data layout

Download the supplied dataset and arrange it as follows. Files are paired by the same relative path and filename stem; TIFF, PNG, BMP, JPEG, and NumPy arrays are supported.

```text
data/
  train/
    degraded/
      sample_001.png
    ground_truth/
      sample_001.png
```

The official KLA archive layout is also accepted without renaming:

```text
train/
  NoisyLR/
  GT/
```

For an official held-out set, use the same `degraded/` and `ground_truth/` layout under `data/val/`. The training command otherwise creates a reproducible validation split.

## Phase 1 — audit and baseline

```powershell
python audit.py --data-root data/train --report artifacts/audit_train.json
python train.py --data-root data/train --epochs 30 --batch-size 8 --output-dir artifacts/baseline
python evaluate.py --input-dir data/val/degraded --output-dir outputs/val --weights artifacts/baseline/best.pt
```

`audit.py` detects missing pairs, reports image sizes, intensity ranges, and 2x scale compliance. Do not train until it reports zero missing or mismatched pairs.

## Submission inference

The evaluator-facing command has no manual source edits:

```powershell
python evaluate.py --input-dir <test_images_dir> --output-dir <restored_outputs_dir> --weights <model.pt>
```

It processes each image, restores it at 2x resolution, and preserves the input filename. `--device cuda` is used automatically when available.

Score a paired validation set with:

```powershell
python score.py --prediction-dir outputs/val --ground-truth-dir data/val/ground_truth
```

Add `--lpips` when the optional learned-perceptual package and its pretrained weights are available. LPIPS is computed by replicating the grayscale channel only for the metric; the restoration model itself remains grayscale.

Create slide-ready visual evidence from a held-out prediction directory:

```powershell
python visualize_results.py --input-dir data/train/NoisyLR --prediction-dir outputs/val --ground-truth-dir data/train/GT --keys-file artifacts/run/validation_keys.txt --output artifacts/comparison.png
```

## Project phases

1. Audit the provided paired data and establish the bicubic-plus-network baseline.
2. Train with robustness augmentations and compare loss/model ablations.
3. Validate on random and source-held-out splits using PSNR, SSIM, and LPIPS.
4. Export final weights, benchmark inference, and package the public repository and PDF evidence.

## Submission handoff

See `submission/SUBMISSION_NOTES.md` for the measured checkpoint, exact environment lockfile, required publish steps, and official-deck handoff. A prefilled official-template deck is available at `submission/TeamName_KLA_PS01_DRAFT.pptx`; replace its bracketed team and publication fields before exporting the required PDF.
