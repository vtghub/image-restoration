# KLA PS01 submission handoff

## Completed technical evidence

- Paired-data audit: 3,200 complete 128x128 to 256x256 pairs; no pairing or 2x scale failures.
- Validation set: deterministic 15% split (480 images), recorded in `artifacts/final_cpu/validation_keys.txt`.
- Best CPU checkpoint: `artifacts/final_cpu/best.pt`.
- Held-out metrics: PSNR 27.20, SSIM 0.713, LPIPS 0.370.
- Official test outputs: `outputs/final_cpu_test/` (400 restored images).
- Visual evidence: `artifacts/final_cpu/comparison.png`.

## Reproduce inference

```powershell
python evaluate.py --input-dir <test_images_dir> --output-dir <restored_outputs_dir> --weights artifacts/final_cpu/best.pt
```

## Before publishing

1. Create a public GitHub repository and add the repository URL to the deck.
2. Upload `artifacts/final_cpu/best.pt` to Git LFS, Hugging Face, or Drive; document the automatic download location in `README.md` if it is not committed directly.
3. Add team name, members, college, and contact details to the official template's team-details slide.
4. Use `requirements.lock.txt` for the exact environment and retain `requirements.txt` as the concise install specification.
5. Run the inference command in a clean environment, then export the official deck as `TeamName_KLA_PS01.pdf`.

## Official-template deck draft

`TeamName_KLA_PS01_DRAFT.pptx` uses the official i4C template, contains nine submission slides, and is prefilled with the measured CPU-run evidence. Replace only the bracketed fields on the first and eighth slides:

- team name, members, academic year, college, leader contact, and email;
- public GitHub repository URL; and
- optional demo-video URL.

After those fields are complete, rename using the official `TeamName_PSNo` convention and export as PDF for portal upload.

## Known limitation

The final checkpoint was trained on CPU with a compact network. It is fully runnable but should be superseded by a larger CUDA/H100-trained candidate if compute becomes available.
