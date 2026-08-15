# Architecture

Last updated: 2026-08-15

```mermaid
flowchart LR
    A[Official data\nNoisyLR + GT] --> B[audit.py\nPairing and 2× checks]
    B --> C[train.py]
    C --> D[PairedRestorationDataset\nCrop, flips, synthetic degradation]
    D --> E[Robust median/MAD\nnormalization]
    E --> F[JointRestorationNet]
    F --> G[Checkpoint\nbest.pt]

    G --> H[evaluate.py]
    I[Input directory] --> H
    H --> J[Robust normalization]
    J --> F
    F --> K[Restore intensity scale]
    K --> L[Output directory]

    L --> M[score.py\nPSNR · SSIM · LPIPS]
    L --> N[visualize_results.py\ncomparison panels]

    C --> O[experiments/kaggle/run_experiments.py\nP100/T4 candidate matrix]
    O --> P[Per-run checkpoints + predictions]
    P --> Q[scoreboard.json\nranked PSNR / SSIM]
    Q --> R[Measured T4 full winner\nwide_90e]

    subgraph Network[1-channel residual 2× network]
      F1[3×3 stem] --> F2[Residual blocks\nGroupNorm + depthwise conv]
      F2 --> F3[Fusion + skip]
      F3 --> F4[PixelShuffle ×2 head]
      F5[Bicubic ×2 base] --> F6[Residual addition]
      F4 --> F6
    end
    F --- F1
```

## Boundaries

- `restoration/io.py` owns supported image discovery and grayscale read/write behavior.
- `restoration/data.py` owns pairing, augmentation, cropping, and normalization.
- `restoration/model.py` owns only the neural model topology.
- Command-line scripts orchestrate input/output paths and create reproducible artifacts. The Kaggle runner composes those existing entry points rather than duplicating training logic.

Update this diagram whenever a pipeline component, model boundary, persisted artifact, or evaluation dependency changes.
