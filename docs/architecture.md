# Architecture

Last updated: 2026-08-12

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
- Command-line scripts orchestrate input/output paths and create reproducible artifacts.

Update this diagram whenever a pipeline component, model boundary, persisted artifact, or evaluation dependency changes.
