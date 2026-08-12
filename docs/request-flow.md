# Request flow

Last updated: 2026-08-12

```mermaid
sequenceDiagram
    autonumber
    participant User as Evaluator / operator
    participant CLI as evaluate.py
    participant IO as Image I/O
    participant Norm as Robust normalization
    participant Model as JointRestorationNet
    participant Disk as Output directory

    User->>CLI: --input-dir --output-dir --weights
    CLI->>CLI: Load checkpoint and select CPU/CUDA
    CLI->>IO: Discover supported input images
    alt No supported images
        IO-->>CLI: Empty list
        CLI-->>User: FileNotFoundError
    else Images available
        loop Each input image
            CLI->>IO: Read grayscale image
            IO-->>CLI: 1×H×W tensor
            CLI->>Norm: Median/MAD normalize
            Norm-->>CLI: Normalized tensor + center + scale
            CLI->>Model: Predict 2× restored normalized image
            Model-->>CLI: Restored normalized tensor
            CLI->>Norm: Restore center and scale
            CLI->>IO: Write output preserving relative path
            IO->>Disk: Persist restored image
        end
        CLI-->>User: Image count and mean model latency
    end
```

## Failure behavior

| Condition | Current behavior | Follow-up action |
| --- | --- | --- |
| Missing model weights | Checkpoint load fails before processing. | Verify published model location and checksum. |
| Invalid input directory / unsupported files | Raises `FileNotFoundError`. | Provide supported grayscale image files. |
| Requested keys absent from input | Raises `FileNotFoundError` after filtering. | Reconcile the manifest with the input directory. |
| GPU unavailable | Defaults to CPU. | Use `--device cuda` only when a compatible CUDA runtime is available. |

Update this diagram and table whenever CLI arguments, validation, batching, hardware selection, output rules, or error behavior change.
