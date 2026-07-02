# Results Folder

This folder stores the generated experiment data used by the notebooks.

```text
results/
|-- raw/             # Per-run training/evaluation logs and metadata
|-- processed/       # Combined evaluation and robustness CSV files
|-- final/           # Final summary tables used in analysis and reporting
`-- run_tracker.csv  # Small run-tracking placeholder/file
```

The notebooks read these folders directly through `project_utils.py`, so the
folder names should stay unchanged.

