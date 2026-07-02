# Baseline Experiment

This folder contains the baseline stage of the project. It provides the first
comparison point before the longer 1M-step and tuned experiments.

## Structure

```text
baseline/
|-- notebooks/          # Baseline workflow notebooks
|-- results/            # Baseline CSV outputs
|-- figures/            # Baseline learning curves and report figures
|-- baseline_summary/   # Baseline summary data and plots
|-- baseline_summary.md
|-- environment.yml
|-- requirements.txt
`-- README.md
```

The notebooks, result tables, figures, and summary outputs are kept together so
the baseline stage can be read and rerun as a separate part of the project.
Trained model files are generated locally when the baseline training notebooks
are run.
