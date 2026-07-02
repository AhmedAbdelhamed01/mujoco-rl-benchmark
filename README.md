# MuJoCo Deep RL Benchmark

This project studies continuous-control reinforcement learning on three MuJoCo
locomotion environments: `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5`.

The benchmark compares five algorithms:

- `PPO`
- `SAC`
- `TD3`
- `DDPG`
- `TQC`

The workflow starts with environment and registry checks, then runs the main
training experiments, evaluates the trained policies, tests robustness to action
noise, aggregates the numerical results, and generates the final figures and
tables.

## Project Structure

```text
project_root/
|-- notebooks/              # Main experiment notebooks in run order
|-- baseline/               # Baseline-stage notebooks, outputs, and figures
|-- results/                # CSV and JSON outputs used by the analysis
|-- figures/                # Learning curves, diagnostics, and report figures
|-- report/                 # Project report PDF
|-- project_utils.py        # Shared training, evaluation, statistics, and plotting code
|-- environment.yml         # Conda environment
|-- requirements.txt        # CPU/general Python dependencies
|-- requirements-gpu.txt    # GPU-oriented dependency file
|-- .gitignore              # Generated files that should stay local
`-- README.md
```

The included notebooks already contain their executed outputs. The CSV files in
`results/` and the images in `figures/` are the recorded experiment outputs used
by the analysis and report.

Trained model files, checkpoints, replay buffers, TensorBoard logs, and videos
are generated locally by the training notebooks. They are kept outside the
tracked project files because they are large and can be regenerated from the
notebook workflow.

## Experiment Sequence

The main workflow is in `notebooks/`:

1. `00_project_setup_and_registry.ipynb`
   checks package versions, environment availability, result folders, and the
   experiment registry.

2. `01_train_halfcheetah_full.ipynb`
   trains and summarizes the `HalfCheetah-v5` runs.

3. `02_train_hopper_full.ipynb`
   trains and summarizes the `Hopper-v5` runs.

4. `03_train_walker2d_full.ipynb`
   trains and summarizes the `Walker2d-v5` runs.

5. `04_evaluate_models_and_robustness.ipynb`
   evaluates trained policies and measures robustness under action noise.

6. `05_aggregate_results_and_statistics.ipynb`
   builds the final performance tables, confidence intervals, rankings, sample
   efficiency summaries, and pairwise comparisons.

7. `06_report_figures_tables_diagnostics.ipynb`
   generates the report figures and displays the final visual outputs.

The three training notebooks follow the same structure for each environment:

1. define the environment and candidate settings,
2. build the default, screening, and tuned run plans,
3. check existing completed runs and checkpoints,
4. train missing runs or resume interrupted runs,
5. summarize registry rows and learning curves.

## Baseline Stage

The `baseline/` folder keeps the first experimental stage of the project. It
contains its own notebooks, result tables, figures, and summary files. These
outputs provide the early comparison point before the longer 1M-step and tuned
experiments.

## Results

Important result locations:

```text
results/raw/        # Per-run training evaluation logs and metadata
results/processed/  # Combined final evaluation and robustness files
results/final/      # Tables used for final analysis
figures/report_ready/
figures/diagnostics/
```

The main final evaluation file is:

```text
results/processed/final_eval_all.csv
```

The robustness evaluation file is:

```text
results/processed/robustness_eval_all.csv
```

## Environment Setup

Using conda:

```bash
conda env create -f environment.yml
conda activate RL_PROJECT
```

Using pip:

```bash
pip install -r requirements.txt
```

For CUDA training, install the GPU dependency set only if it matches the CUDA
version available on the machine:

```bash
pip install -r requirements-gpu.txt
```

## Re-running

To reproduce the workflow, open the notebooks in the order listed above. The
training notebooks are complete: if models or checkpoints exist locally, they
can skip or resume completed work; if they do not exist, the same notebooks can
train the runs again from the start.

Long training runs can take many hours, especially when all environments,
algorithms, settings, and seeds are executed.
