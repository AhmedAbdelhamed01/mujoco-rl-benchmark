# Project Notebooks

This folder contains the main notebook workflow for the MuJoCo RL benchmark.

Run order:

1. `00_project_setup_and_registry.ipynb`
2. `01_train_halfcheetah_full.ipynb`
3. `02_train_hopper_full.ipynb`
4. `03_train_walker2d_full.ipynb`
5. `04_evaluate_models_and_robustness.ipynb`
6. `05_aggregate_results_and_statistics.ipynb`
7. `06_report_figures_tables_diagnostics.ipynb`

The training notebooks contain the full training sequence for each environment.
The later notebooks evaluate policies, aggregate results, and generate the final
figures and tables.
