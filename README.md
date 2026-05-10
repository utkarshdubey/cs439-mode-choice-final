# CS439 Final Project: Segmentation-Aware Travel Mode Choice Prediction

`netID: ud38`

This repository reproduces the final report experiments for CS439::S26 data science final project at Rutgers. The project combines unsupervised traveler segmentation with supervised choice prediction on the Statsmodels Travel Mode Choice dataset.

## What is included

- `src/run_project.py`: full preprocessing, segmentation, model training, evaluation, and figure generation pipeline.
- `data/modechoice_statsmodels.csv`: exported copy of the dataset after loading from `statsmodels` and adding engineered features.
- `results/`: generated metrics, K-Means diagnostics, segment summaries, and SHAP feature importances.
- `figures/`: generated figures used in the report.
- `report/`: LaTeX source, NeurIPS style file, and compiled PDF.
- `notebooks/reproduce_results.ipynb`: lightweight notebook wrapper for running the pipeline.

## Reproduce the results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_project.py
```

The script uses a fixed random seed (`439`) and a group-aware train/test split by traveler ID, so rows from the same traveler never appear in both train and test sets.