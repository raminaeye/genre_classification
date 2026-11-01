# Music Genre Classification

An end-to-end ML pipeline that classifies music tracks into genres from audio features and title text. Each step is an MLflow project, Hydra handles config, and Weights & Biases versions the data and models.

## What's inside

- `main.py` and `config.yaml` define the pipeline
- `download/`, `preprocess/`, `check_data/`, `segregate/`, `random_forest/` and `evaluate/` are the steps

## How to run

```bash
pip install -r requirements.txt
wandb login
mlflow run .
```

You need conda, since MLflow builds each step's environment. Change `scikit-learn=0.24.1` to `scikit-learn>=1.0` in `random_forest/conda.yml` and `evaluate/conda.yml`, and delete `min_impurity_split` from `config.yaml`, first.
