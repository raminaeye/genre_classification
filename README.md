# Music Genre Classification

An end-to-end ML pipeline that classifies music tracks into genres (Dark Trap, Rap, techno, trance, and more) from Spotify-style audio features such as danceability, energy, loudness, and tempo. Originally a Udacity MLOps course exercise, it is a working example of how to structure a multi-step ML workflow with experiment tracking.

## What the pipeline does

Six steps, each an independent MLflow project that passes data to the next through Weights and Biases artifacts:

1. **download** - Streams the raw `genres_mod.parquet` dataset from its URL and logs it as a W&B artifact.
2. **preprocess** - Drops duplicate rows and engineers a `text_feature` column by concatenating the track `title` and `song_name` for TF-IDF use.
3. **check_data** - A pytest suite that validates the sample data: expected columns and dtypes, known genre labels, plausible value ranges, and a Kolmogorov-Smirnov drift check against a reference dataset.
4. **segregate** - Splits the preprocessed data into stratified train/test CSVs (seeded, reproducible).
5. **random_forest** - Trains a RandomForestClassifier on three feature tracks: standardized numerical features, ordinal-encoded categoricals, and TF-IDF text. Reports macro one-vs-rest ROC AUC on a validation split, then logs a feature-importance chart and a normalized confusion matrix.
6. **evaluate** - Loads the exported model and scores it on the held-out test set with the same ROC AUC and confusion matrix.

## Repo structure

```
main.py               # Pipeline entry point: reads config.yaml, launches each step as an MLflow run
config.yaml           # All knobs: steps to run, data URL, split sizes, random seeds, model hyperparameters
requirements.txt      # Python dependencies
download/             # Step 1: download_data.py
preprocess/           # Step 2: run.py
check_data/           # Step 3: conftest.py + test_data.py (pytest data tests)
segregate/            # Step 4: run.py
random_forest/        # Step 5: run.py (pipeline, training, plots, export)
evaluate/             # Step 6: run.py
```

Each step directory also carries its own `MLproject` and `conda.yml` so it can run standalone. The `mlruns/`, `wandb/`, `outputs/`, and `artifacts/` directories are logs and artifacts from previous runs; they are committed history, not source.

## How to run

Prerequisites: Python 3.8+ and a Weights and Biases account (`wandb login` once, or set `WANDB_API_KEY`).

```bash
pip install -r requirements.txt

# Full pipeline
mlflow run .

# Only some steps (comma-separated)
mlflow run . -P hydra_options="main.execute_steps=download,preprocess,check_data"

# Override config values, e.g. a different random seed
mlflow run . -P hydra_options="main.random_seed=123"
```

To run a single step directly, go into its directory and call it, for example:

```bash
cd download
python download_data.py --file_url <url> --artifact_name raw_data.parquet \
    --artifact_type raw_data --artifact_description "Data as downloaded"
```

Key settings live in `config.yaml`: the dataset URL, `test_size`, the stratify column, and the full `random_forest_pipeline` block (model hyperparameters, TF-IDF `max_features`, and the numerical/categorical/NLP feature lists).

## What you learn

- Orchestrating multi-step ML workflows with MLflow projects and Hydra configs
- Tracking datasets and models as versioned W&B artifacts between steps
- Writing data validation tests (schema, ranges, class names, distribution drift) that gate a pipeline
- Building a mixed preprocessing pipeline (scaling, encoding, TF-IDF) with sklearn ColumnTransformer
- Evaluating a multiclass classifier with macro ROC AUC and confusion matrices
