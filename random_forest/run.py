#!/usr/bin/env python
"""Train a Random Forest genre classifier inside a preprocessing pipeline.

Downloads the training split from W&B, builds a sklearn pipeline that
handles numerical, categorical, and text features, splits off a validation
set, trains a RandomForestClassifier, scores it with macro ROC AUC (one-vs-rest),
optionally exports the fitted pipeline to MLflow, and logs feature-importance
and confusion-matrix plots to W&B.
"""
import argparse
import itertools
import logging
import os

import yaml
import tempfile
import mlflow
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import wandb
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """Train, validate, and optionally export the Random Forest genre model.

    Pulls the training artifact from W&B, carves a validation split out of
    it, fits the preprocessing + RandomForest pipeline from
    :func:`get_training_inference_pipeline`, reports macro one-vs-rest ROC
    AUC on validation, exports the model when requested, and logs a
    feature-importance bar chart and a normalized confusion matrix.

    Args:
        args: Parsed CLI arguments (see ``__main__`` for the full list).
    """

    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify] if args.stratify != "null" else None,
        random_state=args.random_seed,
    )

    logger.info("Setting up pipeline")

    pipe, used_columns = get_training_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train[used_columns], y_train)

    # Evaluate
    pred = pipe.predict(X_val[used_columns])
    pred_proba = pipe.predict_proba(X_val[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    # Export if required
    if args.export_artifact != "null":

        export_model(run, pipe, used_columns, X_val, pred, args.export_artifact)

    # Some useful plots
    fig_feat_imp = plot_feature_importance(pipe)

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_val[used_columns],
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
            "confusion_matrix": wandb.Image(fig_cm),
        }
    )


def export_model(run, pipe, used_columns, X_val, val_pred, export_artifact):
    """Export the fitted pipeline as an MLflow model and log it to W&B.

    Args:
        run: The active W&B run used to log the artifact.
        pipe: The fitted sklearn pipeline to export.
        used_columns: Feature columns the pipeline was trained on, used to
            infer the model signature.
        X_val: Validation features (a couple of rows double as input example).
        val_pred: Validation predictions used for the signature output.
        export_artifact: Name for the W&B artifact holding the exported model.
    """

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


def plot_feature_importance(pipe):
    """Bar chart of per-feature importance for the fitted pipeline.

    Takes the importances of the numerical and categorical features as-is and
    collapses the TF-IDF text dimensions into a single "title + song_name"
    bar by summing them, so the text signal is comparable with the tabular
    features at a glance.

    Args:
        pipe: The fitted pipeline (expects "preprocessor" and "classifier"
            steps).

    Returns:
        The matplotlib figure with the sorted importance bars.
    """
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names) :])
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_training_inference_pipeline(args):
    """Build the preprocessing + RandomForest pipeline from the YAML config.

    Assembles three preprocessing tracks in a ColumnTransformer:
    median-imputed + standardized numerical features, constant-imputed +
    ordinal-encoded categorical features, and a TF-IDF text track (with a
    reshape so SimpleImputer's 2-D output feeds TfidfVectorizer's 1-D
    input). A RandomForestClassifier built from the same config is appended,
    so the returned pipeline is used for both training and inference.

    Args:
        args: Parsed CLI arguments; ``args.model_config`` is the path to the
            YAML file with the ``random_forest``, ``tfidf``, and ``features``
            sections.

    Returns:
        Tuple of (pipeline, used_columns), where used_columns lists the raw
        dataframe columns the pipeline consumes.
    """
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features
    # Categorical preprocessing pipeline
    categorical_features = sorted(model_config["features"]["categorical"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
    )
    # Numerical preprocessing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    # Textual ("nlp") preprocessing pipeline
    nlp_features = sorted(model_config["features"]["nlp"])
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=True, max_features=model_config["tfidf"]["max_features"]
        ),
    )
    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    used_columns = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config["random_forest"])),
        ]
    )
    return pipe, used_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    go(args)
