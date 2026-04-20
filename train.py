#!/usr/bin/env python3
"""
Training pipeline for the spam detection classifier.

Usage:
    python train.py --dataset <path_to_csv> [--test-ratio 0.2] [--version 1]
"""

import argparse
import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

MODEL_STORE = os.path.join(os.path.dirname(__file__), "Model_Store")


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the CSV dataset. Supports both text/label and v1/v2 column naming."""
    if not os.path.exists(path):
        logger.error("Dataset file not found: %s", path)
        sys.exit(1)

    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception as exc:
        logger.error("Failed to parse dataset file '%s': %s", path, exc)
        sys.exit(1)

    # Support UCI spam dataset (v1=label, v2=text) as well as text/label columns
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    missing = [col for col in ("text", "label") if col not in df.columns]
    if missing:
        logger.error("Dataset is missing required column(s): %s", ", ".join(missing))
        sys.exit(1)

    df = df[["text", "label"]].dropna()

    if len(df) == 0:
        logger.error("Dataset is empty after dropping null values.")
        sys.exit(1)

    return df


def train(dataset_path: str, test_ratio: float, version: int) -> None:
    df = load_dataset(dataset_path)

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True)),
        ("clf", MultinomialNB()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_STORE, exist_ok=True)
    model_path = os.path.join(MODEL_STORE, f"model_v{version}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Model saved to %s", model_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the spam detection classifier.")
    parser.add_argument("--dataset", required=True, help="Path to labeled CSV file with 'text' and 'label' columns.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of data to use for testing (default: 0.2).")
    parser.add_argument("--version", type=int, default=1, help="Model version number used in the output filename (default: 1).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args.dataset, args.test_ratio, args.version)
