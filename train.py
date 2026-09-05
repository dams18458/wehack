#!/usr/bin/env python3
"""
Model Training Script for Vocal Distress Classification.

Trains a scikit-learn RandomForestClassifier on extracted acoustic biomarkers
from RAVDESS and CREMA-D subsets.

Evaluates performance on a held-out test split, explicitly reporting:
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Feature Importance ranking

Serializes the trained model bundle to models/distress_model.joblib.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from distress_detector.classifier import DistressClassifier
from distress_detector.features import FEATURE_NAMES


def load_dataset_from_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Load feature matrix X and target labels y from CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}. Run prepare_dataset.py first.")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV at '{csv_path}' is empty.")

    X = []
    y = []
    for row in rows:
        feats = [float(row[name]) for name in FEATURE_NAMES]
        X.append(feats)
        y.append(int(row["label"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), rows


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest vocal distress classifier.")
    parser.add_argument("--csv", type=str, default="data/distress_features.csv", help="Path to input features CSV.")
    parser.add_argument("--output-model", type=str, default="models/distress_model.joblib", help="Output model path.")
    parser.add_argument("--metrics-out", type=str, default="models/evaluation_metrics.json", help="Path to write JSON evaluation metrics.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split proportion (default: 0.20).")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest number of trees (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    model_path = Path(args.output_model).resolve()
    metrics_path = Path(args.metrics_out).resolve()

    print("=" * 75)
    print("      VOCAL DISTRESS RANDOM FOREST MODEL TRAINING")
    print("=" * 75)
    print(f"Dataset:       {csv_path}")
    print(f"Output Model:  {model_path}")
    print(f"Test Split:    {args.test_size * 100:.0f}% (Stratified)")
    print("=" * 75)

    X, y, raw_rows = load_dataset_from_csv(csv_path)
    total_samples = len(y)
    distress_count = int(np.sum(y == 1))
    neutral_count = int(np.sum(y == 0))

    print(f"Loaded {total_samples} samples | Distress: {distress_count} | Neutral: {neutral_count}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    print(f"Training set:  {len(y_train)} samples")
    print(f"Held-out test: {len(y_test)} samples")

    # Fit classifier
    classifier = DistressClassifier(
        n_estimators=args.n_estimators,
        max_depth=6,
        random_state=args.seed,
    )
    classifier.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_test_scaled = classifier.scaler.transform(X_test)
    y_probs = classifier.model.predict_proba(y_test_scaled)[:, 1]
    y_preds = (y_probs >= 0.50).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rate calculations
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    acc = float(accuracy_score(y_test, y_preds))
    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_probs)) if len(np.unique(y_test)) > 1 else 0.0

    print("\n" + "=" * 75)
    print(" HELD-OUT TEST SPLIT EVALUATION RESULTS")
    print("=" * 75)
    print(f"Confusion Matrix:")
    print(f"  True Negatives (Neutral correct):     {tn:2d}")
    print(f"  False Positives (Neutral misflagged): {fp:2d}  --> FPR: {fpr * 100:.1f}%")
    print(f"  False Negatives (Distress missed):    {fn:2d}  --> FNR: {fnr * 100:.1f}%")
    print(f"  True Positives (Distress correct):    {tp:2d}")
    print("-" * 75)
    print(f"  Accuracy:                 {acc * 100:.2f}%")
    print(f"  Precision:                {prec * 100:.2f}%")
    print(f"  Recall (Sensitivity):     {rec * 100:.2f}%")
    print(f"  F1-Score:                 {f1:.3f}")
    print(f"  ROC-AUC Score:            {roc_auc:.3f}")
    print("=" * 75)

    # Feature Importance
    importances = classifier.model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importance Rankings:")
    for rank, idx in enumerate(sorted_idx, 1):
        feat_name = FEATURE_NAMES[idx]
        imp = importances[idx]
        bar = "#" * int(round(imp * 40))
        print(f"  {rank}. {feat_name:<24} : {imp:6.3f} | {bar}")

    # Save model artifact
    saved_model = classifier.save(model_path)
    print(f"\nTrained model successfully serialized to:\n  {saved_model}")

    # Save metrics JSON
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dict = {
        "dataset": str(csv_path.name),
        "total_samples": total_samples,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_accuracy": round(acc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "feature_importances": {
            FEATURE_NAMES[idx]: round(float(importances[idx]), 4) for idx in sorted_idx
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to:\n  {metrics_path}")


if __name__ == "__main__":
    main()