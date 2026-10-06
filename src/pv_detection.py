"""Train and validate a PV-ownership detector from the pattern features in
``src/pv_features.py`` against the surveyed ``Installation_HasPVSystem`` flag.

The flag is treated purely as a validation target (per the task brief): the
households used for the Level-1 grouped forecasting step are split using the
*known* flag directly, not this detector's output. This script only reports
how well pattern-based detection recovers that flag, and additionally applies
the fitted detector to the 165 households with no survey answer, as an
optional secondary grouping.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.pv_features import FEATURE_COLUMNS

FEATURES_PATH = "reports/pv_features.csv"


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, index_col=0)


def main() -> None:
    df = load_features()
    known = df[df["Installation_HasPVSystem"].isin([True, False])].copy()
    X = known[FEATURE_COLUMNS]
    y = known["Installation_HasPVSystem"].astype(bool)

    clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, learning_rate=0.1, random_state=0
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Out-of-fold predictions: every household is scored by a model that never
    # saw it during training, so these metrics are a fair estimate of how the
    # detector would do on an unsurveyed household.
    oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred = oof_proba >= 0.5

    report = classification_report(y, oof_pred, target_names=["no_pv", "pv"], output_dict=True)
    cm = confusion_matrix(y, oof_pred)
    auc = roc_auc_score(y, oof_proba)

    print("=== 5-fold out-of-fold PV-detector performance vs surveyed flag ===")
    print(f"n = {len(y)} households with known flag ({y.sum()} PV, {(~y).sum()} no-PV)")
    print(f"ROC AUC: {auc:.3f}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(
        f"PV class  -> precision {report['pv']['precision']:.3f}  "
        f"recall {report['pv']['recall']:.3f}  f1 {report['pv']['f1-score']:.3f}"
    )
    print(
        f"No-PV cls -> precision {report['no_pv']['precision']:.3f}  "
        f"recall {report['no_pv']['recall']:.3f}  f1 {report['no_pv']['f1-score']:.3f}"
    )
    print("Confusion matrix [rows=true no_pv/pv, cols=pred no_pv/pv]:")
    print(cm)

    # Fit on all known-label data, then score the unsurveyed households as a
    # secondary, detector-based grouping (not used for the Level-1 forecasting
    # split itself -- see module docstring).
    clf.fit(X, y)
    unknown = df[df["Installation_HasPVSystem"].isna()].copy()
    unknown_proba = clf.predict_proba(unknown[FEATURE_COLUMNS])[:, 1]
    unknown_pred = pd.Series(unknown_proba >= 0.5, index=unknown.index, name="detected_pv")

    print(f"\nUnsurveyed households scored: {len(unknown)}")
    print(f"Detector says PV: {unknown_pred.sum()}, no-PV: {(~unknown_pred).sum()}")

    # Persist everything the downstream aggregation/report steps need.
    # HistGradientBoostingClassifier has no built-in feature_importances_, so
    # use permutation importance (drop in mean ROC AUC when a column is shuffled).
    perm = permutation_importance(
        clf, X, y, scoring="roc_auc", n_repeats=20, random_state=0
    )
    feature_importance = pd.Series(
        perm.importances_mean, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    out = df.copy()
    out["detector_proba_pv"] = np.nan
    out.loc[known.index, "detector_proba_pv"] = oof_proba  # out-of-fold, no leakage
    out.loc[unknown.index, "detector_proba_pv"] = unknown_proba  # from full-fit model
    out["detector_pred_pv"] = out["detector_proba_pv"] >= 0.5
    out.to_csv("reports/pv_detector_scores.csv")

    metrics = {
        "n_known": int(len(y)),
        "n_pv": int(y.sum()),
        "n_no_pv": int((~y).sum()),
        "roc_auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "pv_precision": float(report["pv"]["precision"]),
        "pv_recall": float(report["pv"]["recall"]),
        "pv_f1": float(report["pv"]["f1-score"]),
        "no_pv_precision": float(report["no_pv"]["precision"]),
        "no_pv_recall": float(report["no_pv"]["recall"]),
        "no_pv_f1": float(report["no_pv"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "n_unsurveyed": int(len(unknown)),
        "n_unsurveyed_detected_pv": int(unknown_pred.sum()),
        "feature_importance": feature_importance.round(4).to_dict(),
    }
    with open("reports/pv_detection_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nFeature importances:")
    print(feature_importance)


if __name__ == "__main__":
    main()
