import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .features import prepare_training_data, prepare_prediction_features, FEATURE_COLS


def train_random_forest(X: np.ndarray, y: np.ndarray) -> Tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    train_acc = accuracy_score(y, model.predict(X_scaled))
    return model, scaler, {"train_accuracy": round(train_acc, 4)}


def train_xgboost(X: np.ndarray, y: np.ndarray) -> Tuple:
    if not XGBOOST_AVAILABLE:
        return None, None, {"error": "XGBoost not available"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_scaled, y, verbose=False)
    train_acc = accuracy_score(y, model.predict(X_scaled))
    return model, scaler, {"train_accuracy": round(train_acc, 4)}


def cross_validate_model(model_type: str, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if model_type == "rf":
            m = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        elif model_type == "xgb" and XGBOOST_AVAILABLE:
            m = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
        else:
            continue

        m.fit(X_train_s, y_train)
        acc = accuracy_score(y_val, m.predict(X_val_s))
        accuracies.append(acc)

    return {
        "mean_accuracy": round(np.mean(accuracies), 4) if accuracies else 0.0,
        "std_accuracy": round(np.std(accuracies), 4) if accuracies else 0.0,
        "n_splits": n_splits,
    }


@st.cache_data(show_spinner=False, ttl=5)
def train_and_predict(_df_hash: str, df_serialized: str, symbol: str) -> Dict:
    import json
    from io import StringIO
    df = pd.read_json(StringIO(df_serialized))

    X, y, _ = prepare_training_data(df, horizon=5)
    if len(X) < 100:
        return {"error": "Insufficient data for training", "rf": None, "xgb": None}

    min_train = max(50, int(len(X) * 0.8))
    X_train, y_train = X[:min_train], y[:min_train]
    X_test, y_test = X[min_train:], y[min_train:]

    rf_model, rf_scaler, rf_meta = train_random_forest(X_train, y_train)
    if len(X_test) > 0:
        rf_test_acc = accuracy_score(y_test, rf_model.predict(rf_scaler.transform(X_test)))
        rf_meta["test_accuracy"] = round(rf_test_acc, 4)

    xgb_model, xgb_scaler, xgb_meta = train_xgboost(X_train, y_train)
    if xgb_model is not None and len(X_test) > 0:
        xgb_test_acc = accuracy_score(y_test, xgb_model.predict(xgb_scaler.transform(X_test)))
        xgb_meta["test_accuracy"] = round(xgb_test_acc, 4)

    X_latest, df_feat = prepare_prediction_features(df)
    rf_pred, rf_prob = None, None
    xgb_pred, xgb_prob = None, None

    if X_latest is not None and len(X_latest) > 0:
        X_last = X_latest[-1:] 
        X_last_rf = rf_scaler.transform(X_last)
        rf_pred = int(rf_model.predict(X_last_rf)[0])
        rf_prob = float(rf_model.predict_proba(X_last_rf)[0][1])

        if xgb_model is not None:
            X_last_xgb = xgb_scaler.transform(X_last)
            xgb_pred = int(xgb_model.predict(X_last_xgb)[0])
            xgb_prob = float(xgb_model.predict_proba(X_last_xgb)[0][1])

    feature_importance = {}
    if hasattr(rf_model, "feature_importances_"):
        fi = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
        feature_importance = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10])

    combined_prob = None
    if rf_prob is not None and xgb_prob is not None:
        combined_prob = (rf_prob + xgb_prob) / 2
    elif rf_prob is not None:
        combined_prob = rf_prob

    return {
        "rf": {"prediction": rf_pred, "probability": rf_prob, "meta": rf_meta},
        "xgb": {"prediction": xgb_pred, "probability": xgb_prob, "meta": xgb_meta},
        "combined_probability": combined_prob,
        "direction": "UP" if (combined_prob or 0.5) > 0.5 else "DOWN",
        "confidence": abs((combined_prob or 0.5) - 0.5) * 2,
        "feature_importance": feature_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
