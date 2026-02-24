"""
train.py — STEP 1: Train the model on your supply chain dataset and save as pickle.

HOW TO RUN:
    python train.py

This will:
1. Load your CSV from data/ folder
2. Preprocess (encode, scale, feature engineer)
3. Train 3 models → select best by F1 score
4. Save everything needed for prediction to model/model_bundle.pkl

Run this ONCE. After that, only app.py needs to be run.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "supply_chain_disruption_custom_columns_3000_rows.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model_bundle.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Step 1: Load Data ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading training data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")


# ── Step 2: Feature Engineering ───────────────────────────────
print("\nSTEP 2: Feature engineering...")

def weather_category(x):
    """Convert numeric weather severity to category."""
    if x <= 3:   return "Normal"
    elif x <= 6: return "Moderate"
    else:        return "Severe"

if "weather_condition_severity" in df.columns:
    df["weather_category"] = df["weather_condition_severity"].apply(weather_category)
    df.drop("weather_condition_severity", axis=1, inplace=True)
    print("  ✓ weather_condition_severity → weather_category")


# ── Step 3: Encode Categoricals ───────────────────────────────
print("\nSTEP 3: Encoding categorical columns...")

le_dict = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
# Remove target if it's string
target_col = "disruption_occurred"
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  ✓ Encoded: {col} → {list(le.classes_)[:5]}{'...' if len(le.classes_) > 5 else ''}")


# ── Step 4: Split Features / Target ───────────────────────────
print("\nSTEP 4: Splitting features and target...")

X = df.drop(target_col, axis=1)
y = df[target_col]
feature_columns = X.columns.tolist()

print(f"  Features ({len(feature_columns)}): {feature_columns}")
print(f"  Target distribution:\n{y.value_counts().to_string()}")


# ── Step 5: Train/Test Split ───────────────────────────────────
print("\nSTEP 5: Train/test split (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")


# ── Step 6: Scale Features ─────────────────────────────────────
print("\nSTEP 6: Scaling features...")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("  ✓ StandardScaler fitted")


# ── Step 7: Train All Models ───────────────────────────────────
print("\nSTEP 7: Training models...")

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

model_results = {}
best_model, best_score, best_name = None, 0, ""

for name, model in candidates.items():
    model.fit(X_train_s, y_train)
    preds   = model.predict(X_test_s)
    f1      = f1_score(y_test, preds)
    acc     = accuracy_score(y_test, preds)
    prec    = precision_score(y_test, preds)
    rec     = recall_score(y_test, preds)

    model_results[name] = {
        "model":     model,
        "f1":        round(f1,  4),
        "accuracy":  round(acc, 4),
        "precision": round(prec,4),
        "recall":    round(rec, 4),
    }
    print(f"  {name:25s} | F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")

    if f1 > best_score:
        best_score, best_model, best_name = f1, model, name

print(f"\n  ✅ Best model: {best_name} (F1={best_score:.4f})")


# ── Step 8: Feature Importances ───────────────────────────────
print("\nSTEP 8: Extracting feature importances...")

importances = {}
if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(feature_columns, best_model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 KRIs:")
    for feat, imp in top5:
        print(f"    {feat:35s}: {imp:.4f}")
else:
    print("  (Logistic Regression — no feature importances)")


# ── Step 9: Compute Training Baseline Stats ────────────────────
print("\nSTEP 9: Computing training baseline statistics...")

# These are used later for scenario simulation (median values per feature)
training_stats = {
    "median": X.median().to_dict(),
    "mean":   X.mean().to_dict(),
    "std":    X.std().to_dict(),
    "min":    X.min().to_dict(),
    "max":    X.max().to_dict(),
}
print("  ✓ Training stats saved (median, mean, std, min, max)")


# ── Step 10: Compute Training Disruption Profile ──────────────
print("\nSTEP 10: Building training data profile for comparison...")

# Store sample of training data for comparison in app
train_profile = {
    "disruption_rate":    float(y.mean()),
    "total_records":      int(len(df)),
    "high_risk_records":  int((best_model.predict_proba(scaler.transform(X))[:, 1] >= 0.7).sum()),
    "feature_columns":    feature_columns,
}
print(f"  Training disruption rate: {train_profile['disruption_rate']*100:.1f}%")


# ── Step 11: Classification Report ────────────────────────────
print("\nSTEP 11: Full classification report for best model...")
y_pred_best = best_model.predict(X_test_s)
print(classification_report(y_test, y_pred_best, target_names=["No Disruption","Disruption"]))


# ── Step 12: Save Pickle Bundle ───────────────────────────────
print("STEP 12: Saving model bundle to pickle...")

model_bundle = {
    # Core model artifacts
    "best_model":      best_model,
    "best_name":       best_name,
    "best_score":      best_score,
    "scaler":          scaler,
    "le_dict":         le_dict,
    "feature_columns": feature_columns,

    # All 3 trained models + their scores
    "all_models":      {
        name: {
            "model":     res["model"],
            "f1":        res["f1"],
            "accuracy":  res["accuracy"],
            "precision": res["precision"],
            "recall":    res["recall"],
        }
        for name, res in model_results.items()
    },

    # Feature importances (KRIs)
    "importances":     importances,

    # Training statistics (for scenario baselines)
    "training_stats":  training_stats,

    # Training profile (for comparison against new data)
    "train_profile":   train_profile,

    # Test set predictions (for confusion matrix reference)
    "y_test":          y_test.values,
    "y_pred_test":     y_pred_best,

    # Weather category function info
    "weather_col_present": "weather_category" in feature_columns,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\n{'='*60}")
print(f"✅ Model bundle saved to: {MODEL_PATH}")
print(f"   Bundle keys: {list(model_bundle.keys())}")
print(f"{'='*60}")
print("\nNext step → run:  streamlit run app.py")
print("="*60)