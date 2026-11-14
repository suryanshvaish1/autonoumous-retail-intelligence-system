# train_model.py - run ONCE to train and save the model
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

CAT = ["zone", "gender", "age_group"]
NUM = ["dwell_seconds", "hour", "is_weekend", "promotion_active"]
FEATURES = CAT + NUM

df = pd.read_csv("synthetic_retail_visitor_data.csv")
X = df[FEATURES].copy()
X[["is_weekend", "promotion_active"]] = X[["is_weekend", "promotion_active"]].astype(int)
y = df["converted"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])
model = Pipeline([
    ("pre", pre),
    ("clf", GradientBoostingClassifier(random_state=42)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 3),
    "roc_auc": round(roc_auc_score(y_test, proba), 3),
    "train_rows": len(X_train),
}

# Feature importance, grouped back to the original columns
names = model.named_steps["pre"].get_feature_names_out()
imps = model.named_steps["clf"].feature_importances_
importance = {f: 0.0 for f in FEATURES}
for name, imp in zip(names, imps):
    clean = name.split("__", 1)[1]
    for f in FEATURES:
        if clean == f or clean.startswith(f + "_"):
            importance[f] += float(imp)
            break

joblib.dump(
    {"model": model, "features": FEATURES, "metrics": metrics, "importance": importance},
    "model.joblib",
)
print("[SUCCESS] Model saved as model.joblib")
print(metrics)
print({k: round(v, 3) for k, v in sorted(importance.items(), key=lambda x: -x[1])})