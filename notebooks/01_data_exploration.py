#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


import pandas as pd
df = pd.read_csv("../data/manufacturing_defect_dataset.csv")

df.head()


# In[3]:


import re
df.columns = [
    re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower()
    for col in df.columns
]
df.head()


# In[4]:


n_rows, n_cols = df.shape
summary = {
    "rows": n_rows,
    "columns": n_cols,
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing_by_col": df.isna().sum().to_dict(),
    "duplicates": int(df.duplicated().sum())
}

summary


# In[5]:


df.describe()


# In[6]:


df['defect_status'].value_counts(normalize=True) * 100


# In[7]:


import numpy as np

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Identify target and feature columns
target_col = 'defect_status'

# Ensure target is integer binary {0,1}
df[target_col] = df[target_col].astype(int)

# Separate features (X) and target (y)
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()

# Class balance
class_counts = y.value_counts().sort_index()
class_ratio = {int(k): int(v) for k, v in class_counts.items()}
minority_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 1 else 1.0

print("Class distribution:")
for label, count in class_ratio.items():
    print(f"  Class {label}: {count} samples ({count / len(y) * 100:.2f}%)")

print(f"\nMinority-to-majority ratio: {minority_ratio:.2f}")
if minority_ratio < 0.5:
    print("Significant class imbalance detected!")


print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("-------Shape of train-test data---------")
print(f'Train feature shape: {X_train.shape}')
print(f'Test feature shape: {X_test.shape}')


# In[9]:


# Preprocessing Pipeline
# Only numeric columns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import warnings, lightgbm as lgb
warnings.filterwarnings("ignore")

class SilentLogger:
    def info(self, msg): pass
    def warning(self, msg): pass

lgb.register_logger(SilentLogger())

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
preprocessor_scaled = ColumnTransformer(
    transformers=[("num", numeric_transformer, X.columns)]
)
preprocessor_passthrough = ColumnTransformer(
    transformers=[("num", "passthrough", X.columns)]
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    # Linear Models
    "LogisticRegression": Pipeline(steps=[
        ("prep", preprocessor_scaled),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "LinearSVC": Pipeline(steps=[
        ("prep", preprocessor_scaled),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ]),

    # Kernel-based Models
    "SVC": Pipeline(steps=[
        ("prep", preprocessor_scaled),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42))
    ]),

    # Tree-Based (Nonlinear) Models
    "DecisionTree": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ]),
    "RandomForest": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            random_state=42, class_weight="balanced_subsample"
        ))
    ]),
    "ExtraTrees": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", ExtraTreesClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample"
        ))
    ]),

    # Boosting Ensembles
    "AdaBoost": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", AdaBoostClassifier(n_estimators=400, random_state=42))
    ]),
    "GradientBoosting": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", GradientBoostingClassifier(n_estimators=400, random_state=42))
    ]),
    "HistGradientBoosting": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", HistGradientBoostingClassifier(random_state=42))
    ]),
    "XGBoost": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            use_label_encoder=False, random_state=42, scale_pos_weight=5,  # for imbalance 
        ))
    ]),
    "LightGBM": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.1, random_state=42, class_weight="balanced",
            verbosity=-1,          # suppress info/warnings
            force_col_wise=True 
        ))
    ]),
    "CatBoost": Pipeline(steps=[
        ("prep", preprocessor_passthrough),
        ("clf", CatBoostClassifier(
            iterations=400, learning_rate=0.1, depth=6,
            verbose=False, random_state=42, auto_class_weights="Balanced"
        ))
    ]),

    # Distance- or Probabilistic-based Models
    "KNN": Pipeline(steps=[
        ("prep", preprocessor_scaled),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ]),
    "NaiveBayes": Pipeline(steps=[
        ("prep", preprocessor_scaled),
        ("clf", GaussianNB())
    ])
}


# In[10]:


from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}
cv_results = {}

for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    cv_results[name] = {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
    }

cv_df = (
    pd.DataFrame(cv_results)
    .T.sort_values(by="roc_auc_mean", ascending=False)
    .reset_index().rename(columns={"index": "model"})
)
cv_df


# In[11]:


top3 = cv_df.head(3)["model"].tolist()

print(top3)

tuned_models = {}


# In[12]:


from scipy.stats import randint, uniform, loguniform

param_spaces = {
    # --- Linear / Margin-based ---
    "LogisticRegression": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear", "saga"],
    },
    "SVC": {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__gamma": loguniform(1e-3, 1e1),
        "clf__kernel": ["rbf"],  # you already set probability=True in the pipeline
    },
    "LinearSVC": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__loss": ["hinge", "squared_hinge"],
    },

    # --- Trees / Forests ---
    "DecisionTree": {
        "clf__max_depth": randint(3, 21),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 11),
    },
    "RandomForest": {
        "clf__n_estimators": randint(300, 1001),
        "clf__max_depth": [None] + list(range(6, 21, 2)),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 11),
        "clf__max_features": ["sqrt", "log2", None],
    },
    "ExtraTrees": {
        "clf__n_estimators": randint(300, 1001),
        "clf__max_depth": [None] + list(range(6, 21, 2)),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 11),
        "clf__max_features": ["sqrt", "log2", None],
    },

    # --- Boosted trees (sklearn) ---
    "GradientBoosting": {
        "clf__n_estimators": randint(300, 1001),
        "clf__learning_rate": loguniform(1e-2, 2e-1),
        "clf__max_depth": randint(2, 6),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 21),
        "clf__subsample": uniform(0.6, 0.4),  # 0.6–1.0
    },
    "HistGradientBoosting": {
        "clf__learning_rate": loguniform(1e-2, 2e-1),
        "clf__max_depth": randint(3, 21),           # can also try None
        "clf__max_leaf_nodes": randint(15, 64),
        "clf__min_samples_leaf": randint(5, 51),
        "clf__l2_regularization": uniform(0.0, 1.0),
        "clf__max_bins": randint(128, 256),
        # early_stopping True in your pipe is fine
    },

    # --- Gradient boosting (external) ---
    "XGBoost": {
        "clf__n_estimators": randint(400, 1201),
        "clf__learning_rate": loguniform(1e-2, 2e-1),
        "clf__max_depth": randint(3, 11),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 11),
        "clf__gamma": uniform(0.0, 5.0),
        "clf__scale_pos_weight": uniform(3.0, 6.0),  # ~3–9 around your class ratio
    },
    "LightGBM": {
        "clf__num_leaves": randint(15, 64),
        "clf__max_depth": randint(3, 11),            # keep modest; -1 also possible
        "clf__learning_rate": loguniform(1e-2, 2e-1),
        "clf__n_estimators": randint(400, 1201),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_samples": randint(5, 51),
        "clf__reg_alpha": uniform(0.0, 1.0),
        "clf__reg_lambda": uniform(0.0, 1.0),
    },
    "CatBoost": {
        "clf__iterations": randint(400, 1201),
        "clf__learning_rate": loguniform(1e-2, 2e-1),
        "clf__depth": randint(4, 11),
        "clf__l2_leaf_reg": loguniform(1e-2, 10),
        "clf__subsample": uniform(0.6, 0.4),
    },

    # --- Instance / Probabilistic ---
    "KNN": {
        "clf__n_neighbors": randint(3, 51),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "NaiveBayes": {
        "clf__var_smoothing": loguniform(1e-12, 1e-6),
    },
}

scoring = "roc_auc"   # primary metric for search

tuning_rows = []

for name in top3:
    pipe = models[name]
    space = param_spaces.get(name, {})

    if space:  # do a randomized search
        n_iter = 40 if name in {"LightGBM","XGBoost","HistGradientBoosting","GradientBoosting"} else 25
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        tuned_models[name] = best_est
        tuning_rows.append({"model": name, "best_cv_roc_auc": search.best_score_, "best_params": search.best_params_})
    else:      # no space provided -> just fit as-is
        best_est = pipe.fit(X_train, y_train)
        tuned_models[name] = best_est
        tuning_rows.append({"model": name, "best_cv_roc_auc": np.nan, "best_params": "(no tuning space)"})

result_df = pd.DataFrame(tuning_rows).sort_values("best_cv_roc_auc", ascending=False)
result_df.head()


# In[13]:


from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

best_model_name = result_df.iloc[0]["model"]
best_model = tuned_models[best_model_name]

# Fit best model on all training data
best_model.fit(X_train, y_train)

# ---------- Test Set Evaluation ----------
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

test_acc_all_features = accuracy_score(y_test, y_pred)
test_pr_all_features, test_rc_all_features, test_f1_all_features, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
test_roc_all_features = roc_auc_score(y_test, y_proba)

report = classification_report(y_test, y_pred, digits=3)

print(best_model_name)
print(report)

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Score": [test_acc_all_features, test_pr_all_features, test_rc_all_features, test_f1_all_features, test_roc_all_features]
})

print(summary)


# In[14]:


clf = best_model.named_steps["clf"]

# Extract feature importances
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Visualize
importances.plot(kind="barh", figsize=(8, 6))
plt.title("Feature Importance (LightGBM)")
plt.show()

# Optionally, keep top 10 features
top_features = importances.head(10).index.tolist()
X_top = X[top_features]


# In[15]:


from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

X_train_top, X_test_top, y_train, y_test = train_test_split(
    X_top, y, test_size=0.20, random_state=42, stratify=y
)
cols = X_top.columns.tolist()

clf = clone(best_model.named_steps["clf"])  # fresh copy of the tuned classifier

# Trees/boosting don’t need scaling; passthrough is fine
new_prep = ColumnTransformer([("num", "passthrough", cols)], remainder="drop")

best_model_top = Pipeline([
    ("prep", new_prep),
    ("clf", clf),
])

# 3) Fit on the top-feature training set
best_model_top.fit(X_train_top, y_train)


y_proba = best_model_top.predict_proba(X_test_top)[:, 1]


prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1[:-1])
best_thr = thr[best_idx]
print(f"Best threshold by F1: {best_thr:.4f}")


y_pred_opt = (y_proba >= best_thr).astype(int)

test_acc = accuracy_score(y_test, y_pred_opt)
test_pr, test_rc, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_opt, average="binary", zero_division=0)
test_roc = roc_auc_score(y_test, y_proba)

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Score": [test_acc, test_pr, test_rc, test_f1, test_roc]
})

report = classification_report(y_test, y_pred_opt, digits=3)

print(report)
print(summary)


# In[16]:


# ---- Metrics before and after feature selection ----
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

before = [test_acc_all_features, test_pr_all_features, test_rc_all_features, test_f1_all_features, test_roc_all_features]
after  = [test_acc, test_pr, test_rc, test_f1, test_roc]

x = np.arange(len(metrics))
width = 0.35  # bar width

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, before, width, label="Before Feature Selection")
bars2 = plt.bar(x + width/2, after,  width, label="After Feature & Threshold Selection")

# Add labels and title
plt.ylabel("Score")
plt.title("Model Performance Before vs After Feature & Threshold Selection" + ' ' + best_model_name)
plt.xticks(x, metrics)
plt.ylim(0.8, 1.0)
plt.legend(loc="lower right")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# In[17]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={test_roc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_opt, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()


# In[19]:


from sklearn.ensemble import StackingClassifier

base_estimators = []
for name in top3:
    est = tuned_models.get(name, models[name])
    base_estimators.append((name, est))

# meta-learner: simple, strong, well-calibrated
final_lr = LogisticRegression(max_iter=2000, class_weight="balanced")

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_lr,
    stack_method="predict_proba",   # use probabilities from base learners
    passthrough=False,              # set True to include original features for meta
    cv=cv
)

# --- 3) fit on ALL features ---
stack.fit(X_train, y_train)

# --- 4) Evaluate on test at thr=0.50 and at best-F1 threshold ---
def metrics_at_threshold(y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    return acc, pr, rc, f1, auc, y_pred

# probabilities from the stack
y_proba_stack = stack.predict_proba(X_test)[:, 1]

# default 0.50
acc50, pr50, rc50, f150, auc50, ypred50 = metrics_at_threshold(y_test, y_proba_stack, 0.50)

# best F1 threshold
prec, rec, thr = precision_recall_curve(y_test, y_proba_stack)
f1s = 2*prec*rec/(prec+rec+1e-9)
best_idx = np.argmax(f1s[:-1])      # last thr element has no pair
best_thr_stack = thr[best_idx]

accB, prB, rcB, f1B, aucB, ypredB = metrics_at_threshold(y_test, y_proba_stack, best_thr_stack)

print(f"\nStacking best F1 threshold: {best_thr_stack:.4f}")
print("\nClassification report @ best F1 threshold:")
print(classification_report(y_test, ypredB, digits=3))

stack_rows = [
    {"Model": "Stack (all features) @0.50",       "Accuracy":acc50, "Precision":pr50, "Recall":rc50, "F1":f150, "ROC-AUC":auc50},
    {"Model": "Stack (all features) @best-F1",    "Accuracy":accB,  "Precision":prB,  "Recall":rcB,  "F1":f1B,  "ROC-AUC":aucB},
]


# In[28]:


lgbm_rows = [{
    "Model": "Best model (top feats) @best-F1" + " - " + best_model_name,
    "Accuracy": float(test_acc),
    "Precision": float(test_pr),
    "Recall": float(test_rc),
    "F1": float(test_f1),
    "ROC-AUC": float(test_roc)
}]

# concat comparison
compare_df = pd.DataFrame(lgbm_rows + stack_rows).sort_values("F1", ascending=False)
compare_df


# In[33]:


best_row = compare_df.iloc[0]
print(best_row["Model"])


# In[35]:


import json, joblib, pathlib

# --- paths
pathlib.Path("models").mkdir(exist_ok=True)

# <<< replace these with your actual variables >>>
winner_model = stack      # your fitted StackingClassifier pipeline
winner_threshold = best_thr_stack    # the F1-optimal threshold for the stack
feature_list = X.columns.tolist()    # stack used all columns

# 1) model
joblib.dump(winner_model, "models/best_model.pkl")

# 2) threshold
with open("models/threshold.json", "w") as f:
    json.dump({"threshold": float(winner_threshold)}, f)

# 3) features
with open("models/features.txt", "w") as f:
    for c in feature_list:
        f.write(f"{c}\n")

# 4) optional artifacts you already built
compare_df.to_csv("models/leaderboard_test.csv", index=False)   # your table with F1, AUC, etc.
summary.to_csv("models/test_metrics.csv", index=False)          # the metrics block you printed

