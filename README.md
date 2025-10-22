# 🏭 Manufacturing Defect Detection – Model Optimization Project

This project explores **machine learning model selection and optimization** for predicting manufacturing defects.  

- It focuses on building, tuning, and evaluating multiple classification models. Ultimately finding the most effective one for high-accuracy defect prediction.

---

## 📘 Overview

The goal was to identify whether a product batch is **defective** based on production and quality metrics.  
The pipeline includes:

1. **Data preprocessing** and scaling  
2. **Model comparison** using 13 classifiers  
3. **Cross-validation (5-fold Stratified K-Fold)** with metrics:
   - Accuracy  
   - F1-score  
   - ROC-AUC  
4. **Hyperparameter tuning** on the top 3 models (LightGBM, HistGradientBoosting, GradientBoosting)  
5. **Feature importance & selection**  
6. **Threshold optimization** (F1-maximization)  
7. **Model stacking** and final performance comparison  

The notebook includes all plots, reports, and metrics.

---

## Final Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:--|--:|--:|--:|--:|--:|
| **Stack (all features) @best-F1** | **0.952160** | 0.950877 | **0.994495** | **0.972197** | 0.837891 |
| Best model (top features) @best-F1 – LightGBM | 0.950617 | 0.950791 | 0.992661 | 0.971275 | **0.849826** |
| Stack (all features) @0.50 | 0.942901 | 0.950355 | 0.983486 | 0.966637 | 0.837891 |

### **Final selected model:**  
**Stacking Classifier (LightGBM + GradientBoosting + HistGradientBoosting)** using all features and an optimized threshold (max F1).

---

## How to Run

```bash
git clone https://github.com/<your-username>/manufacturing-defect-detection.git
cd manufacturing-defect-detection

pip install -r requirements.txt

jupyter lab notebooks/01_data_exploration.ipynb
    - This notebook runs the entire pipeline from model comparison through final evaluation.

```

## To load and use the final saved model for new predictions:

```py
import joblib, json, pandas as pd

# Load saved model and threshold
model = joblib.load("notebooks/models/best_model.pkl")
thr = json.load(open("notebooks/models/threshold.json"))["threshold"]
feats = [l.strip() for l in open("notebooks/models/features.txt")]

# Load new data and predict
new = pd.read_csv("data/manufacturing_defect_dataset.csv")[feats]
proba = model.predict_proba(new)[:, 1]
pred  = (proba >= thr).astype(int)

print(pred[:10])
```

## Key Insights

- **LightGBM** achieved the highest ROC-AUC, confirming its ranking reliability.  
- **Stacking ensemble** Stacking the top-performing models (LightGBM, HistGradientBoosting, GradientBoosting) marginally improved F1-score by achieving a better balance between recall and precision, confirming ensemble robustness.
- **Threshold tuning** improved detection sensitivity (recall ↑ without overfitting).  
- **Feature importance analysis** Feature importance analysis revealed that product quality indicators (defect_rate, quality_score) and operational efficiency factors (production_volume, energy_efficiency, maintenance_hours) were the most influential in predicting manufacturing defects.

---

## Tech Stack

- **Python:** 3.13  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`,  `joblib`  
- **Environment:** Jupyter Notebook

---

## License

This project is released for educational and demonstration purposes.

- **declaimer** This dataset is synthetic; use insights directionally and validate on real-world process data before acting.
- [dataset] (https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)

---

## ✨ Author

**Mohamed Ragab Awad Bahr**  


