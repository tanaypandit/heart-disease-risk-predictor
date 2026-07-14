# ❤️ Heart Disease Risk Prediction — End-to-End ML Project

A complete, portfolio-ready machine learning project that predicts a patient's risk
of heart disease from 13 clinical measurements, built on the Cleveland Heart
Disease dataset (UCI Machine Learning Repository).

This started as a simple logistic-regression + Streamlit mini project and has been
rebuilt into a full pipeline: EDA → feature engineering → model comparison →
hyperparameter tuning → evaluation → SHAP interpretability → a polished, deployable
Streamlit app.

## 🎯 Highlights

- **8 algorithms compared** with 5-fold stratified cross-validation (Logistic
  Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost,
  Naive Bayes)
- **Hyperparameter tuning** of the top 3 candidates via `GridSearchCV`, optimizing
  ROC-AUC
- **Feature engineering**: age group, cholesterol-per-age, heart-rate reserve, and
  a composite risk-factor count added to the 13 raw clinical inputs
- **Best model on held-out test set:** Logistic Regression — **86.7% accuracy,
  0.92 ROC-AUC, 100% precision, 71% recall** (see `reports/figures/` and the
  notebook for the full comparison table)
- **Model interpretability** with global feature importance and SHAP values
  (summary plot + per-prediction waterfall plot in the app)
- **Production-style packaging**: a single `sklearn.Pipeline` (feature engineering
  + scaling + classifier) is exported so the app only ever needs the 13 raw
  clinical inputs — no manual preprocessing to keep in sync
- **Polished Streamlit app** with a clinical-style input form, risk gauge, and
  live SHAP explanation for every prediction

## 📁 Project structure

```
heart_disease_project/
├── data/
│   └── heart_cleveland_upload.csv       # Raw dataset (297 patients, 13 features + target)
├── notebooks/
│   └── heart_disease_analysis.ipynb     # Full EDA → modeling → tuning → SHAP notebook
├── src/
│   ├── column_info.py                   # Data dictionary / column reference
│   └── feature_engineering.py           # Shared FeatureEngineer transformer
├── models/
│   ├── heart_disease_pipeline.pkl       # Final fitted pipeline (raw input -> prediction)
│   └── model_metadata.pkl               # Best model name + test metrics
├── app/
│   ├── streamlit_app.py                 # Deployable prediction app
│   └── feature_engineering.py           # Copy of the transformer (self-contained app)
├── reports/
│   └── figures/                         # All EDA & evaluation charts (PNG)
├── requirements.txt
└── README.md
```

## 🚀 Getting started

```bash
# 1. Create environment & install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-run the full training pipeline from scratch
jupyter nbconvert --to notebook --execute --inplace notebooks/heart_disease_analysis.ipynb

# 3. Launch the app
cd app
streamlit run streamlit_app.py
```

The repo already ships with a trained `models/heart_disease_pipeline.pkl`, so you
can skip straight to step 3 if you just want to try the app.

## 📊 Dataset

Source: [Cleveland Heart Disease dataset, UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
(297 patients, 13 clinical features, binary target: presence/absence of heart
disease). Feature descriptions are in `src/column_info.py`.

## 🧠 Methodology

1. **EDA** — target balance, distributions, correlation heatmap, categorical
   breakdowns, outlier check
2. **Feature engineering** — 4 derived clinical features on top of the raw 13
3. **Baseline comparison** — 8 models, identical 5-fold CV, tracked on accuracy,
   precision, recall, F1 and ROC-AUC
4. **Tuning** — `GridSearchCV` on the top 3 models
5. **Final evaluation** — held-out 20% test set, confusion matrix + ROC curve
6. **Interpretability** — feature importances + SHAP summary and waterfall plots
7. **Export** — a single deployable `sklearn.Pipeline`

Full details, code, and all charts are in `notebooks/heart_disease_analysis.ipynb`.

## ⚕️ Disclaimer

This project is for educational and portfolio purposes only. It is **not** a
medical device and must not be used for real clinical diagnosis.

## 🛠️ Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib/seaborn

## 🩹 Troubleshooting

**`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**
(or similar attribute errors when the app loads the model)

This means the scikit-learn version installed on your machine doesn't match
the version used to save `models/heart_disease_pipeline.pkl`. scikit-learn's
internal model format changes between versions, so pickled models aren't
always portable across them.

Fix — install the exact pinned version:
```bash
pip install -r requirements.txt
```
This installs `scikit-learn==1.7.2`, matching the shipped model.

If your Python version can't install that scikit-learn release (e.g. Python
< 3.9), re-run the training notebook to regenerate the pickle with whatever
scikit-learn version you do have:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/heart_disease_analysis.ipynb
```
