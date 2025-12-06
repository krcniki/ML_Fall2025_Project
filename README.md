# ML_Fall2025_Project

## 📌 Project Overview
This project is an end-to-end machine learning pipeline designed to predict if a diabetic patient will be readmitted to the hospital within **30 days** of discharge.

The primary challenge of this dataset is severe **Class Imbalance** (only 11.1% of patients are readmitted). Standard models suffer from the **"Accuracy Paradox,"** achieving 91% accuracy by simply ignoring all readmitted patients.

**Goal:** This project solves that paradox by trading "Vanity Accuracy" for "Clinical Utility" (Recall). I implemented a custom feature engineering pipeline, Synthetic Minority Over-sampling (SMOTE), and F1-score optimization to build a model that actually identifies at-risk patients.

## 🏗 Architecture
This project uses a modular production-style structure designed for reproducibility:

```text
├── data/                   # Raw dataset (diabetic_data.csv)
├── notebooks/              # Analysis & Reporting
│   ├── 01_EDA.ipynb        # Data justification (Missingness, Grouping)
│   └── 02_Modeling.ipynb   # Audit: SHAP Plots, Confusion Matrices
├── src/                    # Source code modules
│   ├── preprocess.py       # Data cleaning & Pipeline construction
│   ├── feature_engineering.py  # ICD-9 diagnosis grouping logic
│   └── evaluate.py         # Metrics logging & Plotting functions
├── train.py                # Main pipeline: Loads data, applies SMOTE, trains models
├── tune.py                 # Optimization: RandomizedSearch for Hyperparameters
├── experiments.csv         # Automated log of all model runs
└── requirements.txt        # Dependencies
```

## Methodology
1. Data Cleaning (src/preprocess.py)
- Dropped columns with excessive missingness (e.g., weight > 96%, payer_code) to reduce noise.
- Removed administrative IDs to prevent overfitting.
- Converted the target variable to Binary: 1 if readmitted <30 days, 0 otherwise.

2. Feature Engineering (src/feature_engineering.py)
- ICD-9 Grouping: The dataset contained 700+ unique diagnosis codes. I mapped these into 9 clinical categories (e.g., 250.xx -> "Diabetes", 390-459 -> "Circulatory") to reduce dimensionality and improve model stability.

3. Handling Imbalance
- Used ImbPipeline to apply SMOTE (Synthetic Minority Over-sampling Technique) only on the training folds during Cross-Validation. This prevents data leakage and ensures the model learns to distinguish minority class examples.

4. Interpretability
- Verified the model using SHAP (SHapley Additive exPlanations), finding that Prior Inpatient Visits was the strongest predictor of readmission risk.

## Key Results
I optimized for **Recall (Sensitivity)** to minimize False Negatives (missing a sick patient).

**Insight:** While the Random Forest had higher accuracy, the Logistic Regression was the only model to achieve clinically useful Recall (~53%), proving that simpler models often generalize better on noisy medical data.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run optimization: `python tune.py` (Saves best params to json)
3. Train models: `python train.py` (Loads json and logs results)

