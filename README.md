# DA House Prices Prediction (LGBM, CatBoost, XGBoost)

 [A project for predicting house prices using ensemble gradient boosting models (LightGBM, CatBoost, XGBoost).](https://github.com/ptereschuk01/DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost/blob/main/notebooks/House_Prices_Prediction_LGBM_CatBoost_XGBoost.ipynb)

---

##  Description

This repository contains a complete ML project for solving a regression problem: predicting house prices. The project includes:

- Data preparation and analysis (EDA)
- Handling missing values and creating new features
- Training and comparing multiple models (LightGBM, CatBoost, XGBoost)
- Visualizing results and evaluating model performance

The goal is to build a high-accuracy prediction model and document the entire ML workflow.

---

## Contents

-  **data/raw/** — raw data (train/test)  
-  **notebooks/** — Jupyter notebooks with analysis, model training, and visualizations  
-  **requirements.txt** — Python project dependencies  
-  **.gitignore** — Git ignore file


├─ data/

│ ├─ raw/         # Original train/test datasets

│ └─ processed/   # Cleaned and feature-engineered datasets

├─ notebooks/     # Jupyter notebooks with analysis and modeling

├─ requirements.txt # Python dependencies

├─ README.md # Project description

└─ .gitignore
```
---

## 🚀 Installation

1. Clone the repository:  
```bash
git clone https://github.com/ptereschuk01/DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost.git
```
2. Go to the project folder:  
```bash
cd DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost
```
3. Create a virtual environment (recommended):  
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
4. Install dependencies:  
```bash
pip install -r requirements.txt
```

---

##  Usage

Open and run notebooks in the **notebooks/** directory:
```bash
jupyter notebook
```

Workflow steps:

1. **EDA** — data exploration, visualization of distributions and correlations
2. **Feature Engineering** — create new features, handle missing values, encode categoricals
3. **Model Training & Tuning** — train LightGBM, CatBoost, XGBoost, tune hyperparameters
4. **Evaluation & Comparison** — compute RMSE, R², visualize predictions

---

##  Model Comparison

| Model      | RMSE   | R²     |
|------------|--------|--------|
| LightGBM   | 0.123  | 0.85   |
| CatBoost   | 0.119  | 0.87   |
| XGBoost    | 0.122  | 0.86   |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
lightgbm
catboost
xgboost
matplotlib
seaborn
jupyter
```

*(Exact versions listed in `requirements.txt`)*

---

## 🏷 Technologies

- Python 3.x  
- Jupyter Notebook  
- Scikit-Learn, LightGBM, CatBoost, XGBoost  
- Pandas, NumPy  
- Matplotlib & Seaborn

---
