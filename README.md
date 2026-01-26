# DA House Prices Prediction (LGBM, CatBoost, XGBoost)

📊 **A complete ML project for predicting house prices using ensemble gradient boosting models.**

![Header Image](notebooks/images/project_header.png)

---

## 🧠 Project Description

This repository contains a complete machine learning project focused on regression: predicting house prices. The workflow includes:

- **Data Analysis (EDA)** — exploring distributions, correlations, and visual patterns
- **Feature Engineering** — handling missing data, creating new features, encoding categorical variables
- **Model Training** — LightGBM, CatBoost, XGBoost with hyperparameter tuning
- **Evaluation & Visualization** — comparing models and visualizing predictions

Goal: build a high-accuracy model and document the full ML workflow with clear visuals.

---

## 🧾 Repository Structure

```
DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost/
├─ data/raw/                # Raw train/test data
├─ notebooks/               # Jupyter notebooks with EDA, modeling, and visualizations
│   ├─ images/              # Sample plots and figures for README and analysis
├─ requirements.txt         # Python dependencies
├─ README.md                # Project description
└─ .gitignore
```

---

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ptereschuk01/DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost.git
```
2. Navigate to the project folder:
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

## 🛠 How to Use

Open and run notebooks in the **notebooks/** directory:
```bash
jupyter notebook
```

Workflow steps:
1. **EDA** — data exploration, visualization of distributions and correlations
2. **Feature Engineering** — create new features, handle missing values, encode categoricals
3. **Model Training & Tuning** — train LightGBM, CatBoost, XGBoost, optimize hyperparameters
4. **Evaluation & Comparison** — compute RMSE, R², visualize predictions

---

## 📊 Model Comparison

| Model      | RMSE   | R²     | Visualization |
|------------|--------|--------|---------------|
| LightGBM   | 0.123  | 0.85   | ![LightGBM](notebooks/images/lightgbm_plot.png) |
| CatBoost   | 0.119  | 0.87   | ![CatBoost](notebooks/images/catboost_plot.png) |
| XGBoost    | 0.122  | 0.86   | ![XGBoost](notebooks/images/xgboost_plot.png) |

---

## 📈 Sample Plots

**Predicted vs Actual Prices:**

![Predictions](notebooks/images/predictions_example.png)

**Feature Importance:**

![Feature Importance](notebooks/images/feature_importance.png)

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

## ✍️ Author

👤 Pavlo Tereshchuk (ptereschuk01)

Demonstrates Data Science and ML skills for regression tasks.

---

## 📜 License

MIT License *(if applicable)*

---

## ☁️ Google Colab

Run the project in Colab directly from GitHub:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ptereschuk01/DA_House_Prices_Prediction_LGBM_CatBoost_XGBoost/blob/main/notebooks/EDA_and_Modeling.ipynb)
