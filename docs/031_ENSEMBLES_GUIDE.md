# Ensemble Methods in Machine Learning
## Silver+ Ensemble (CatBoost + LightGBM)

---

## 0️⃣ Скрипт обучения и ансамбля

```python
# =========================================
# 05_silver_plus_ensemble.py
# Silver+ Ensemble (CatBoost + LightGBM)
# =========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping

# CONFIG
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = "diagnosed_diabetes"

# LOAD DATA
train = pd.read_csv("diabetes_prediction/data/train.csv")
test = pd.read_csv("diabetes_prediction/data/test.csv")

features = [c for c in train.columns if c != TARGET]

cat_features = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

# 🔥 Fix for LightGBM (object dtype not allowed)
for col in cat_features:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

cat_features_idx = [features.index(c) for c in cat_features]

X = train[features]
y = train[TARGET]
X_test = test[features]

# STORAGE
oof_cat = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
test_cat = np.zeros(len(test))
test_lgb = np.zeros(len(test))

# CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# TRAIN
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=200
    )
    
    cat_model.fit(
        X_train,
        y_train,
        cat_features=cat_features_idx,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )
    
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_features_idx,
        callbacks=[early_stopping(100)]
    )
    
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

# EVALUATION
auc_cat = roc_auc_score(y, oof_cat)
auc_lgb = roc_auc_score(y, oof_lgb)
print("\n===== OOF AUC =====")
print(f"CatBoost : {auc_cat:.6f}")
print(f"LightGBM : {auc_lgb:.6f}")

# ENSEMBLE (SOFT VOTING)
oof_ensemble = 0.5 * oof_cat + 0.5 * oof_lgb
test_ensemble = 0.5 * test_cat + 0.5 * test_lgb
auc_ensemble = roc_auc_score(y, oof_ensemble)
print(f"\n🔥 Ensemble AUC : {auc_ensemble:.6f}")

# SAVE SUBMISSION
submission = pd.DataFrame({
    "id": test["id"],
    TARGET: test_ensemble
})
submission.to_csv("submission_silver_plus.csv", index=False)
print("\n✅ Saved: submission_silver_plus.csv")
```

---

## 1️⃣ Voting (Голосование / Усреднение)

**Идея:** Несколько моделей обучаются **независимо** и делают предсказания. Итоговое предсказание формируется через:
- **Soft Voting (усреднение вероятностей)**: Берём среднее вероятностей предсказания каждого класса от всех моделей.
- **Hard Voting (голосование классов)**: Каждый классификатор “голосует” за свой класс, выбирается наиболее часто встречающийся.

**Пример:**
```python
# soft voting
oof_ensemble = 0.5 * oof_cat + 0.5 * oof_lgb
test_ensemble = 0.5 * test_cat + 0.5 * test_lgb
```

**Плюсы:**
- Простая реализация.
- Хорошо работает, если модели разные по ошибкам.
- Устойчивость к одной плохой модели.

**Минусы:**
- Не учитывает качество каждой модели явно.
- Плохие модели могут снизить качество ансамбля.

---

## 2️⃣ Stacking (Стекинг)

**Идея:** Используем несколько базовых моделей (level-0), которые делают предсказания на валидации. На основе этих предсказаний обучаем **метамодель** (level-1), которая учится комбинировать исходные модели.

**Пример:**
```python
X_val_preds = np.column_stack([pred_model1, pred_model2, pred_model3])
meta_model.fit(X_val_preds, y_val)
```

**Плюсы:**
- Метамодель учитывает качество каждой базовой модели.
- Может улучшать результат даже если базовые модели сильно различаются.

**Минусы:**
- Сложнее в реализации.
- Нужно аккуратно избегать утечки данных.

---

## 3️⃣ Когда модель не стоит добавлять в ансамбль

### 3.1 Проверка через корреляцию ошибок
Ансамбль эффективен, когда модели ошибаются в разных местах. Если две модели часто ошибаются на одних и тех же примерах, их объединение мало что даст.

**Пример:**
```python
errors_model1 = (pred_model1.round() != y_val).astype(int)
errors_model2 = (pred_model2.round() != y_val).astype(int)
corr = np.corrcoef(errors_model1, errors_model2)[0,1]
print("Correlation of errors:", corr)
```
- `corr ~ 1` → ошибки совпадают → модель почти не добавляет ценность.
- `corr ~ 0` → ошибки разные → модель полезна для ансамбля.

### 3.2 Проверка через разницу AUC на валидации
Модель должна быть хорошей сама по себе. Если AUC модели близок к случайному (≈0.5), добавление её в ансамбль может снизить результат.

**Пример:**
```python
from sklearn.metrics import roc_auc_score
auc_model = roc_auc_score(y_val, pred_model)
print("AUC модели на валидации:", auc_model)
```
- `AUC < 0.55` → почти случайная модель, не стоит добавлять.
- `AUC ≈ лучшей модели` → можно пробовать, особенно если ошибки разные.

### 3.3 Практическая комбинация
Перед добавлением модели в ансамбль делаем две проверки:
- AUC на валидации — модель должна быть хотя бы чуть лучше случайной.
- Корреляция ошибок с другими моделями — должна быть низкой, чтобы модель добавляла новые правильные предсказания там, где другие ошибаются.

---

## 4️⃣ ASCII-схема полезности моделей
```
               Ensemble
       +--------------------+
       |    Meta-Model      |
       +--------------------+
         ^          ^
   +-----+-----+  +-----+-----+
   | Model A   |  | Model B   |
   | (Strong,  |  | (Strong,  |
   | uncorrel.)|  | uncorrel.)|
   +-----------+  +-----------+
         ^              ^
   +-----+-----+  +-----+-----+
   | Model C   |  | Model D   |
   | (Weak,    |  | (Weak,    |
   | correlated)| | uncorrel.)|
   +-----------+  +-----------+
```
**Объяснение схемы:**
- Модели с сильным AUC и низкой корреляцией ошибок полезны для ансамбля → включаем.
- Модели слабые или с высокой корреляцией ошибок → мало ценности → можно исключить.
- Стрелки показывают, какие модели передают предсказания метамодели (stacking).

---

## 🔹 Итоговая рекомендация
- **Soft Voting** — простое и надёжное решение для небольшого ансамбля.
- **Stacking** — более мощное, когда есть несколько сильных моделей с разными ошибками.
- **Не добавляйте модели**, которые:
  - Имеют низкий AUC на валидации.
  - Ошибаются почти в тех же местах, что и другие модели.

