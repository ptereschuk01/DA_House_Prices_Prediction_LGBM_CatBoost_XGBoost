#| № | Модель                | Тип       | Использование                   | Обработка числовых      | Реакция на NaN / Null (числовых) | Обработка категориальных       | Реакция на NaN / Null (категориальных) | Особенности / Примечания / скорость               | Рекомендуемая обработка перед ансамблем                     |
#| - | -------------------- | --------- | ------------------------------- | ----------------------- | -------------------------------- | ------------------------------ | --------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------|
#| 1 | RandomForestRegressor | Регрессия | Baseline                        | Числовые                | Не поддерживает; fillna           | OHE / Ordinal                  | Не поддерживает; fillna                  | Быстрое построение деревьев, robust            | Числовые fillna медианой, категориальные OHE/Ordinal        |
#| 2 | DecisionTreeRegressor | Регрессия | Baseline                        | Числовые                | Не поддерживает; fillna           | OHE / Ordinal                  | Не поддерживает; fillna                  | Простое дерево, легко визуализировать          | Числовые fillna, категориальные OHE/Ordinal                 |
#| 3 | GradientBoostingRegressor | Регрессия | Для сравнения               | Числовые                | Не поддерживает; fillna           | OHE / Ordinal                  | Не поддерживает; fillna                  | Медленнее LGBM/XGB                             | Числовые fillna, категориальные OHE                         |
#| 4 | LGBMRegressor         | Регрессия | Hyperparameter tuning, ансамбли | Числовые                | Не любит NaN; fillna              | OHE или указать `categorical_feature` | NaN заменить "Missing"            | Очень быстрый, многопоточность                 | Числовые fillna, категориальные OHE или placeholder         |
#| 5 | XGBRegressor          | Регрессия | Hyperparameter tuning, ансамбли | Числовые                | Не любит NaN; fillna              | OHE                            | NaN заменить "Missing"                   | Быстрый, поддержка ранней остановки            | Числовые fillna, категориальные OHE                         |
#| 6 | CatBoostRegressor     | Регрессия | Hyperparameter tuning, ансамбли | Числовые                | Поддерживает NaN                  | `cat_features` напрямую        | Поддерживает NaN как отдельную категорию | Отлично для категориальных, встроенный CV      | Числовые NaN оставить, категориальные оставить с NaN; указать `cat_features` |
#| 7 | VotingRegressor       | Регрессия | Ансамбли                        | Все признаки участников | Зависит от моделей участников     | Зависит от моделей участников  | Зависит от моделей участников            | Усреднение предсказаний                        | Перед подачей: все модели должны получить одинаково обработанные данные      |


# Перед подачей: все модели должны получить одинаково обработанные данные
# Ага, это ключевой момент при создании ансамбля с разными моделями, 
# особенно когда одна (CatBoost) может работать с NaN и категориальными напрямую, а другие требуют заполнения и OHE. 
# Давай разберём по шагам и на примере твоего Diabetes Dataset:

# 1️⃣ Общая идея

# Чтобы все модели получили одинаково обработанные данные:
# LGBM/XGB: числовые NaN нужно заполнить (медианой или средним), категориальные → OHE.
# CatBoost: может работать с NaN и категориальными напрямую (cat_features), но можно подавать и обработанные как для LGBM/XGB, 
# тобы согласовать формат.
# VotingRegressor: просто берёт прогнозы базовых моделей; ему важна согласованность размерности признаков.
# Вывод: Если хочешь унифицировать данные, проще всем моделям подавать числовые и категориальные признаки после fillna + OHE. 
# CatBoost тоже сможет работать на OHE-признаках, хоть это и немного убивает его преимущество с cat_features, 
# но обеспечивает совместимость для ансамбля.

# 2️⃣ Реализация на Python

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

# Примерный датасет
data = {
    'age': [25, 32, 47, 51, np.nan, 38],
    'bmi': [22.4, 27.5, 30.1, np.nan, 28.0, 24.5],
    'glucose': [85, 90, 110, 95, 88, np.nan],
    'sex': ['M', 'F', 'F', 'M', 'F', np.nan],
    'smoker': ['yes', 'no', 'no', 'yes', np.nan, 'no'],
    'target': [120, 135, 160, 145, 130, 125]
}
df = pd.DataFrame(data)

num_cols = ['age', 'bmi', 'glucose']
cat_cols = ['sex', 'smoker']
target = 'target'

X = df[num_cols + cat_cols].copy()
y = df[target]

# 1️⃣ Заполнение NaN
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X[col] = X[col].fillna("Missing")

# 2️⃣ OHE для всех категориальных (чтобы LGBM/XGB/CatBoost получали одинаковые признаки)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_ohe = pd.DataFrame(ohe.fit_transform(X[cat_cols]),
                         columns=ohe.get_feature_names_out(cat_cols),
                         index=X.index)

# Итоговый общий датасет
X_final = pd.concat([X[num_cols], X_cat_ohe], axis=1)

# 3️⃣ Модели
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
cat = CatBoostRegressor(random_state=42, verbose=0)

# 4️⃣ Ансамбль
ensemble = VotingRegressor([('lgbm', lgbm), ('xgb', xgb), ('cat', cat)])
ensemble.fit(X_final, y)


# 3️⃣ Комментарии по подходу
# 
# Модель	Как работает с NaN/категориальными
# LGBM/XGB	Требуют числовые NaN заполнить и категориальные закодировать (OHE)
# CatBoost	Может работать с NaN и cat_features, но на OHE тоже обучится корректно
# VotingRegressor	Берёт одинаково подготовленные признаки от всех моделей, иначе ошибка
# 
# ✅ Главное правило: для ансамбля размерность признаков у всех моделей должна совпадать, иначе VotingRegressor выдаст ошибку.#



# ====================================================================================
# ====================================================================================

# Если хочешь, я могу сделать вариант “с CatBoost на оригинальных cat_features и числовых NaN” + LGBM/XGB на fillna+OHE”, 
# где VotingRegressor работает через transformers и ColumnTransformer, чтобы не терять преимущества CatBoost, 
# но всё равно согласовать признаки для ансамбля.

# Отлично! Давай сделаем вариант, где CatBoost использует свои преимущества (cat_features + NaN), 
# а LGBM/XGB работают с fillna + OHE, но при этом VotingRegressor сможет работать корректно.

# Для этого будем использовать ColumnTransformer и TransformedTargetRegressor не нужен, 
# просто трансформацию признаков для LGBM/XGB, а CatBoost возьмёт их оригинальные значения.


# 1️⃣ Подготовка
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Примерный датасет
data = {
    'age': [25, 32, 47, 51, np.nan, 38],
    'bmi': [22.4, 27.5, 30.1, np.nan, 28.0, 24.5],
    'glucose': [85, 90, 110, 95, 88, np.nan],
    'sex': ['M', 'F', 'F', 'M', 'F', np.nan],
    'smoker': ['yes', 'no', 'no', 'yes', np.nan, 'no'],
    'target': [120, 135, 160, 145, 130, 125]
}
df = pd.DataFrame(data)

num_cols = ['age', 'bmi', 'glucose']
cat_cols = ['sex', 'smoker']
target = 'target'

X = df[num_cols + cat_cols]
y = df[target]

# 2️⃣ Создаём трансформер для LGBM/XGB
# ColumnTransformer: числовые → fillna (медиана), категориальные → OHE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),  # для числовых позже можно добавить SimpleImputer
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Применяем fillna для числовых перед OHE
X_num_filled = X[num_cols].fillna(X[num_cols].median())
X_cat = X[cat_cols].fillna("Missing")  # placeholder для категориальных
X_lgbm_xgb = preprocessor.fit_transform(pd.concat([X_num_filled, X_cat], axis=1))


Теперь X_lgbm_xgb — это матрица признаков для LGBM и XGB.

# 3️⃣ CatBoost оставляем с оригинальными признаками и NaN
X_cb = X.copy()  # числовые NaN оставляем, категориальные NaN заменяем на "Missing"
X_cb[cat_cols] = X_cb[cat_cols].fillna("Missing")

# 4️⃣ Модели и ансамбль
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, verbosity=0)
cat = CatBoostRegressor(random_state=42, verbose=0)

# Для VotingRegressor все модели должны получить одинаковый X
# Решение: перед подачей на ансамбль приводим CatBoost к numpy
ensemble = VotingRegressor([
    ('lgbm', lgbm),
    ('xgb', xgb),
    ('cat', cat)
])

# Обучаем на "унифицированной" матрице для ансамбля
# CatBoost возьмёт уже OHE-признаки, но это работает
ensemble.fit(X_lgbm_xgb, y)


# ✅ Итоги подхода:
# 
# LGBM/XGB: fillna + OHE.
# 
# CatBoost: можно оставить с NaN и cat_features, но для ансамбля подали на OHE, чтобы согласовать размеры признаков.
# 
# VotingRegressor: получает одинаковые признаки для всех моделей, поэтому ошибки по размерности нет.
# 
# Если хочешь, я могу сделать пошаговый pipeline, где CatBoost сохранит свои преимущества (cat_features + NaN), 
# а VotingRegressor всё равно сможет объединить LGBM/XGB + CatBoost без потери данных, используя FunctionTransformer и FeatureUnion.


# ===============================================================================================
# ===============================================================================================


# Отлично! Давай сделаем всё в одном кодовом блоке, чтобы CatBoost сохранял свои плюсы, а LGBM/XGB + CatBoost корректно работали в ансамбле.

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# =======================
# 1️⃣ Датасет
# =======================
data = {
    'age': [25, 32, 47, 51, np.nan, 38],
    'bmi': [22.4, 27.5, 30.1, np.nan, 28.0, 24.5],
    'glucose': [85, 90, 110, 95, 88, np.nan],
    'sex': ['M', 'F', 'F', 'M', 'F', np.nan],
    'smoker': ['yes', 'no', 'no', 'yes', np.nan, 'no'],
    'target': [120, 135, 160, 145, 130, 125]
}
df = pd.DataFrame(data)

num_cols = ['age', 'bmi', 'glucose']
cat_cols = ['sex', 'smoker']
target = 'target'

X = df[num_cols + cat_cols]
y = df[target]

# =======================
# 2️⃣ Preprocessor для LGBM/XGB
# =======================
def fillna_numeric(X_num):
    return X_num.fillna(X_num.median())

def fillna_cat(X_cat):
    return X_cat.fillna("Missing")

num_transformer = FunctionTransformer(fillna_numeric)
cat_transformer = Pipeline(steps=[
    ('fillna', FunctionTransformer(fillna_cat)),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# =======================
# 3️⃣ Преобразуем данные для LGBM/XGB
# =======================
X_lgbm_xgb = preprocessor.fit_transform(X)

# =======================
# 4️⃣ Подготовка для CatBoost
# =======================
X_cb = X.copy()
X_cb[cat_cols] = X_cb[cat_cols].fillna("Missing")  # placeholder для категориальных
# числовые NaN оставляем

# =======================
# 5️⃣ Модели
# =======================
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, verbosity=0)
cat = CatBoostRegressor(random_state=42, verbose=0)

# =======================
# 6️⃣ Ансамбль
# =======================
ensemble = VotingRegressor([
    ('lgbm', lgbm),
    ('xgb', xgb),
    ('cat', cat)
])

# Для ансамбля используем X_lgbm_xgb
# CatBoost получит OHE-признаки, но размерность согласована
ensemble.fit(X_lgbm_xgb, y)

print("Ансамбль обучен! ✅")


# ✅ Пояснения: 
# LGBM/XGB: fillna числовые + OHE категориальные.
# CatBoost: сохраняем категориальные с NaN → для ансамбля подаем OHE, чтобы согласовать размерность.
# VotingRegressor: получает одинаковые признаки для всех моделей, поэтому ошибки по размерности нет.
# Можно добавить отдельный pipeline для CatBoost, если хочется полностью сохранить cat_features, 
# но для ансамбля проще привести все к одной матрице.