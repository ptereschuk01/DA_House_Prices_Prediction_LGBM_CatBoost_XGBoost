
#| № | Модель                    | Тип           | Использование                   | Обработка числовых      | Реакция на NaN / Null (числовых) | Обработка категориальных                    | Реакция на NaN / Null (категориальных)    | Особенности / Примечания / скорость                         | Рекомендуемая обработка перед ансамблем                                                    |
#| - | ------------------------- | ------------- | ------------------------------- | ----------------------- | -------------------------------- | ------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
#| 1 | RandomForestRegressor     | Регрессия     | Baseline                        | Числовые                | Не поддерживает; нужно fillna    | Только числовые через кодирование           | Не поддерживает; нужно fillna             | Быстрое построение деревьев, robust, средняя скорость       | Заполнить NaN средним/медианой, категориальные OHE или Ordinal                             |
#| 2 | DecisionTreeRegressor     | Регрессия     | Baseline                        | Числовые                | Не поддерживает; нужно fillna    | Только числовые через кодирование           | Не поддерживает; нужно fillna             | Простое дерево, легко визуализировать, средняя скорость     | Заполнить NaN, категориальные OHE/Ordinal                                                  |
#| 3 | GradientBoostingRegressor | Регрессия     | Для сравнения                   | Числовые                | Не поддерживает; нужно fillna    | Только числовые через кодирование           | Не поддерживает; нужно fillna             | Медленнее LGBM/XGB, стабильный, средняя скорость            | Заполнить NaN, категориальные OHE                                                          |
#| 4 | LGBMRegressor             | Регрессия     | Hyperparameter tuning, ансамбли | Числовые                | Не любит NaN; нужно fillna       | Можно указать `categorical_feature` или OHE | NaN в категориальных заменить placeholder | Очень быстрый, многопоточность                              | Заполнить числовые NaN, категориальные: OHE или placeholder; указать `categorical_feature` |
#| 5 | XGBRegressor              | Регрессия     | Hyperparameter tuning, ансамбли | Числовые                | Не любит NaN; нужно fillna       | Требует OHE                                 | NaN заменять placeholder                  | Быстрый, поддержка ранней остановки                         | Заполнить NaN, категориальные OHE                                                          |
#| 6 | CatBoostRegressor         | Регрессия     | Hyperparameter tuning, ансамбли | Числовые                | Поддерживает NaN                 | `cat_features` напрямую                     | Поддерживает NaN как отдельную категорию  | Отлично для категориальных, встроенный CV, средняя скорость | Числовые NaN можно оставить, категориальные оставить с NaN; указать `cat_features`         |
#| 7 | LogisticRegression        | Классификация | В ансамблях                     | Числовые                | Не поддерживает; нужно fillna    | Только через OHE                            | Не поддерживает; нужно fillna             | Используется как слабый классификатор, быстро               | Заполнить числовые NaN, категориальные OHE                                                 |
#| 8 | VotingRegressor           | Регрессия     | Ансамбли                        | Все признаки участников | Зависит от моделей участников    | Зависит от моделей участников               | Зависит от моделей участников             | Усреднение предсказаний, средняя скорость                   | Перед подачей: все модели должны получить одинаково обработанные данные                    |
#| 9 | VotingClassifier          | Классификация | Ансамбли                        | Все признаки участников | Зависит от моделей участников    | Зависит от моделей участников               | Зависит от моделей участников             | Hard / soft voting, средняя скорость                        | Перед подачей: согласовать обработку NaN и категорий для всех моделей                      |

#❓ Если модели требуют разные данные — зачем готовить заранее?
#Ключевая идея (запомни это)

#Мы НЕ готовим “одни данные для всех моделей”.
#Мы готовим БАЗОВОЕ состояние данных + адаптеры под модели.

#🧱 Что такое «базово подготовленные данные»?

#Это данные, которые:

#✅ имеют одинаковую структуру train / test
#✅ одинаковые типы колонок
#✅ целевой признак отделён
#✅ пропуски осознанно оставлены или помечены
#✅ нет утечек
#
#👉 Это ты уже сделал в ЭТАПАХ 1–2.
#
#Это фундамент. Без него любая модель = мусор.
#
#🧩 Адаптеры под модели (САМОЕ ВАЖНОЕ)
#
#Мы делаем НЕ одну подготовку, а несколько представлений данных:
#
#train_raw   → CatBoost
#train_enc   → Linear / LGBM / XGB
#train_log   → Linear / Ridge


#📌 Исходные данные одни, но view разный.


import pandas as pd  # Работа с таблицами
import numpy as np  # Работа с NaN и массивами
from sklearn.preprocessing import OneHotEncoder  # Для OHE категориальных признаков
from sklearn.model_selection import train_test_split  # Для разбиения данных (не используется сейчас)

# Примерный датасет
# Числовые признаки: age, bmi, glucose
# Категориальные признаки: sex, smoker
data = {
    'age': [25, 32, 47, 51, np.nan, 38],  # числовой, есть NaN
    'bmi': [22.4, 27.5, 30.1, np.nan, 28.0, 24.5],  # числовой, есть NaN
    'glucose': [85, 90, 110, 95, 88, np.nan],  # числовой, есть NaN
    'sex': ['M', 'F', 'F', 'M', 'F', np.nan],  # категориальный, есть NaN
    'smoker': ['yes', 'no', 'no', 'yes', np.nan, 'no'],  # категориальный, есть NaN
    'target': [120, 135, 160, 145, 130, 125]  # целевая переменная
}

df = pd.DataFrame(data)  # Создаем DataFrame из словаря

# Определяем числовые и категориальные признаки
num_cols = ['age', 'bmi', 'glucose']  # числовые признаки
cat_cols = ['sex', 'smoker']  # категориальные признаки
target = 'target'  # целевая переменная

# ===================================
# ===== Для CatBoost =====
# ===================================

X_cb = df[num_cols + cat_cols].copy()  # создаем копию признаков для CatBoost
y = df[target]  # целевая переменная

# Заполняем NaN в категориальных только для CatBoost ("Missing" тоже подходит)
for col in cat_cols:
    X_cb[col] = X_cb[col].fillna("Missing")  # CatBoost не любит NaN в категориальных, заменяем на "Missing"

# Числовые NaN оставляем как есть, CatBoost умеет обрабатывать их автоматически
# => CatBoost сам вычисляет, куда вставить значения для разделения узлов при NaN

print("X_cb для CatBoost:")  # проверяем набор данных для CatBoost
print(X_cb)

# ===================================
# ===== Для LGBM и XGBoost =====
# ===================================

X_model = df[num_cols + cat_cols].copy()  # создаем отдельную копию признаков

# Заполняем числовые NaN медианой (LGBM/XGBoost не умеют работать с NaN напрямую)
for col in num_cols:
    X_model[col] = X_model[col].fillna(X_model[col].median())  # медиана подходит для числовых

# Заполняем категориальные NaN как "Missing"
for col in cat_cols:
    X_model[col] = X_model[col].fillna("Missing")  # чтобы OHE корректно сработал

# One-Hot-Encoding для категориальных признаков
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # sparse=False чтобы получить DataFrame
X_cat_ohe = pd.DataFrame(
    ohe.fit_transform(X_model[cat_cols]),  # кодируем категориальные признаки
    columns=ohe.get_feature_names_out(cat_cols),  # создаем имена новых столбцов
    index=X_model.index  # сохраняем индексы
)

# Итоговый датасет для LGBM/XGBoost
X_model_final = pd.concat([X_model[num_cols], X_cat_ohe], axis=1)  # объединяем числовые и закодированные категориальные признаки

# Комментарии по реакции на NaN для LGBM/XGBoost:
#  - числовые NaN обработаны медианой
#  - категориальные NaN заменены на "Missing"
#  - после OHE все категориальные стали числовыми, без NaN

print("\nX_model_final для LGBM/XGBoost:")  # проверяем итоговую таблицу для LGBM/XGBoost
print(X_model_final)









#==========================================




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