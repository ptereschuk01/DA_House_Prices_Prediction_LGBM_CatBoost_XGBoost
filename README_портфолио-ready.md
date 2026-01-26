# House Prices Prediction (Mid-Level Project)

**Проект:** Предсказание цен на недвижимость  
**Цель:** Разработка модели, которая точно прогнозирует цену продажи домов на основе доступных признаков.  
**Технологии:** Python, Pandas, NumPy, Scikit-learn, CatBoost, Matplotlib, Seaborn  
**Уровень:** Intermediate (подходит для портфолио аналитика данных / ML-инженера)

---

## 📂 Структура проекта

house_prices_mid/  
├── data/  
│   ├── raw/ — Исходные данные  
│   ├── processed/ — Обработанные данные  
├── notebooks/  
│   ├── 01_eda.ipynb — Exploratory Data Analysis  
│   ├── 02_features.ipynb — Feature Engineering  
│   ├── 03_modeling.ipynb — Моделирование и подбор гиперпараметров  
├── src/  
│   ├── preprocessing.py — Функции обработки данных  
│   ├── train.py — Скрипт обучения модели  
├── submissions/  
│   ├── submission_v1.csv — Финальные предсказания  
├── images/ — Примеры графиков и визуализаций  
├── README.md  
└── requirements.txt

---

## 📝 Описание данных

- **Числовые признаки:** площадь, количество комнат, этаж и др.  
- **Категориальные признаки:** район, тип дома, состояние  
- **Дата продажи и дополнительные признаки**  

**Целевая переменная:** `SalePrice` — цена дома.

---

## 🔍 Этапы проекта

### 1️⃣ EDA (notebooks/01_eda.ipynb)

- Анализ распределений признаков  
- Проверка корреляций и мультиколлинеарности  
- Поиск выбросов и аномалий  
- Анализ пропусков  

**Примеры графиков:**  
![Distribution of SalePrice](images/saleprice_distribution.png)  
![Correlation Heatmap](images/correlation_heatmap.png)

### 2️⃣ Feature Engineering (notebooks/02_features.ipynb)

- Создание новых признаков: возраст дома, отношение жилой площади к площади участка  
- Кодирование категориальных признаков: Ordinal, Target Encoding  
- Лог-трансформация признаков для снижения skewness  

**Примеры:**  
![Feature Importance Before Modeling](images/feature_importance_before.png)

### 3️⃣ Моделирование (notebooks/03_modeling.ipynb)

- Разделение данных на train/validation  
- Обучение моделей: Linear Regression, Random Forest, CatBoost  
- Подбор гиперпараметров через CV  
- Оценка метрик: RMSE, MAE, R²  

**Результаты:**  
| Модель        | RMSE (CV) | R² (CV) |
|---------------|-----------|---------|
| LinearReg     | 0.210     | 0.82    |
| RandomForest  | 0.145     | 0.91    |
| CatBoost      | 0.123     | 0.93    |

**Feature Importance CatBoost:**  
![CatBoost Feature Importance](images/catboost_feature_importance.png)

---

## 🚀 Production-ready скрипты

- `src/preprocessing.py` — функции для очистки данных, кодирования и feature engineering  
- `src/train.py` — основной скрипт, который обучает модель, сохраняет модель и генерирует submission  

**Запуск:**  
pip install -r requirements.txt
python src/train.py



---

## 📊 Submission

- Финальный результат: `submissions/submission_v1.csv`  
- Формат: `Id`, `SalePrice`  
- Каждая версия submission отслеживается для сравнения улучшений.

---

## 💡 Выводы

- CatBoost показал лучшую производительность на всех метриках  
- Важнейшие признаки: OverallQual, GrLivArea, YearBuilt, Neighborhood  
- Feature Engineering и правильное кодирование категориальных признаков значительно улучшили RMSE  

**Ключевая ценность проекта:**  
- Полный цикл ML-проекта: EDA → Feature Engineering → Modeling → Submission  
- Production-ready структура  
- Подходит для демонстрации навыков аналитика/ML-инженера

---

## 📚 Источники данных

- Kaggle: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 👨‍💻 Автор

**Павел Терещук** — Аналитик данных / Junior ML  
- Email: your.email@example.com  
- GitHub: [github.com/yourusername](https://github.com/yourusername)  
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)






=====================================================
ZUSAZLICHE
=====================================================


## ♻️ Reproducibility & Environment

Проект настроен с акцентом на **полную воспроизводимость результатов** и может быть развёрнут двумя способами — через `conda` или `pip`.

### 🟢 Вариант 1. Полное воспроизводимое окружение (рекомендуется)

Используется файл `environment.yml`, который фиксирует:

* версию Python
* conda-пакеты
* pip-зависимости (CatBoost, LightGBM)

Создание окружения:

```bash
conda env create -f environment.yml
```

Активация:

```bash
conda activate data_ml
```

Этот способ рекомендуется для:

* ML-проектов
* командной работы
* воспроизводимых экспериментов

---

### 🟡 Вариант 2. Быстрая установка зависимостей

Используется файл `requirements.txt`.

После активации любого Python-окружения:

```bash
pip install -r requirements.txt
```

Подходит для:

* быстрого запуска
* Kaggle / Colab
* ознакомления с проектом

---

### 🔁 Повторный запуск проекта

Окружение `data_ml` сохраняется после создания и не требует повторной установки.

Достаточно выполнить:

```bash
conda activate data_ml
code .
```

---

### ✅ Примечания

* Окружение не зависит от операционной системы
* Все эксперименты воспроизводимы
* `environment.yml` и `requirements.txt` полностью синхронизированы
* Окружение удаляется только вручную:

```bash
conda remove -n data_ml --all
```
