# House Prices Prediction (Mid-Level Project)

**Проект:** Предсказание цен на недвижимость  
**Цель:** Разработка модели, которая точно прогнозирует цену продажи домов на основе доступных признаков.  
**Технологии:** Python, Pandas, NumPy, Scikit-learn, CatBoost, Matplotlib, Seaborn  
**Уровень:** Intermediate (подходит для портфолио аналитика данных / ML-инженера)

---

## 📂 Структура проекта

house_prices_mid/
│
├── data/
│ ├── raw/ # Исходные данные
│ ├── processed/ # Обработанные данные
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ ├── 02_features.ipynb # Feature Engineering
│ ├── 03_modeling.ipynb # Моделирование и подбор гиперпараметров
│
├── src/
│ ├── preprocessing.py # Функции обработки данных
│ ├── train.py # Скрипт обучения модели
│
├── submissions/
│ ├── submission_v1.csv # Финальные предсказания
│
├── README.md
└── requirements.txt

---

## 📝 Описание данных

**Датасет включает:**
- Числовые признаки (площадь, количество комнат, этаж и др.)
- Категориальные признаки (район, тип дома, состояние)
- Дата продажи и другие дополнительные признаки

**Целевая переменная:** `SalePrice` — цена дома в долларах/евро.

---

## 🔍 Этапы проекта

1. **EDA (01_eda.ipynb)**  
   - Анализ распределений признаков  
   - Поиск выбросов и аномалий  
   - Проверка корреляций  
   - Анализ пропусков  

2. **Feature Engineering (02_features.ipynb)**  
   - Создание новых признаков (например, возраст дома, соотношение комнат)  
   - Кодирование категориальных признаков (Ordinal, Target Encoding)  
   - Нормализация и лог-трансформации при необходимости  

3. **Моделирование (03_modeling.ipynb)**  
   - Разделение данных на train/validation  
   - Обучение моделей: Linear Regression, Random Forest, CatBoost  
   - Подбор гиперпараметров (GridSearch / RandomSearch / CV)  
   - Оценка метрик: RMSE, MAE, R²  

4. **Production-ready скрипты (src/)**  
   - `preprocessing.py` — функции обработки данных  
   - `train.py` — основной скрипт для обучения модели и сохранения результатов  

---

## 📊 Результаты

- **Лучшая модель:** CatBoost Regressor  
- **Метрика RMSE на валидации:** 0.12 (пример, указать реальные цифры)  
- **Submission:** `submissions/submission_v1.csv`  

💡 *Каждая версия submission отслеживается, чтобы видеть улучшения.*

---

## 🚀 Как запустить проект

1. Установить зависимости:
```bash
pip install -r requirements.txt

2. Подготовить данные:
Поместить исходные CSV в data/raw/

3. Обработка данных и обучение модели:
python src/train.py

4. Генерация submission:
Результат появится в submissions/submission_v1.csv

📈 Визуализация и анализ
Корреляционные матрицы и тепловые карты
Распределение цен по районам
Влияние признаков на целевую переменную
Feature importance от CatBoost

⚡ Ценность проекта
Показывает весь цикл ML-проекта: EDA → Feature Engineering → Modeling → Submission
Использует продвинутые методы кодирования категориальных признаков
Production-ready структура: ноутбуки для экспериментов, скрипты для воспроизводимости
Готово для использования в портфолио аналитика данных или ML-инженера

📚 Источники данных
Kaggle: House Prices: Advanced Regression Techniques
Дополнительно: синтетические или приватные данные для тренировки моделей

👨‍💻 Автор
Павел Терещук — Аналитик данных / Junior ML
Email: your.email@example.com
GitHub: github.com/yourusername
LinkedIn: linkedin.com/in/yourprofile



