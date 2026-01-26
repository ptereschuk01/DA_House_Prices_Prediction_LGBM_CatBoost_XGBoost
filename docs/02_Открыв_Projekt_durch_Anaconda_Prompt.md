# Открыв_Projekt_durch_Anaconda_Prompt

1️⃣ Открываем Anaconda Prompt

2️⃣ Создаём окружение 
conda create -n data_ml python=3.11 -y 
После окончания установки должно появиться сообщение вроде: done 

3️⃣ Активируем окружение 
conda activate data_ml 
В терминале должно появиться: (data_ml) C:\Users\UserS2025> ✅ Это значит, что окружение активно. 

4️⃣ Устанавливаем нужные библиотеки 
conda install -y numpy pandas matplotlib seaborn scikit-learn jupyter notebook pip install catboost lightgbm

5️⃣ Проверяем установку 
python -c "import pandas, numpy, matplotlib, seaborn, catboost, lightgbm; print('OK')" 
Если вывод OK — всё готово. 

6️⃣ Открываем VS Code через data_ml code . 
В VS Code → Python: Select Interpreter → выбираем: Python 3.11 ('data_ml': conda) 
В Jupyter Kernel → тоже выбираем Python (data_ml) Теперь всё готово для Kaggle-проекта.




## ✅ Сохраняется ли окружение `data_ml`?

**Да.**
Если ты один раз создал окружение:

```bash
conda create -n data_ml python=3.11 -y
```

и установил туда библиотеки —
➡️ **оно сохраняется навсегда, пока ты сам его не удалишь.**

Компьютер можно:

* перезагружать
* выключать
* закрывать Anaconda / VS Code

**Ничего не пропадёт.**

---

## 🔁 Как запускать ВТОРОЙ и все следующие разы

### 1️⃣ Открываешь Anaconda Prompt

(или обычный терминал, если `conda` прописана в `PATH`)

### 2️⃣ Активируешь окружение

```bash
conda activate data_ml
```

Если видишь:

```text
(data_ml) C:\Users\...
```

✅ окружение активно
✅ все библиотеки доступны
✅ Python = **3.11**

---

### 3️⃣ Заходишь в папку проекта (если нужно)

```bash
cd путь\к\house_prices_mid
```

### 4️⃣ Запускаешь VS Code

```bash
code .
```

---

### 5️⃣ Проверяешь интерпретатор (обычно один раз)

В VS Code:

* `Python: Select Interpreter`
* выбираешь:
  **Python 3.11 (`data_ml`: conda)**

После этого VS Code **запомнит выбор для проекта**.

---

### 6️⃣ Jupyter / ноутбуки

В ноутбуке:

* `Kernel → Select Kernel`
* выбираешь:
  **Python (`data_ml`)**

Тоже обычно делается **один раз**.

---

## 🧠 Как это выглядит в реальной жизни (коротко)

Каждый день / каждый раз:

```bash
conda activate data_ml
code .
```

**ВСЁ.**
Никаких переустановок.
Никаких `pip install` каждый раз.

---

## 🔍 Проверка, что всё ок (если сомневаешься)

```bash
python -c "import pandas, catboost; print('OK')"
```

---

## ❌ Когда окружение может пропасть

**Только если ты сам удалишь его:**

```bash
conda remove -n data_ml --all
```

Пока этого не сделал — **оно живёт**.

---

## 💡 Рекомендация (как у мидлов)

**1 окружение = 1 тип проектов**

`data_ml` — идеально для:

* Kaggle
* EDA
* ML
* ноутбуков

Если потом понадобится:

* `nlp_env`
* `cv_env`
* `prod_env`

— создашь отдельные.
