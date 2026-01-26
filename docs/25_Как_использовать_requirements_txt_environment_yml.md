🔁 Как использовать requirements.txt

После активации окружения:

conda activate data_ml
pip install -r requirements.txt

💡 Если спросят на собеседовании

“Для воспроизводимости я использую conda-окружение, а зависимости фиксирую в requirements.txt без жёстких версий, чтобы проект легко разворачивался на любой системе.”

Это правильный и безопасный ответ.


🔁 Как использовать environment.yml

▶️ Создание окружения из файла
conda env create -f environment.yml

▶️ Активация
conda activate data_ml





🔁 Когда использовать environment.yml, а когда requirements.txt
Сценарий	Что использовать
Быстро поставить зависимости	requirements.txt
Полная воспроизводимость	environment.yml
Командная работа	environment.yml
Kaggle / Colab	requirements.txt
Портфолио ML	оба ✅

👉 Сильные проекты всегда имеют оба файла

💼 Как это выглядит для рекрутера

“Проект поддерживает два способа развёртывания: через pip (requirements.txt) и через conda (environment.yml) для полной воспроизводимости ML-окружения.”

Это очень хороший сигнал уровня.

📌 Где хранить в проекте
house_prices_mid/
├── environment.yml   👈
├── requirements.txt
├── README.md