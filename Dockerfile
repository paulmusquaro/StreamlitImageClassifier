# Використовуємо офіційну версію Python як базовий образ
FROM python:3.9-slim

# Встановлюємо змінні середовища для правильного виконання Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо всі файли проекту в контейнер
COPY . .

# Копіюємо моделі в контейнер (припускаємо, що моделі зберігаються в директорії "models")
COPY ./models /app/models

# Встановлюємо залежності для TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо Python-залежності з requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт 8501 для Streamlit
EXPOSE 8501

# CMD: Запускаємо Streamlit і відкриваємо браузер
CMD ["python", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

