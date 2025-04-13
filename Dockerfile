FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE $PORT
CMD ["python", "slack_bot.py"]
