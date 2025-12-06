# Lightweight container for the Flask web app
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default password can be overridden via environment variable
ENV APP_PASSWORD=MITcoding

EXPOSE 8000

CMD ["flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "8000"]
