# Runtime image for FastAPI inference service
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source and artifacts (model files are mounted in compose/k8s)
COPY . /app

EXPOSE 8000
ENV API_KEY="" \
    UVICORN_WORKERS=2 \
    HOST=0.0.0.0 \
    PORT=8000

CMD ["sh", "-c", "uvicorn api.main:app --host ${HOST} --port ${PORT} --workers ${UVICORN_WORKERS}"]
