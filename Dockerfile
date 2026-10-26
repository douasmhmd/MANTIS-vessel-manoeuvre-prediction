# Image unique MANTIS : pipelines Kedro, API FastAPI, interfaces Streamlit.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY conf ./conf
COPY src ./src
COPY api ./api
COPY ui ./ui
RUN pip install -e .

# data/ et mlflow.db sont montés en volume (voir docker-compose.yml)
EXPOSE 8000 8501 8502 5000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]