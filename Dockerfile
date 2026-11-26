FROM python:3.13-slim AS api
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/main.py .
ARG MODEL_VERSION=v1.0
ARG BROKEN_INFER=0
ARG BROKEN_HEALTH=0
ENV MODEL_VERSION=${MODEL_VERSION} BROKEN_INFER=${BROKEN_INFER} BROKEN_HEALTH=${BROKEN_HEALTH}
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM api AS ml
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY app/titanic.py .
COPY model.pkl .
ENV MODEL_VERSION=v3.0
