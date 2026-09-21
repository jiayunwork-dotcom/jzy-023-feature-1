FROM python:3.12-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY hmm_service ./hmm_service
COPY run.py ./

# 非 root 运行；模型档 SQLite 落在 /data（可挂卷持久化）。
RUN adduser -D hmm && mkdir -p /data && chown hmm:hmm /data
USER hmm

ENV HMM_DB_PATH=/data/hmm_models.db \
    PORT=8000

# 对外只暴露 HTTP。
EXPOSE 8000

CMD ["python", "run.py"]
