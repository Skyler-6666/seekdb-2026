FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MINIVECTORDB_DATA_DIR=/data

WORKDIR /app

# NumPy is used only as a vectorized math library for exact L2 scans.
RUN pip install --no-cache-dir numpy==1.26.4

COPY minivectordb /app/minivectordb

RUN mkdir -p /data

VOLUME ["/data"]

ENTRYPOINT ["python3", "-m", "minivectordb.main"]
CMD ["--data-dir", "/data"]
