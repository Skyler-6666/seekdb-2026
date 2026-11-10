FROM python:3.11-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY minivectordb/fastscan.cpp /build/fastscan.cpp
RUN g++ -std=c++17 -O3 -march=native -fPIC -shared \
    -static-libgcc -static-libstdc++ \
    -o /build/libfastscan.so /build/fastscan.cpp

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VECTORDB_DATA_DIR=/data

WORKDIR /app

# NumPy remains the portable fallback; compiled fastscan is the primary path.
RUN pip install --no-cache-dir numpy==1.26.4

COPY minivectordb /app/minivectordb
COPY --from=build /build/libfastscan.so /app/minivectordb/libfastscan.so

RUN mkdir -p /data

VOLUME ["/data"]

ENTRYPOINT ["python3", "-m", "minivectordb.main"]
CMD ["--data-dir", "/data"]
