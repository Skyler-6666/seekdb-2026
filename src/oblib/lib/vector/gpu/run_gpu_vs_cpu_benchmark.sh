#!/usr/bin/env bash
#
# Run the HNSW build benchmark twice, once with the GPU path disabled and
# once with it enabled, and compare the two results.
#
# The script never starts the observer itself, the deployment of seekdb is
# different on every machine. It calls the stop and start commands that are
# given on the command line and exports the environment that the observer
# needs in the matching mode.
#
#   ./run_gpu_vs_cpu_benchmark.sh \
#       --stop-cmd "obd cluster stop seekdb" \
#       --start-cmd "obd cluster start seekdb" \
#       --rows 200000 --dim 128
#
# The GPU mode additionally needs the plugin, either through
# OB_GPU_HNSW_PLUGIN_PATH or through the well known paths that the loader
# probes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STOP_CMD=""
START_CMD=""
ROWS=200000
DIM=128
M=16
EF_CONSTRUCTION=200
QUERIES=100
TOPK=10
DISTANCE="l2"
SEED=42
OUTPUT_DIR="$(pwd)"
PLUGIN_PATH="${OB_GPU_HNSW_PLUGIN_PATH:-}"
HOST="${MYSQL_HOST:-127.0.0.1}"
PORT="${MYSQL_PORT:-2881}"
USER="${MYSQL_USER:-root@test}"
PASSWORD="${MYSQL_PASSWORD:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-cmd) STOP_CMD="$2"; shift 2 ;;
    --start-cmd) START_CMD="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --dim) DIM="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --ef-construction) EF_CONSTRUCTION="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --distance) DISTANCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --plugin) PLUGIN_PATH="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${START_CMD}" ]]; then
  echo "--start-cmd is required, it has to start the observer" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

BENCH_ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --user "${USER}"
  --rows "${ROWS}"
  --dim "${DIM}"
  --m "${M}"
  --ef-construction "${EF_CONSTRUCTION}"
  --queries "${QUERIES}"
  --topk "${TOPK}"
  --distance "${DISTANCE}"
  --seed "${SEED}"
)

if [[ -n "${PASSWORD}" ]]; then
  BENCH_ARGS+=(--password "${PASSWORD}")
fi

wait_for_observer() {
  local attempt=0
  while [[ ${attempt} -lt 120 ]]; do
    if mysql -h "${HOST}" -P "${PORT}" -u "${USER}" ${PASSWORD:+-p${PASSWORD}} \
         -e "select 1" >/dev/null 2>&1; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 2
  done
  echo "the observer did not come up on ${HOST}:${PORT}" >&2
  return 1
}

run_phase() {
  local label="$1"
  shift
  echo "=== phase ${label} ==="
  if [[ -n "${STOP_CMD}" ]]; then
    eval "${STOP_CMD}"
  fi
  export "$@"
  eval "${START_CMD}"
  wait_for_observer
  python3 "${SCRIPT_DIR}/benchmark_hnsw_build.py" "${BENCH_ARGS[@]}" \
    --label "${label}" --output "${OUTPUT_DIR}/result_${label}.json"
}

run_phase cpu OB_GPU_HNSW_BUILD=0

run_phase gpu OB_GPU_HNSW_BUILD=1 OB_GPU_HNSW_PLUGIN_PATH="${PLUGIN_PATH}"

python3 "${SCRIPT_DIR}/compare_benchmark_results.py" \
  "${OUTPUT_DIR}/result_cpu.json" "${OUTPUT_DIR}/result_gpu.json" \
  --output "${OUTPUT_DIR}/comparison.json"
