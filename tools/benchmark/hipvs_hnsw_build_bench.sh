#!/usr/bin/env bash
# Copyright (c) 2025 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
  cat <<'EOF'
Measure CREATE VECTOR INDEX time on an already loaded seekdb table.

Required environment variables:
  DB_NAME        Database containing the benchmark table
  TABLE_NAME     Table containing the vector column
  COLUMN_NAME    Vector column name

Optional environment variables:
  DB_HOST        Default: 127.0.0.1
  DB_PORT        Default: 2881
  DB_USER        Default: root@sys
  DB_PASSWORD     Default: empty; passed through MYSQL_PWD
  MYSQL          Default: mysql
  INDEX_NAME     Default: idx_hipvs_hnsw_bench
  M              Default: 32
  EF_CONSTRUCTION Default: 200
  EF_SEARCH      Default: 200
  DISTANCE       Default: l2
  LABEL          Default: current-server

Run the script once with SEEKDB_HIPVS=0 and once with SEEKDB_HIPVS=1,
restarting seekdb between runs. The output is a single JSON object.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${DB_NAME:?DB_NAME is required}"
: "${TABLE_NAME:?TABLE_NAME is required}"
: "${COLUMN_NAME:?COLUMN_NAME is required}"

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-2881}"
DB_USER="${DB_USER:-root@sys}"
DB_PASSWORD="${DB_PASSWORD:-}"
MYSQL="${MYSQL:-mysql}"
INDEX_NAME="${INDEX_NAME:-idx_hipvs_hnsw_bench}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
EF_SEARCH="${EF_SEARCH:-200}"
DISTANCE="${DISTANCE:-l2}"
LABEL="${LABEL:-current-server}"

mysql_cmd=(
  "${MYSQL}"
  --host="${DB_HOST}"
  --port="${DB_PORT}"
  --user="${DB_USER}"
  --database="${DB_NAME}"
  --batch
  --skip-column-names
)

if [[ -n "${DB_PASSWORD}" ]]; then
  export MYSQL_PWD="${DB_PASSWORD}"
fi

"${mysql_cmd[@]}" -e "DROP INDEX IF EXISTS \`${INDEX_NAME}\` ON \`${TABLE_NAME}\`" >/dev/null 2>&1 || true

start_ns="$(date +%s%N)"
"${mysql_cmd[@]}" -e \
  "CREATE VECTOR INDEX \`${INDEX_NAME}\` ON \`${TABLE_NAME}\`(\`${COLUMN_NAME}\`) WITH (distance=${DISTANCE}, type=hnsw, lib=vsag, m=${M}, ef_construction=${EF_CONSTRUCTION}, ef_search=${EF_SEARCH})"
end_ns="$(date +%s%N)"

"${mysql_cmd[@]}" -e "DROP INDEX \`${INDEX_NAME}\` ON \`${TABLE_NAME}\`" >/dev/null

elapsed_ns="$((end_ns - start_ns))"
printf '{"label":"%s","build_time_ms":%.3f,"m":%s,"ef_construction":%s}\n' \
  "${LABEL}" "$(awk -v ns="${elapsed_ns}" 'BEGIN { printf "%.3f", ns / 1000000 }')" \
  "${M}" "${EF_CONSTRUCTION}"
