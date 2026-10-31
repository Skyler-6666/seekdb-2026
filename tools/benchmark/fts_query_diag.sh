#!/usr/bin/env bash
# FTS 查询/构建诊断（不属于官方计分基准）。
# 目标：回答三个问题
#   1) LIMIT 是否真的下推/早停（同一谓词下 LIMIT 1/20/200/2000/无 LIMIT 的耗时曲线）
#   2) COUNT(*) MATCH 的单文档成本（结合 hits 计算 ns/doc）
#   3) 索引构建的冷/热差异（同列重建一次对比，区分字典加载与写入）
# 同时对查询阶段做 gdb 采样，输出热点栈，便于定位自研代码里的开销。
#
# 用法（实例已运行、基准数据已灌好）:
#   MYSQL="..." OUTPUT=./fts_diag_report.txt bash tools/benchmark/fts_query_diag.sh

set -uo pipefail

MYSQL="${MYSQL:-mysql -h127.0.0.1 -P2881 -uroot -N -s --default-character-set=utf8mb4}"
MYSQL_VERBOSE="${MYSQL_VERBOSE:-mysql -h127.0.0.1 -P2881 -uroot --default-character-set=utf8mb4}"
OUTPUT="${OUTPUT:-/dev/null}"
ROUNDS="${ROUNDS:-30}"
SAMPLES="${SAMPLES:-3}"
SAMPLE_CNT="${SAMPLE_CNT:-20}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fts_query_diag.XXXXXX")"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

log() { echo "$@" >&2; }
now_ns() { date +%s%N; }
ms() { awk "BEGIN {printf \"%.4f\", ($2 - $1) / 1000000}"; }

REPORT=""

emit() { REPORT="${REPORT}$1: $2
"; }

sql_once() {
  $MYSQL -e "SET NAMES utf8mb4; SET ob_query_timeout = 100000000000; $1" 2>/dev/null
}

explain_sql() {
  local name="$1" sql="$2"
  {
    echo "### EXPLAIN ${name}"
    echo "-- ${sql}"
    $MYSQL_VERBOSE -e "SET NAMES utf8mb4; EXPLAIN ${sql}" 2>&1
    echo
  } >> "${TMP_DIR}/explain.txt"
}

# 计时：SAMPLES 组 × ROUNDS 次，取最小均值，减少抖动影响
time_sql() {
  local name="$1" sql="$2"
  local reps_file="${TMP_DIR}/${name}.sql"
  {
    echo "SET NAMES utf8mb4;"
    echo "SET ob_query_timeout = 100000000000;"
    echo "SET ob_trx_timeout = 100000000000;"
    for ((i = 0; i < ROUNDS; i++)); do echo "${sql}"; done
  } > "${reps_file}"

  local best="" s total
  # warmup
  $MYSQL < "${reps_file}" >/dev/null 2>&1
  for ((s = 1; s <= SAMPLES; s++)); do
    local t0 t1
    t0=$(now_ns)
    $MYSQL < "${reps_file}" >/dev/null 2>&1
    t1=$(now_ns)
    total=$(ms "${t0}" "${t1}")
    local avg
    avg=$(awk "BEGIN {printf \"%.4f\", ${total} / ${ROUNDS}}")
    log "  ${name}: sample=${s} avg_ms=${avg}"
    if [[ -z "${best}" ]] || awk "BEGIN {exit !(${avg} < ${best})}"; then best="${avg}"; fi
  done
  emit "${name}_avg_ms" "${best}"
}

declare -A HITS
hits_sql() {
  local name="$1" sql="$2"
  local cnt
  cnt=$(sql_once "${sql}" | tail -1)
  : "${cnt:=0}"
  HITS["${name}"]="${cnt}"
  emit "${name}_hits" "${cnt}"
}

query_phase() {
  log "[1/3] query diagnostics ..."

  L20="SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(content) AGAINST('倒排索引 tokenizer')"
  SUB20="SELECT COUNT(*) FROM (SELECT id FROM fts_large_bench.docs WHERE MATCH(content) AGAINST('倒排索引 tokenizer') LIMIT 20) t"

  hits_sql "d_q_content_no_limit" "${L20}"
  hits_sql "d_q_sub_limit20" "${SUB20}"

  explain_sql "q_content_no_limit" "${L20}"
  explain_sql "q_sub_limit20" "${SUB20}"
  explain_sql "q_cn_official" "SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title, content) AGAINST('全文索引 分词器')"

  time_sql "d_q_sub_limit1" "SELECT COUNT(*) FROM (SELECT id FROM fts_large_bench.docs WHERE MATCH(content) AGAINST('倒排索引 tokenizer') LIMIT 1) t"
  time_sql "d_q_sub_limit20" "${SUB20}"
  time_sql "d_q_sub_limit200" "SELECT COUNT(*) FROM (SELECT id FROM fts_large_bench.docs WHERE MATCH(content) AGAINST('倒排索引 tokenizer') LIMIT 200) t"
  time_sql "d_q_sub_limit2000" "SELECT COUNT(*) FROM (SELECT id FROM fts_large_bench.docs WHERE MATCH(content) AGAINST('倒排索引 tokenizer') LIMIT 2000) t"
  time_sql "d_q_content_no_limit" "${L20}"
  time_sql "d_q_cn_official" "SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title, content) AGAINST('全文索引 分词器')"
  time_sql "d_q_beng_official" "SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title_en, content_en) AGAINST('database performance audit')"
}

build_phase() {
  log "[2/3] build cold/warm diagnostics ..."
  local t0 t1
  t0=$(now_ns)
  sql_once "USE fts_large_bench; ALTER TABLE docs ADD FULLTEXT INDEX fti_diag_ik(content) WITH PARSER ik;" >/dev/null
  t1=$(now_ns)
  emit "d_build_ik_content_first_sec" "$(awk "BEGIN {printf \"%.3f\", ($(ms "${t0}" "${t1}")) / 1000}")"

  sql_once "USE fts_large_bench; ALTER TABLE docs DROP INDEX fti_diag_ik;" >/dev/null
  t0=$(now_ns)
  sql_once "USE fts_large_bench; ALTER TABLE docs ADD FULLTEXT INDEX fti_diag_ik(content) WITH PARSER ik;" >/dev/null
  t1=$(now_ns)
  emit "d_build_ik_content_second_sec" "$(awk "BEGIN {printf \"%.3f\", ($(ms "${t0}" "${t1}")) / 1000}")"
}

sample_phase() {
  log "[3/3] gdb sampling during query loop ..."
  local pid
  pid=$(pgrep -f "observer/seekdb" | head -1 || true)
  if [[ -z "${pid}" ]]; then
    emit "d_gdb_samples" "no_pid"
    return
  fi
  local loop_file="${TMP_DIR}/loop.sql"
  {
    echo "SET NAMES utf8mb4;"
    for ((i = 0; i < 20000; i++)); do
      echo "SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title, content) AGAINST('全文索引 分词器');"
      echo "SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title_en, content_en) AGAINST('database performance audit');"
    done
  } > "${loop_file}"

  ( $MYSQL < "${loop_file}" >/dev/null 2>&1 & echo $! > "${TMP_DIR}/loop.pid" )
  sleep 1
  local raw="${TMP_DIR}/bt.txt"
  : > "${raw}"
  for ((i = 1; i <= SAMPLE_CNT; i++)); do
    gdb -p "${pid}" -batch -ex "set pagination off" -ex "thread apply all bt 12" >> "${raw}" 2>/dev/null
    sleep 0.15
  done
  kill "$(cat "${TMP_DIR}/loop.pid")" 2>/dev/null || true

  emit "d_gdb_samples" "${SAMPLE_CNT}"
  {
    echo "### gdb sampled stacks (frames inside src/)"
    grep -E '^#[0-9]+ ' "${raw}" \
      | sed -E 's/^#[0-9]+ +//; s/\(.*//; s/0x[0-9a-f]+ in //; s/ \+ [0-9]+$//; s/[[:space:]]+$//' \
      | grep -E 'oceanbase::' \
      | sort | uniq -c | sort -rn | head -60
    echo
  } >> "${TMP_DIR}/explain.txt"
}

query_phase
build_phase
sample_phase

{
  echo "========================================"
  echo "FTS Query/Build Diagnostics"
  echo "========================================"
  echo "rounds:                 ${ROUNDS}"
  echo "samples:                ${SAMPLES}"
  echo "----------------------------------------"
  printf '%s' "${REPORT}"
  echo "----------------------------------------"
  echo "hits (total matched docs per predicate):"
  for k in "${!HITS[@]}"; do echo "  ${k}: ${HITS[$k]}"; done
  echo "----------------------------------------"
  echo "ns/doc = (avg_ms - select1_avg_ms) * 1e6 / hits"
  echo "========================================"
  echo
  cat "${TMP_DIR}/explain.txt"
} | tee -a "${OUTPUT}"
