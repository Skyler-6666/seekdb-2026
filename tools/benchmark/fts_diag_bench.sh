#!/usr/bin/env bash
# TOKENIZE 微基准（诊断用，不属于官方计分基准）：
# 把每次 SELECT tokenize() 的开销按解析器与文本长度拆开，
# 便于定位每轮优化前后的固定开销与解析核心开销。
#
# 用法（需实例已在运行，官方基准跑完后即可复用）:
#   MYSQL="..." ROUNDS=2000 OUTPUT=./fts_diag_report.txt bash tools/benchmark/fts_diag_bench.sh
#
# 环境变量:
#   MYSQL    mysql 客户端命令（默认 -h127.0.0.1 -P2881 -uroot -N -s --default-character-set=utf8mb4）
#   ROUNDS   每组 SQL 轮次（默认 2000）
#   SAMPLES  采样组数（默认 3）
#   WARMUP   预热轮次（默认 30）
#   OUTPUT   结果追加写入的文件（可选）

set -euo pipefail

MYSQL="${MYSQL:-mysql -h127.0.0.1 -P2881 -uroot -N -s --default-character-set=utf8mb4}"
ROUNDS="${ROUNDS:-2000}"
SAMPLES="${SAMPLES:-3}"
WARMUP="${WARMUP:-30}"
OUTPUT="${OUTPUT:-}"

IK_TEXT="OceanBase是一款非常稳定的数据库，全文索引的分词器热路径优化包括解析器实例复用、停用词全局单例、内存池化数据结构等技术，在大量文档索引和查询场景下可以显著降低 CPU 开销。"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fts_diag_bench.XXXXXX")"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

log() { echo "$@" >&2; }

now_ns() { date +%s%N; }

elapsed_ms() {
  local start_ns="$1" end_ns="$2"
  awk "BEGIN {printf \"%.4f\", (${end_ns} - ${start_ns}) / 1000000}"
}

calc_sample_stats() {
  local file="$1"
  awk '
    {
      a[NR] = $1
      sum += $1
      sumsq += $1 * $1
    }
    END {
      if (NR == 0) {
        printf "N/A N/A"
        exit
      }
      mean = sum / NR
      var = sumsq / NR - mean * mean
      if (var < 0) {
        var = 0
      }
      printf "%.4f %.4f", mean, sqrt(var)
    }
  ' "${file}"
}

run_one() {
  local name="$1" sql="$2"
  local warmup_file="${TMP_DIR}/${name}.warmup.sql"
  local sample_file="${TMP_DIR}/${name}.samples"

  : > "${sample_file}"

  if (( WARMUP > 0 )); then
    {
      echo "SET NAMES utf8mb4;"
      echo "SET ob_query_timeout = 100000000000;"
      echo "SET ob_trx_timeout = 100000000000;"
      for ((i = 0; i < WARMUP; i++)); do
        echo "${sql}"
      done
    } > "${warmup_file}"
    $MYSQL < "${warmup_file}" >/dev/null
  fi

  for ((sample = 1; sample <= SAMPLES; sample++)); do
    local sql_file="${TMP_DIR}/${name}.${sample}.sql"
    {
      echo "SET NAMES utf8mb4;"
      echo "SET ob_query_timeout = 100000000000;"
      echo "SET ob_trx_timeout = 100000000000;"
      for ((i = 0; i < ROUNDS; i++)); do
        echo "${sql}"
      done
    } > "${sql_file}"
    local start_ns end_ns total_ms avg_ms
    start_ns=$(now_ns)
    $MYSQL < "${sql_file}" >/dev/null
    end_ns=$(now_ns)
    total_ms=$(elapsed_ms "${start_ns}" "${end_ns}")
    avg_ms=$(awk "BEGIN {printf \"%.4f\", ${total_ms} / ${ROUNDS}}")
    echo "${avg_ms}" >> "${sample_file}"
    log "  ${name}: sample=${sample}/${SAMPLES} rounds=${ROUNDS} avg_ms=${avg_ms}"
  done

  local stats
  stats="$(calc_sample_stats "${sample_file}")"
  report="${report}${name}_avg_ms:$(echo "${stats}" | awk '{print $1}')
${name}_stdev_ms:$(echo "${stats}" | awk '{print $2}')
"
}

report=""
run_one "diag_select1" "SELECT 1;"
run_one "diag_tok_ik_1char" "SELECT tokenize('你', 'ik');"
run_one "diag_tok_ik_2char" "SELECT tokenize('数据库', 'ik');"
run_one "diag_tok_beng_1char" "SELECT tokenize('a', 'beng');"
run_one "diag_tok_space_full" "SELECT tokenize('${IK_TEXT}', 'space');"
run_one "diag_tok_ngram_full" "SELECT tokenize('${IK_TEXT}', 'ngram');"
run_one "diag_tok_ik_full" "SELECT tokenize('${IK_TEXT}', 'ik');"

{
  echo "========================================"
  echo "FTS Tokenize Micro-benchmark (diagnostic)"
  echo "========================================"
  echo "rounds:                 ${ROUNDS}"
  echo "samples:                ${SAMPLES}"
  echo "warmup:                 ${WARMUP}"
  echo "----------------------------------------"
  printf '%s' "${report}"
  echo "----------------------------------------"
  echo "decompose (avg - select1):"
  echo "  ik fixed (1char):      see tok_ik_1char - select1"
  echo "  beng fixed (1char):    see tok_beng_1char - select1"
  echo "  space full:            SQL+map+json with trivial parser core"
  echo "  ngram full:            parser core without dict lookups"
  echo "========================================"
} | tee -a "${OUTPUT:-/dev/null}"
