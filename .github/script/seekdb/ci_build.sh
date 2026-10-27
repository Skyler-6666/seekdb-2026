#!/usr/bin/env bash
# Remote compile check for the file-sql branch.
#
# Phases: dependency init -> CMake release configure -> per-file compile check
# of the sources touched by the file-sql feature -> publish logs.
#
# The branch only supports Unity builds, so instead of building the whole
# observer binary (which does not fit on a 4-core/14GB runner) every changed
# translation unit is compiled with the exact flags CMake generated for its
# target (-Wall -Wextra -Werror included).
#
# Outputs (in the workspace root): build.log, status.txt, errors.txt
# Publication: those files are force-pushed to ${CI_LOG_BRANCH} (ci/build-log)
# so the result can be fetched with a plain `git fetch`.
#
# Env:
#   CI_PUSH_TOKEN      token used to publish logs (skipped when empty)
#   GITHUB_REPOSITORY  owner/repo
#   GITHUB_SHA, GITHUB_REF_NAME, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_EVENT_NAME
#   BUILD_DIR          CMake build directory (default build_release)
#   CI_LOG_BRANCH      log branch (default ci/build-log)
#   CHECK_FILES        extra/override cpp files to check (space separated)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR" || exit 2

BUILD_DIR="${BUILD_DIR:-build_release}"
LOG_BRANCH="${CI_LOG_BRANCH:-ci/build-log}"
START_EPOCH="$(date +%s)"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() { echo "[ci] $*"; }

publish_log() {
  local token="${CI_PUSH_TOKEN:-}"
  if [[ -z "$token" || -z "${GITHUB_REPOSITORY:-}" ]]; then
    log "no CI_PUSH_TOKEN/GITHUB_REPOSITORY: skip publishing logs"
    return 0
  fi
  local dir="${RUNNER_TEMP:-/tmp}/ci-log-repo"
  rm -rf "$dir"
  git init -q "$dir" || return 1
  git -C "$dir" config user.name "github-actions[bot]"
  git -C "$dir" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  local f
  for f in build.log status.txt errors.txt; do
    [[ -f "$f" ]] && cp -f "$f" "$dir"/
  done
  git -C "$dir" add -A || return 1
  git -C "$dir" commit -q -m "ci: log for ${GITHUB_SHA:-unknown} run ${GITHUB_RUN_ID:-0}.${GITHUB_RUN_ATTEMPT:-0}" || true
  local url="https://x-access-token:${token}@github.com/${GITHUB_REPOSITORY}.git"
  local attempt
  for attempt in 1 2 3; do
    if git -C "$dir" push --force "$url" "HEAD:refs/heads/${LOG_BRANCH}"; then
      log "published build.log/status.txt/errors.txt to ${LOG_BRANCH}"
      return 0
    fi
    log "log push attempt ${attempt} failed"
    sleep 10
  done
  return 1
}

write_status() {
  local rc="$1"
  local status="PASS"
  if [[ "$rc" -ne 0 ]]; then
    status="FAIL"
  fi
  local end_epoch duration last_phase failed_files summary
  end_epoch="$(date +%s)"
  duration=$((end_epoch - START_EPOCH))
  last_phase="$(grep -o '=== phase: .*' build.log 2>/dev/null | tail -1 | sed 's/=== phase: //')"
  failed_files="$(grep -E '=== (CHECK|UNITY)-FAILED:' build.log 2>/dev/null | sed -E 's/=== (CHECK|UNITY)-FAILED: //' | paste -sd, -)"
  summary="$(grep -E '=== (CHECK|UNITY)-SUMMARY:' build.log 2>/dev/null | tr '\n' ' ')"
  {
    echo "STATUS=${status}"
    echo "EXIT_CODE=${rc}"
    echo "LAST_PHASE=${last_phase:-unknown}"
    echo "FAILED_FILES=${failed_files:-none}"
    echo "CHECK_SUMMARY=${summary:-none}"
    echo "SHA=${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null)}"
    echo "REPO=${GITHUB_REPOSITORY:-unknown}"
    echo "BRANCH=${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null)}"
    echo "RUN_ID=${GITHUB_RUN_ID:-0}"
    echo "RUN_ATTEMPT=${GITHUB_RUN_ATTEMPT:-0}"
    echo "EVENT=${GITHUB_EVENT_NAME:-local}"
    echo "STARTED_UTC=${START_UTC}"
    echo "FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "DURATION_SEC=${duration}"
    echo "LOG_BRANCH=${LOG_BRANCH}"
  } > status.txt
  cat status.txt
  grep -E 'error:|fatal error:|=== CHECK-FAILED:|=== CHECK-NO-TARGET:|=== UNITY-FAILED:|FAILED:' build.log > errors.txt 2>/dev/null || true
  if [[ -f errors.txt ]] && (( $(wc -l < errors.txt) > 300 )); then
    head -300 errors.txt > errors.tmp && mv errors.tmp errors.txt
  fi
  if [[ -f build.log ]] && (( $(stat -c%s build.log) > 2000000 )); then
    tail -c 2000000 build.log > build.log.tmp && mv build.log.tmp build.log
  fi
  ls -la build.log status.txt errors.txt || true
}

check_args=()
if [[ -n "${CHECK_FILES:-}" ]]; then
  # shellcheck disable=SC2206
  check_args=(--only ${CHECK_FILES})
fi

log "repo=${GITHUB_REPOSITORY:-local} sha=${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null)} branch=${GITHUB_REF_NAME:-local}"
export PATH="$REPO_DIR/deps/3rd/usr/local/oceanbase/devtools/bin:$PATH"

set -o pipefail
(
  set -e
  echo "=== ci: repo=${GITHUB_REPOSITORY:-local} branch=${GITHUB_REF_NAME:-local} sha=${GITHUB_SHA:-unknown} run=${GITHUB_RUN_ID:-0}.${GITHUB_RUN_ATTEMPT:-0}"
  echo "=== phase: init"
  ./build.sh init
  echo "=== phase: configure"
  ./build.sh release -DNEED_PARSER_CACHE=OFF
  echo "=== phase: compile-check"
  python3 .github/script/seekdb/build_check.py --build-dir "$BUILD_DIR" --repo "$REPO_DIR" "${check_args[@]}"
  echo "=== phase: unity-check"
  python3 .github/script/seekdb/build_check.py --build-dir "$BUILD_DIR" --repo "$REPO_DIR" --unity
  echo "=== phase: done"
) 2>&1 | tee build.log
rc=$?

write_status "$rc"
publish_log || true

exit "$rc"
