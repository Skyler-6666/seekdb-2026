#!/usr/bin/env bash
# Push a resource/build-log snapshot to the hipvs-ci-results branch via the
# contents API so diagnostics survive even if the runner dies mid-step.
# Usage: hb_snapshot.sh <job> <tag> [log-file-in-workspace]
set +e
JOB="$1"
TAG="$2"
LOGFILE="${3:-}"
HB_TMP="$GITHUB_WORKSPACE/.hb-tmp"
mkdir -p "$HB_TMP"
SNAP="$HB_TMP/snap-$$.txt"
RESP="$HB_TMP/resp-$$.txt"
{
  echo "=== heartbeat $(date -u '+%F %T') run=$GITHUB_RUN_NUMBER sha=$GITHUB_SHA job=$JOB tag=$TAG ==="
  echo "--- uptime ---"; uptime
  echo "--- nproc ---"; nproc
  echo "--- free -m ---"; free -m
  echo "--- df ---"; df -h / "$GITHUB_WORKSPACE" 2>/dev/null
  echo "--- top mem ---"; ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12
  echo "--- dmesg tail ---"; dmesg 2>/dev/null | tail -6
  if [ -n "$LOGFILE" ] && [ -f "$GITHUB_WORKSPACE/$LOGFILE" ]; then
    echo "--- log tail: $LOGFILE ---"
    tail -c 30000 "$GITHUB_WORKSPACE/$LOGFILE" 2>/dev/null
  fi
} > "$SNAP" 2>&1
CONTENT=$(base64 -w0 < "$SNAP" 2>/dev/null || base64 < "$SNAP" 2>/dev/null)
if [ -z "$CONTENT" ]; then
  echo "hb $TAG: empty snapshot"
  exit 0
fi
curl -sS -m 20 -X PUT \
  -H "Authorization: Bearer $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "{\"message\":\"ci: heartbeat run $GITHUB_RUN_NUMBER $JOB $TAG\",\"branch\":\"hipvs-ci-results\",\"content\":\"$CONTENT\"}" \
  "https://api.github.com/repos/$GITHUB_REPOSITORY/contents/hipvs-ci-results/$JOB/hb-$(date +%s)-$RANDOM-$TAG.txt" \
  > "$RESP" 2>&1
echo "hb $TAG rc=$? resp=$(head -c 200 "$RESP" | tr '\n' ' ')"
