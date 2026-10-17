#!/usr/bin/env bash
# Poll for gated access to fdtn-ai/antares-350m; when granted, run download_antares.sh
set -u
cd "$(dirname "$0")/.." || exit 1
COOKIE_FILE=".secrets/antares.cookie"
LOG="logs/antares_access.log"
mkdir -p logs
echo "=== watcher started $(date) ===" >> "$LOG"
for i in $(seq 1 2880); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Cookie: $(cat "$COOKIE_FILE")" \
    "https://huggingface.co/fdtn-ai/antares-350m/resolve/main/config.json")
  echo "[$(date +%H:%M:%S)] attempt $i -> HTTP $code" >> "$LOG"
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    echo "=== ACCESS GRANTED at $(date) — starting download ===" >> "$LOG"
    bash scripts/download_antares.sh >> "$LOG" 2>&1
    echo "=== download script finished exit=$? $(date) ===" >> "$LOG"
    exit 0
  fi
  sleep 60
done
echo "=== watcher timed out $(date) ===" >> "$LOG"
