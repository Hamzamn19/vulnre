#!/usr/bin/env bash
# ============================================================
# vulnre — download Antares-350M/1B (gated repos) via session cookie.
# NOTE: TEMPORARY method: the `hf` CLI cannot use browser cookies.
# Preferred (once we have an official hf_ read token):
#   hf auth login --token hf_xxx
#   hf download fdtn-ai/antares-350m --local-dir models/antares-350m
#   hf download fdtn-ai/antares-1b   --local-dir models/antares-1b
# This script: lists files via API (cookie), downloads each via
# resolve URL with resume + retries, verifies sizes, writes DONE marker.
# ============================================================
set -u
cd "$(dirname "$0")/.." || exit 1

COOKIE_FILE=".secrets/antares.cookie"
LOG="logs/dl-antares.log"
TMPDIR_="/tmp/opencode"

if [ ! -s "$COOKIE_FILE" ]; then
  echo "missing cookie file: $COOKIE_FILE" >&2
  exit 1
fi
COOKIE=$(cat "$COOKIE_FILE")

mkdir -p logs "$TMPDIR_"
: > "$LOG"
exec >>"$LOG" 2>&1
echo "=== Antares download (cookie method) started: $(date) ==="

make_list() {  # $1=tree_json $2=out_tsv
  ANTA_TREE="$1" ANTA_OUT="$2" python3 -c '
import json, os, urllib.parse
data = json.load(open(os.environ["ANTA_TREE"]))
out = open(os.environ["ANTA_OUT"], "w")
for x in data:
    if x.get("type") == "file":
        p = x["path"]; sz = x.get("size") or 0
        out.write("\t".join([urllib.parse.quote(p), p, str(sz)]) + "\n")
out.close()
'
}

download_repo() {  # $1=repo $2=local_dir
  local repo="$1" out="$2" rbase tree list fails=0
  rbase=$(basename "$repo")
  tree="$TMPDIR_/antares_tree_$rbase.json"
  list="$TMPDIR_/antares_files_$rbase.tsv"
  mkdir -p "$out"
  if ! curl -sf -H "Cookie: $COOKIE" \
       "https://huggingface.co/api/models/$repo/tree/main?recursive=true" -o "$tree"; then
    echo "FAIL: cannot fetch file list for $repo"
    return 9
  fi
  make_list "$tree" "$list"
  local nfiles; nfiles=$(wc -l < "$list")
  echo "-- $repo: $nfiles files to process"
  while IFS=$'\t' read -r enc rel sz; do
    local dest have
    dest="$out/$rel"
    mkdir -p "$(dirname "$dest")"
    have=0
    [ -f "$dest" ] && have=$(stat -c%s "$dest")
    if [ -n "$sz" ] && [ "$sz" != "0" ] && [ "$have" = "$sz" ]; then
      echo "SKIP $rel (complete)"
      continue
    fi
    echo "[$(date +%H:%M:%S)] GET $rel  (have=${have}B / want=${sz}B)"
    if [ "$have" -gt 0 ]; then
      curl -sL -C - --retry 8 --retry-delay 5 --retry-all-errors -H "Cookie: $COOKIE" \
        "https://huggingface.co/$repo/resolve/main/$enc" -o "$dest"
    else
      curl -sL --retry 8 --retry-delay 5 --retry-all-errors -H "Cookie: $COOKIE" \
        "https://huggingface.co/$repo/resolve/main/$enc" -o "$dest"
    fi
    have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ -n "$sz" ] && [ "$sz" != "0" ] && [ "$have" != "$sz" ]; then
      echo "FAIL $rel (got ${have}B, want ${sz}B)"
      fails=$((fails+1))
    else
      echo "OK   $rel ($((have/1000000)) MB)"
    fi
  done < "$list"
  return "$fails"
}

download_repo "fdtn-ai/antares-350m" "models/antares-350m"; f1=$?
download_repo "fdtn-ai/antares-1b"   "models/antares-1b";   f2=$?

echo "=== finished $(date): fails_350m=$f1 fails_1b=$f2 ==="
du -sh models/antares-350m models/antares-1b 2>/dev/null
if [ "$f1" = "0" ] && [ "$f2" = "0" ]; then
  touch logs/antares_dl.DONE
  echo "ANTARES_DONE"
fi
