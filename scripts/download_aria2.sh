#!/usr/bin/env bash
# download_aria2.sh KIND REPO OUTDIR
# تحميل مستودع HuggingFace (dataset أو model) بأقصى سرعة عبر aria2c:
#   - lista الملفات من HF API
#   - الملفات الصغيرة: aria2c متوازٍ (-j 32)
#   - الملفات الكبيرة (>64MB): تقسيم متعدد الاتصالات (-x 8 -s 8)
#   - استكمال تلقائي (-c) وإعادة تجربة عند 429
#   - التوكن من .secrets/hf_token (لا يُلمس الكاش الرسمي لـ hf CLI لتجنب تجمد xet)
set -u
KIND="$1"; REPO="$2"; OUTDIR="$3"
ROOT="/home/hamzah/Documents/Default Project/vulnre"
TOKEN=$(cat "$ROOT/.secrets/hf_token" 2>/dev/null || true)
AUTH="Authorization: Bearer $TOKEN"
TMP="/tmp/opencode"; mkdir -p "$TMP" "$OUTDIR"
BN="${REPO//\//_}"
LOG="$ROOT/logs/dl_${BN}.log"
mkdir -p "$ROOT/logs"
if [ "${FOREGROUND:-0}" = "1" ]; then
  exec 2>&1
else
  exec >>"$LOG" 2>&1
fi

if [ "$KIND" = "dataset" ]; then
  API="https://huggingface.co/api/datasets/$REPO"
  DL="https://huggingface.co/datasets/$REPO/resolve/main"
else
  API="https://huggingface.co/api/models/$REPO"
  DL="https://huggingface.co/$REPO/resolve/main"
fi

echo "=== $(date) START $KIND $REPO -> $OUTDIR ==="
curl -sf -H "$AUTH" "$API/tree/main?recursive=true" -o "$TMP/tree_$BN.json" \
  || { echo "FATAL: tree fetch failed for $REPO"; exit 1; }

python3 - "$TMP/tree_$BN.json" "$TMP/aria_$BN.txt" "$DL" "$OUTDIR" <<'PY'
import json, os, sys
tree_path, out_path, dl_base, root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(tree_path))
with open(out_path, 'w') as f:
    for x in d:
        if x.get('type') != 'file':
            continue
        p = x['path']; sz = x.get('size') or 0
        dest = os.path.join(root, p)
        if os.path.exists(dest) and os.path.getsize(dest) == sz:
            print(f"SKIP {p} ({sz} B)", file=sys.stderr)
            continue
        f.write(dl_base + "/" + p + "\n")
        f.write("  dir=" + os.path.dirname(dest) + "\n")
        f.write("  out=" + os.path.basename(dest) + "\n")
        f.write("\n")
PY

aria2c --input-file="$TMP/aria_$BN.txt" \
  --header="$AUTH" \
  -j 32 -x 8 -s 8 -k 4M --min-split-size=4M \
  --max-connection-per-server=8 \
  --max-tries=15 --retry-wait=3 \
  --file-allocation=none --auto-file-renaming=false \
  --allow-overwrite=true --continue=true \
  --console-log-level=notice --summary-interval=10 \
  || true

echo "=== $(date) DONE $KIND $REPO ==="
