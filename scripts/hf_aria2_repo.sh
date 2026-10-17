#!/usr/bin/env bash
# hf_aria2_repo.sh REPO LOCALDIR
# تنزيل مستودع HF كاملاً بتجاوز مشكلة تجمّد xet client:
#   - ملفات صغيرة: curl
#   - ملفات كبيرة: aria2c متعدد الاتصالات (16+)
#   - التحقق: sha256 الحقيقي من paths-info API (lfs.oid) — ملاحظة: xetHash ليس sha256!
# التوكن: يُقرأ من .secrets/hf_token (خارج مخزن hf الرسمي كي يعمل الطابور مجهول الهوية)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$1"; OUT="$2"
TOKEN=$(cat "$ROOT/.secrets/hf_token" 2>/dev/null || true)
AUTH="Authorization: Bearer $TOKEN"
TMP="/tmp/opencode"; mkdir -p "$TMP" "$OUT"
BN=$(echo "$REPO" | tr '/' '_')
TREE="$TMP/tree_$BN.json"; LIST="$TMP/list_$BN.tsv"; PJSON="$TMP/paths_$BN.json"; PRES="$TMP/pres_$BN.json"

curl -sf -H "$AUTH" "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" -o "$TREE" \
  || { echo "FATAL: tree fetch failed for $REPO"; exit 1; }

rm -f "$PRES"
python3 - "$TREE" "$PJSON" <<'PY'
import json, sys
d = [x['path'] for x in json.load(open(sys.argv[1])) if x.get('type') == 'file' and (x.get('size') or 0) >= 67108864]
json.dump({"paths": d}, open(sys.argv[2], 'w'))
PY
curl -sf -X POST -H "$AUTH" -H "Content-Type: application/json" --data-binary @"$PJSON" \
  "https://huggingface.co/api/models/$REPO/paths-info/main" -o "$PRES" || echo "WARN: paths-info unavailable (سيكون التحقق بالحجم فقط)"

python3 - "$TREE" "$LIST" "$PRES" <<'PY'
import json, sys, os
tree = json.load(open(sys.argv[1]))
sha_map = {}
if os.path.exists(sys.argv[3]) and os.path.getsize(sys.argv[3]) > 0:
    try:
        for x in json.load(open(sys.argv[3])):
            oid = ((x.get('lfs') or {}).get('oid')) or ''
            if len(oid) == 64:
                sha_map[x['path']] = oid
    except Exception:
        pass
with open(sys.argv[2], 'w') as f:
    for x in tree:
        if x.get('type') != 'file':
            continue
        f.write("%s\t%s\t%s\n" % (x['path'], x.get('size') or 0, sha_map.get(x['path'], '')))
PY

fails=0
while IFS=$'\t' read -r path size sha; do
  [ -n "$path" ] || continue
  sha="${sha:-}"
  dest="$OUT/$path"
  mkdir -p "$(dirname "$dest")"
  if [ "$size" -ge 67108864 ]; then
    ok=0
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" = "$size" ]; then
      if [ ${#sha} -eq 64 ]; then
        echo "HASH $path ..."
        got=$(sha256sum "$dest" | cut -d' ' -f1)
        if [ "$got" = "$sha" ]; then ok=1; else echo "HASH-MISMATCH $path — إعادة تنزيل"; fi
      else
        ok=1; echo "SKIP $path (حجم فقط، لا sha متاح)"
      fi
    fi
    if [ "$ok" = "1" ]; then echo "OK-EXIST $path"; continue; fi
    rm -f "$dest" "$dest.aria2"
    echo "BIG  $path ($(( size / 1048576 )) MB) via aria2c -x24"
    aria2c -c -x 16 -s 16 -k 4M --min-split-size=4M --max-connection-per-server=16 \
      --max-tries=15 --retry-wait=3 --file-allocation=none --auto-file-renaming=false \
      --allow-overwrite=true --console-log-level=warn --summary-interval=15 \
      --header="$AUTH" -d "$(dirname "$dest")" -o "$(basename "$path")" \
      "https://huggingface.co/$REPO/resolve/main/$path" \
      || { echo "FAIL aria2 $path"; fails=$((fails+1)); continue; }
    if [ ${#sha} -eq 64 ]; then
      got=$(sha256sum "$dest" | cut -d' ' -f1)
      if [ "$got" = "$sha" ]; then echo "OK-SHA $path"; else echo "FAIL-SHA $path"; fails=$((fails+1)); fi
    else
      if [ "$(stat -c%s "$dest")" = "$size" ]; then echo "OK $path (size)"; else echo "FAIL-SIZE $path"; fails=$((fails+1)); fi
    fi
  else
    have=0; [ -f "$dest" ] && have=$(stat -c%s "$dest")
    if [ "$size" != "0" ] && [ "$have" = "$size" ]; then echo "SKIP $path"; continue; fi
    echo "SM   $path ($have -> $size)"
    curl -sfL --retry 8 --retry-delay 3 --retry-all-errors -H "$AUTH" -o "$dest" \
      "https://huggingface.co/$REPO/resolve/main/$path" \
      || { echo "FAIL curl $path"; fails=$((fails+1)); continue; }
    have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$size" != "0" ] && [ "$have" != "$size" ]; then echo "FAIL-SIZE $path"; fails=$((fails+1)); else echo "OK   $path"; fi
  fi
done < "$LIST"
echo "=== $REPO: fails=$fails ==="
exit "$fails"
