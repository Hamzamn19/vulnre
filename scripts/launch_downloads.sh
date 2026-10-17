#!/usr/bin/env bash
# إطلاق منفصل (setsid) لسكربت التنزيل الموحّد — قفل flock يمنع التشغيل المزدوج
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
exec flock -n "$ROOT/logs/download_all.lock" bash "$ROOT/scripts/download_all.sh"
