#!/usr/bin/env bash
# T2b — تنزيل طبقة بيانات الأدوات (R6 في PLAN.md)
# tool use / agentic / terminal / browser / MCP — الإجمالي ≈ 44GB
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

# "repo|local-dir"
REPOS=(
  "Team-ACE/ToolACE|toolace"
  "Nanbeige/ToolMind|toolmind"
  "nvidia/Nemotron-Terminal-Corpus|nemotron-terminal"
  "nvidia/Nemotron-SFT-Agentic-v2|nemotron-agentic"
  "nebius/SWE-rebench-openhands-trajectories|swe-rebench-openhands"
  "internlm/Agent-FLAN|agent-flan"
  "glaiveai/glaive-function-calling-v2|glaive-fc"
  "osunlp/Mind2Web|mind2web"
  "ScaleAI/MCP-Atlas|mcp-atlas"
  "marin-community/mcp-atlas-easy|mcp-atlas-easy"
  "jdaddyalbs/playwright-mcp-toolcalling|playwright-mcp-toolcalling"
)

FAILED=()
for entry in "${REPOS[@]}"; do
  repo="${entry%%|*}"
  dir="${entry##*|}"
  echo "[$(date +%H:%M:%S)] START $repo"
  if "$HF" download "$repo" --repo-type dataset --local-dir "$RAW/$dir"; then
    echo "[$(date +%H:%M:%S)] OK    $repo"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $repo"
    FAILED+=("$repo")
  fi
done

echo "=== TOOL-DATA SUMMARY: ${#FAILED[@]} failed: ${FAILED[*]:-none}"
