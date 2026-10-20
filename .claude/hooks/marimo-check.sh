#!/usr/bin/env bash
# PostToolUse hook: run `marimo check` on a marimo notebook Claude just edited or wrote.
# Exit 2 sends the check's output back to Claude to fix; other files pass silently.
file=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
[[ -f "$file" ]] || exit 0
grep -q "import marimo" "$file" && grep -q "@app.cell" "$file" || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
if ! output=$(uv run --quiet marimo check "$file" 2>&1); then
  echo "marimo check failed for $file:" >&2
  echo "$output" >&2
  exit 2
fi
