#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "➡️ Starting To load data ➡️ "

echo "⌛️ Loading CREMA-D..."
# bash "$DIR/load-crema.sh"
python3 "$DIR/load-crema.py"
echo "✅ Loaded CREMA-D..."

echo "⌛️ Loading EmoGator..."
if [ -x "$DIR/load-emo.sh" ]; then
  python3 "$DIR/load-crema.py"
else
  echo "Skipping EmoGator (script missing)..."
fi

echo "🏁 Finished Loading Data 🏁"
