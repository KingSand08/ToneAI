#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "➡️ Starting To load data ➡️ "

echo "⌛️ Loading CREMA-D..."
python3 "$DIR/load-crema.py"
echo "✅ Loaded CREMA-D..."

echo "⌛️ Loading EmoGator..."
python3 "$DIR/load-emogator.py"
echo "✅ Loaded EmoGator..."

echo "⌛️ Combining Data Files..."
python3 "$DIR/combine_data.py"
echo "✅ Data Files Combined..."

echo "🏁 Finished Loading Data 🏁"
