#!/bin/bash
echo "➡️ Starting To load data ➡️ "

echo "⌛️ Loading CREMA-D..."
./load-crema.sh
echo "✅ Loaded CREMA-D..."

echo "⌛️ Loading EmoGator..."
./load-emo.sh
echo "✅ Loaded EmoGator..."

echo "🏁 Finished Loading Data 🏁"
