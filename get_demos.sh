#!/bin/bash
# Step 2: download the 50 clean aloha-agilex demos for each task (~0.45 GB total).
set -e
cd "$(dirname "$0")/RoboTwin"
mkdir -p data/zips
BASE=https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset
for t in lift_pot beat_block_hammer; do
  echo "Downloading $t ..."
  curl -fL --retry 30 --retry-delay 2 --retry-all-errors -C - -o "data/zips/$t.zip" "$BASE/$t/demo_clean.zip"
done
echo "Done."
