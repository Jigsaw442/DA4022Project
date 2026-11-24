#!/usr/bin/env bash
# Downloads COCO val2014 images + annotations and the POPE question files (~7 GB).
set -euo pipefail
ROOT=${1:-data}
mkdir -p "$ROOT/coco" "$ROOT/pope"
cd "$ROOT/coco"
if [ ! -d val2014 ]; then
  wget -c http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip && rm val2014.zip
fi
if [ ! -f annotations/instances_val2014.json ]; then
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip -q annotations_trainval2014.zip && rm annotations_trainval2014.zip
fi
cd ../pope
for s in adversarial popular random; do
  [ -f coco_pope_$s.json ] || wget -q https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_$s.json
done
echo "data ready in $ROOT"
