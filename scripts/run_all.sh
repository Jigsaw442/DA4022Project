#!/usr/bin/env bash
# Full experiment pipeline. Resumable: finished jobs are skipped on re-run.
#   GPUS=0,1,2,3 bash scripts/run_all.sh            # full run (~12 GPU-hours on A100)
#   QUICK=1 GPUS=0 bash scripts/run_all.sh          # ~40% size run
#   TRAIN=0 ...                                     # skip the optional TF-HH stage
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0}
MODEL=${MODEL:-llava-hf/llava-1.5-7b-hf}
DATA=${DATA:-data}
OUT=${OUT:-results}
TRAIN=${TRAIN:-1}
EXTRA=${EXTRA:-}                 # extra args for every script, e.g. "--device cpu --dtype float32"
TRAIN_EXTRA=${TRAIN_EXTRA:-}     # extra args for train_tfhh.py
if [ "${QUICK:-0}" = "1" ]; then NCAP=${NCAP:-200}; NPOPE=${NPOPE:-1200}; NHEAD=${NHEAD:-150}; STEPS=${STEPS:-100};
else NCAP=${NCAP:-500}; NPOPE=${NPOPE:-3000}; NHEAD=${NHEAD:-300}; STEPS=${STEPS:-200}; fi
MAXTOK=${MAXTOK:-128}
TOPK=${TOPK:-20}
POPE_FILE=${POPE_FILE:-$DATA/pope/coco_pope_adversarial.json}
ACCUM=${ACCUM:-32}

COMMON="--model $MODEL --coco-root $DATA/coco/val2014 $EXTRA"
CAP="python scripts/caption_eval.py $COMMON --ann-dir $DATA/coco/annotations --n $NCAP --max-new-tokens $MAXTOK"
POPE="python scripts/pope_eval.py $COMMON --pope-file $POPE_FILE --n $NPOPE"
mkdir -p "$OUT/jobs" "$OUT/heads"

# configs: name method budget
CONFIGS="none576:none:576 hybrid144:hybrid:144 hybrid64:hybrid:64 hybrid16:hybrid:16"
BASELINES="pool64:pool:64 topk64:topk:64"

echo "== Stage 1: greedy decoding (baselines + compression levels)"
J=$OUT/jobs/stage1.txt; : > $J
for c in $CONFIGS $BASELINES; do IFS=: read n m b <<< "$c"
  echo "$OUT/caption/${n}_greedy/metrics.json | $CAP --method $m --budget $b --out $OUT/caption/${n}_greedy" >> $J
  echo "$OUT/pope/${n}_greedy/metrics.json | $POPE --method $m --budget $b --out $OUT/pope/${n}_greedy" >> $J
done
python scripts/launch.py $J --gpus $GPUS

echo "== Stage 2: hallucination-head attribution at every compression level"
J=$OUT/jobs/stage2.txt; : > $J
for c in $CONFIGS; do IFS=: read n m b <<< "$c"
  echo "$OUT/heads/$n.json | python scripts/find_heads.py $COMMON --captions $OUT/caption/${n}_greedy/captions.jsonl --method $m --budget $b --n $NHEAD --verify-topk $([ $n = hybrid64 ] && echo 24 || echo 0) --out $OUT/heads/$n.json" >> $J
done
python scripts/launch.py $J --gpus $GPUS

echo "== Stage 3: AD-HH / VR-HH interventions, head transfer, adaptive budget"
J=$OUT/jobs/stage3.txt; : > $J
for c in $CONFIGS; do IFS=: read n m b <<< "$c"
  for d in adhh vrhh; do
    H="--decode $d --heads $OUT/heads/$n.json --topk-heads $TOPK"
    echo "$OUT/caption/${n}_$d/metrics.json | $CAP --method $m --budget $b $H --out $OUT/caption/${n}_$d" >> $J
    echo "$OUT/pope/${n}_$d/metrics.json | $POPE --method $m --budget $b $H --out $OUT/pope/${n}_$d" >> $J
  done
done
# do heads found WITHOUT compression transfer to the compressed model?
echo "$OUT/caption/hybrid64_vrhh-transfer/metrics.json | $CAP --method hybrid --budget 64 --decode vrhh --heads $OUT/heads/none576.json --topk-heads $TOPK --out $OUT/caption/hybrid64_vrhh-transfer" >> $J
# hallucination-risk adaptive budget: 16 tokens by default, 144 for risky inputs
for d in greedy vrhh; do
  H="--adaptive 16,144 --adaptive-heads $OUT/heads/hybrid16.json --heads $OUT/heads/hybrid144.json --topk-heads $TOPK --decode $d"
  echo "$OUT/caption/adaptive16-144_$d/metrics.json | $CAP --method hybrid --budget 16 $H --out $OUT/caption/adaptive16-144_$d" >> $J
  echo "$OUT/pope/adaptive16-144_$d/metrics.json | $POPE --method hybrid --budget 16 $H --out $OUT/pope/adaptive16-144_$d" >> $J
done
python scripts/launch.py $J --gpus $GPUS

if [ "$TRAIN" = "1" ]; then
  echo "== Stage 4 (optional): compression-aware TF-HH fine-tuning at 64 tokens"
  J=$OUT/jobs/stage4.txt; : > $J
  echo "$OUT/tfhh_hybrid64/tfhh_delta.pt | python scripts/train_tfhh.py $COMMON $TRAIN_EXTRA --ann-dir $DATA/coco/annotations --pope-file $POPE_FILE --eval-n $NCAP --heads $OUT/heads/hybrid64.json --topk-heads $TOPK --method hybrid --budget 64 --steps $STEPS --grad-accum $ACCUM --out $OUT/tfhh_hybrid64" >> $J
  python scripts/launch.py $J --gpus ${GPUS%%,*}
  J=$OUT/jobs/stage5.txt; : > $J
  D="--delta $OUT/tfhh_hybrid64/tfhh_delta.pt"
  echo "$OUT/caption/hybrid64_tfhh/metrics.json | $CAP --method hybrid --budget 64 $D --out $OUT/caption/hybrid64_tfhh" >> $J
  echo "$OUT/pope/hybrid64_tfhh/metrics.json | $POPE --method hybrid --budget 64 $D --out $OUT/pope/hybrid64_tfhh" >> $J
  echo "$OUT/caption/hybrid16_tfhh/metrics.json | $CAP --method hybrid --budget 16 $D --out $OUT/caption/hybrid16_tfhh" >> $J
  echo "$OUT/pope/hybrid16_tfhh/metrics.json | $POPE --method hybrid --budget 16 $D --out $OUT/pope/hybrid16_tfhh" >> $J
  python scripts/launch.py $J --gpus $GPUS
fi

echo "== Tables and figures"
python scripts/make_report_assets.py --results $OUT
echo "done: see $OUT/tables and $OUT/figures"
