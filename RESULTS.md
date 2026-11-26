# Results: QUICK run (26 Nov 2026)

LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`), fp16, 1x H100 NVL.
`QUICK=1 bash scripts/run_all.sh`: 200 COCO val2014 captions (CHAIR, max 128 new tokens), 1,200 POPE-adversarial
questions, 150 images for head attribution, 100 TF-HH steps (grad-accum 32), top-20 heads.
Everything is in [`results_quick/`](results_quick): tables (`tables/*.md`, `*.tex`), figures (`figures/*.png`, `*.pdf`),
per-config metrics, captions and attention stats, head scores, the TF-HH training log and the run logs.

## Main table

CHAIR_s / CHAIR_i lower is better. Len = average caption length in words.

| Config | Vis. tok | CHAIR_s ↓ | CHAIR_i ↓ | Recall ↑ | Len | POPE Acc ↑ | POPE F1 ↑ | Yes % |
|---|---|---|---|---|---|---|---|---|
| none576_greedy | 576 | 53.0 | 27.4 | 78.9 | 82.0 | 84.6 | 83.5 | 43.2 |
| none576_adhh | 576 | 42.5 | 23.4 | 77.0 | 87.8 | 84.7 | 83.6 | 43.1 |
| none576_vrhh | 576 | 41.5 | 23.9 | 76.9 | 86.9 | 84.6 | 83.5 | 43.2 |
| hybrid144_greedy | 144 | 52.5 | 27.5 | 78.4 | 80.8 | 82.8 | 81.2 | 41.3 |
| hybrid144_adhh | 144 | 41.5 | 22.2 | 76.7 | 85.3 | 82.8 | 81.2 | 41.3 |
| hybrid144_vrhh | 144 | 45.0 | 25.6 | 73.5 | 83.9 | 82.8 | 81.3 | 41.2 |
| hybrid64_greedy | 64 | 45.0 | 26.0 | 72.9 | 79.2 | 80.2 | 77.6 | 38.2 |
| hybrid64_adhh | 64 | 40.5 | 20.7 | 72.0 | 82.8 | 80.2 | 77.7 | 38.3 |
| hybrid64_vrhh | 64 | 27.5 | 16.8 | 61.2 | 61.0 | 80.2 | 77.6 | 38.2 |
| hybrid64_vrhh-transfer | 64 | 37.5 | 21.1 | 68.2 | 84.1 | – | – | – |
| hybrid64_tfhh | 64 | 36.5 | 22.9 | 69.2 | 77.8 | 77.7 | 73.0 | 32.2 |
| hybrid16_greedy | 16 | 45.5 | 29.0 | 57.6 | 76.6 | 71.6 | 63.1 | 26.7 |
| hybrid16_adhh | 16 | 41.0 | 29.3 | 53.4 | 77.8 | 71.6 | 63.1 | 26.7 |
| hybrid16_vrhh | 16 | 49.5 | 32.3 | 54.4 | 74.3 | 71.7 | 63.3 | 26.8 |
| hybrid16_tfhh | 16 | 31.5 | 22.5 | 54.2 | 62.6 | 69.8 | 58.7 | 22.6 |
| adaptive16-144_greedy | 54 (avg) | 48.0 | 27.9 | 64.7 | 77.7 | 75.2 | 69.9 | 31.8 |
| adaptive16-144_vrhh | 54 (avg) | 42.5 | 28.3 | 56.7 | 80.4 | 75.1 | 69.4 | 31.0 |

## Compressors at 64 visual tokens (greedy)

| Config | CHAIR_s ↓ | CHAIR_i ↓ | Recall ↑ | Len | POPE Acc ↑ | POPE F1 ↑ |
|---|---|---|---|---|---|---|
| none576 (reference) | 53.0 | 27.4 | 78.9 | 82.0 | 84.6 | 83.5 |
| pool64 | 53.5 | 31.4 | 70.9 | 79.8 | 77.8 | 74.1 |
| topk64 | 51.0 | 28.7 | 70.0 | 79.9 | 77.6 | 73.5 |
| hybrid64 | 45.0 | 26.0 | 72.9 | 79.2 | 80.2 | 77.6 |

## Decode latency

Measured separately (30 captions per config, one job alone on the GPU). Use this table for speed, not the
`s/caption` column of `results_quick/tables/main_results.md`. In that column, three rows (none576, hybrid144 and hybrid64
greedy) were timed while 3 jobs shared the GPU under the old, slower attention-recording code.

| Visual tokens | s/caption |
|---|---|
| 576 | 1.33 |
| 144 | 1.23 |
| 64 | 1.19 |
| 16 | 1.17 |

## Observations

- Baseline: uncompressed LLaVA-1.5-7B gives POPE-adversarial 84.6 and CHAIR_s 53.0.
- Hybrid compression is the strongest compressor at 64 tokens on both CHAIR and POPE.
- TF-HH at 64 tokens: CHAIR_s 45.0 → 36.5 with almost unchanged caption length (79.2 → 77.8) and a 2.5-point POPE cost.
  During training, the mean log text-attention of the selected heads went from −0.63 to −10.0 while the LM loss stayed flat
  (2.02 → 2.09; `results_quick/tfhh_hybrid64/train_log.json`, `figures/tfhh_training.png`).
- VR-HH at 64 tokens gives the lowest CHAIR (27.5 / 16.8), but the captions are much shorter (61 vs 79 words) and
  recall drops (61.2 vs 72.9). Part of the CHAIR gain comes from the shorter captions.
- VR-HH hurts at 16 tokens (49.5 vs 45.5 greedy). With very few visual tokens, there is little useful visual evidence to
  move the attention mass onto.
- Head transfer: heads found on the uncompressed model still help at 64 tokens (37.5), with longer captions than
  the compression-matched heads.
- The adaptive 16/144 budget (54 tokens on average) does not beat a fixed 64-token budget.
- AD-HH and VR-HH leave POPE essentially unchanged, because POPE is a single yes/no token.
- Head (layer 17, head 0) is in the top-20 list at every compression level.
- Attribution check (`figures/attribution_check.png`, `heads/hybrid64.json` → `verify`): over the 24 checked heads at 64
  tokens, gradient×activation correlates poorly with exact zero-ablation (Pearson r = −0.19). The disagreement comes
  from 5 layer-0 heads, whose exact effect is negative. Without them, r = 0.48 and 18 of 19 heads have a positive exact
  effect. Layer-0 heads should probably be excluded from the ranking.

## Caveats

- QUICK size: with 200 captions, CHAIR_s has about ±3.5 points of standard error, so differences under about 7 points are
  weak evidence.
- Decode speed is overhead-bound (batch 1, eager attention), so fewer visual tokens give only about 12% faster
  captioning. The savings from compression are mainly in KV-cache memory and prefill cost.

## Code changes in this PR

- `hadl/attention.py`: attention statistics are now accumulated on the GPU without any GPU→CPU sync, with cached masks and
  slice-based row selection. Before this change, recording forced about 100 syncs per generated token. Outputs are
  identical to the previous version on the tiny test model: same tokens, stats, trigger counts and TF-HH gradients, and
  `tests/test_pipeline.py` passes. A caption job went from about 67 min (3 jobs sharing the GPU) to about 4 min (one job alone).
- `scripts/launch.py`: a GPU id can be repeated (for example `GPUS=0,0`) to run several jobs on one GPU. Without
  MPS this does not speed things up, so use one job per GPU.
