# HADL: Hallucination-Aware Hybrid Token Compression for VLMs

DA4022 course project. We merge two ideas:

1. **HTC-VLM** (Zhang et al., CVPR 2026, *Hybrid Token Compression for Vision-Language Models*): compress
   the 576 visual tokens of LLaVA into a few *hybrid* tokens (semantic anchors + appearance).
2. **Hallucination heads** (*Mitigating Hallucination in LVLMs via Modular Attribution and Intervention*,
   OpenReview `Bjq4W7P2Us`): a small set of attention heads that over-attend to text cause object
   hallucination. AD-HH (training-free) and TF-HH (fine-tuning) suppress their text attention.

**Question:** when a VLM sees far fewer visual tokens, do hallucination heads lean on text even more, and can
head-level control recover the grounding that compression loses?

## What is implemented

| Component | File | What it does |
|---|---|---|
| Training-free hybrid compressor | `hadl/compress.py` | `hybrid`: top-CLS-attention *anchor* tokens (semantic) + all other patches merged into evenly spaced *appearance* tokens by cosine similarity; anchors first (HTC's best "pre-fusion" order). Baselines: `pool` (continuous only), `topk` (semantic only), `none` (576). |
| Head-level attention control | `hadl/attention.py` | Custom attention kernel registered in `transformers` (numerically identical to eager when idle). Records text/visual/BOS attention mass per head; applies AD-HH or VR-HH to chosen heads. |
| Hallucination-head attribution | `hadl/attribution.py`, `scripts/find_heads.py` | Paper 2's contrastive score `I_hall - I_non-hall` per head, estimated with one backward pass (grad x head output, first-order estimate of zero-ablation). `--verify-topk` checks it against real ablation (Pearson r ~0.96 on the test model). |
| **VR-HH (new)** | `hadl/attention.py` | *Visual-Rebalanced* head intervention: instead of zeroing a hallucination head's text attention (AD-HH), the removed mass is moved onto the visual tokens in proportion to the head's existing visual attention, so rows still sum to 1. Under heavy compression there are only 16-64 visual tokens, so AD-HH's un-normalized rows shrink the head output; VR-HH keeps the scale and pushes the head toward the image. |
| **Hallucination-risk adaptive budget (new)** | `hadl/runner.py` (`--adaptive LOW,HIGH`) | One cheap prefill at LOW tokens measures the text mass of the hallucination heads; risky inputs (top 30% by default) are re-encoded with HIGH tokens. |
| **Compression-aware TF-HH (new, optional)** | `scripts/train_tfhh.py` | Paper 2's loss `CE + lambda * mean log A_text(h)` but trained *while the model sees compressed tokens*; only the q/k rows of the selected heads are updated (gradient mask), so the delta is ~20 heads. Targets are COCO reference captions of images disjoint from the eval sets. |
| Metrics | `hadl/chair.py`, `scripts/pope_eval.py` | Self-contained CHAIR_s / CHAIR_i / recall on COCO val2014; POPE accuracy / F1 / yes-ratio from first-token Yes-vs-No logits. |
| Pipeline + report assets | `scripts/run_all.sh`, `scripts/launch.py`, `scripts/make_report_assets.py` | Resumable multi-GPU job runner; LaTeX + Markdown tables and PDF/PNG figures. |

Model: `llava-hf/llava-1.5-7b-hf` (the base HTC-VLM is built on). HTC-VLM did not release trained weights and retraining
needs 8xA100 for more than a day, so the compressor here is a **training-free stand-in** that follows HTC's
anchor + appearance design. Say this in the report.

## Setup

```bash
conda create -n hadl python=3.10 -y && conda activate hadl
pip install -r requirements.txt
bash scripts/setup_data.sh data          # COCO val2014 + annotations + POPE (~7 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')"  # ~14 GB
```

Optional sanity check (CPU, 1 min, no downloads):
```bash
python tests/make_tiny.py /tmp/tiny && python tests/test_pipeline.py /tmp/tiny
```
It checks our input path gives the same logits as HF LLaVA (max diff ~1e-7), that interventions behave as intended, and that compressors return the right token counts.

## Run everything

```bash
GPUS=0,1,2,3 bash scripts/run_all.sh            # full run
QUICK=1 GPUS=0,1 bash scripts/run_all.sh        # smaller run (200 captions, 1200 POPE questions)
TRAIN=0 GPUS=0 bash scripts/run_all.sh          # skip the optional TF-HH stage
```

Re-running skips finished jobs, so it is safe to stop and restart. Logs go to `logs/`.

| Stage | Jobs | Approx. A100-80GB time |
|---|---|---|
| 1. Greedy captions + POPE for 576/144/64/16 hybrid + pool64/topk64 | 12 | ~3.5 h |
| 2. Hallucination-head attribution at each budget | 4 | ~0.7 h |
| 3. AD-HH / VR-HH at each budget, head transfer, adaptive budget | 21 | ~6 h |
| 4-5. TF-HH training (64 tokens) + eval (optional) | 5 | ~1.5 h |

That is about **12 GPU-hours** in total, or **3-4 hours of wall time on 4 GPUs**. Use `QUICK=1` to cut it by about 60%.
GPU memory: inference needs about 16 GB (A100 40GB, L40S, A6000, RTX 4090 all work); TF-HH needs about 30 GB.
For a 16 GB card (T4/V100), add `EXTRA="--load-4bit"` (inference only; needs `pip install bitsandbytes`).

The attention kernel is eager (it needs the attention weights), so FlashAttention is not used. Paper 2 has the same limitation.

## Outputs (for the report)

```
results/
  tables/main_results.{md,tex}   CHAIR_s, CHAIR_i, recall, length, POPE acc/F1/yes%, tokens, latency
  tables/compressors.{md,tex}    pool vs topk vs hybrid at 64 tokens
  figures/chair_vs_tokens.pdf    hallucination vs visual-token budget, Greedy / AD-HH / VR-HH / adaptive / TF-HH
  figures/pope_vs_tokens.pdf
  figures/diagnosis_attention.pdf  per-layer text vs visual attention; text/visual ratio of hallucination heads per budget
  figures/head_scores.pdf        contrastive head-score heatmaps per budget
  figures/head_overlap.pdf       Jaccard overlap of top-20 hallucination heads across budgets
  figures/attribution_check.pdf  first-order attribution vs exact ablation
  figures/tfhh_training.pdf
  summary.json                   all numbers + diagnosis ratios + overlap matrix
  caption/<config>/captions.jsonl  every caption with its CHAIR labels (good for qualitative examples)
```

Config names are `<method><budget>_<decode>`, e.g. `hybrid64_vrhh`. `hybrid64_vrhh-transfer` uses heads found on the
*uncompressed* model, which tests whether hallucination heads change under compression.

## Running single experiments

```bash
python scripts/caption_eval.py --method hybrid --budget 64 --out results/caption/hybrid64_greedy
python scripts/find_heads.py --captions results/caption/hybrid64_greedy/captions.jsonl \
    --method hybrid --budget 64 --out results/heads/hybrid64.json --verify-topk 24
python scripts/caption_eval.py --method hybrid --budget 64 --decode vrhh --heads results/heads/hybrid64.json \
    --out results/caption/hybrid64_vrhh
python scripts/pope_eval.py --method hybrid --budget 64 --decode vrhh --heads results/heads/hybrid64.json \
    --out results/pope/hybrid64_vrhh
```

Useful knobs: `--topk-heads` (default 20, as in paper 2), `--tau` (default `auto` = the selected heads' mean text
mass, so the intervention fires on above-average steps; paper 2 used 0.4), `--alpha` (fraction of text mass
removed or moved), `--anchor-frac` (share of anchors in the hybrid budget, default 0.5), `--include-bos`
(count the BOS attention sink as text; off by default because it dominates).

## Notes and caveats

- CHAIR is reimplemented (`hadl/chair.py`) with a condensed synonym list. Absolute numbers can differ slightly from the
  original script, but all configs use the same scorer.
- Head attribution uses a first-order estimate of zero-ablation. The `verify` block in `results/heads/hybrid64.json`
  reports how well it matches exact ablation.
- In the adaptive setting, VR-HH uses heads found at the HIGH budget for both budgets.
- Data split: CHAIR uses a fixed random 500-image subset of val2014 (seed 0). POPE uses the adversarial split (3000 questions, 500 images).
  TF-HH training images exclude both.

## References

- J. Zhang et al. *Hybrid Token Compression for Vision-Language Models.* CVPR 2026. Code: github.com/jushengzhang/HybridToken-VLM
- *Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention.* OpenReview Bjq4W7P2Us (ICLR 2025 submission).
- Y. Li et al. *Evaluating Object Hallucination in Large Vision-Language Models* (POPE). EMNLP 2023.
- A. Rohrbach et al. *Object Hallucination in Image Captioning* (CHAIR). EMNLP 2018.
- H. Liu et al. *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). CVPR 2024.
