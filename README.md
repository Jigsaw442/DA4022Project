# HADL: Hallucination-Aware Hybrid Token Compression for VLMs

DA4022 course project. HADL studies what visual-token compression does to object hallucination in a
vision-language model, and uses head-level attention control to win back the grounding that compression loses.
It combines two lines of work:

1. **Hybrid token compression** (HTC-VLM, Zhang et al., CVPR 2026): replace LLaVA's 576 visual tokens with a few
   *hybrid* tokens made of semantic anchors plus appearance tokens.
2. **Hallucination heads** (*Mitigating Hallucination in LVLMs via Modular Attribution and Intervention*,
   OpenReview `Bjq4W7P2Us`): a small set of attention heads that over-attend to text drive object hallucination.
   AD-HH (training-free) and TF-HH (fine-tuning) suppress their text attention.

**Research question:** when a VLM sees far fewer visual tokens, do hallucination heads lean on the text even more,
and can head-level intervention recover the lost grounding?

Base model: `llava-hf/llava-1.5-7b-hf`. HTC-VLM did not release trained weights, and retraining it needs 8xA100 for
more than a day, so the compressor here is a **training-free** version that follows HTC's anchor + appearance design.

## Method

| Component | File | What it does |
|---|---|---|
| Hybrid compressor | `hadl/compress.py` | `hybrid`: the patches with the highest CLS attention become *anchor* tokens (semantic); all other patches are merged into evenly spaced *appearance* tokens by cosine similarity. Anchors come first, following HTC's "pre-fusion" order. Baselines: `pool` (appearance only), `topk` (anchors only) and `none` (all 576 tokens). |
| Head-level attention control | `hadl/attention.py` | Custom attention kernel registered in `transformers` (numerically identical to eager attention when no intervention is active). Records the text / visual / BOS attention mass of every head and applies AD-HH or VR-HH to selected heads. |
| Hallucination-head attribution | `hadl/attribution.py`, `scripts/find_heads.py` | The contrastive head score `I_hall - I_non-hall` from paper 2, estimated with one backward pass (gradient x head output, a first-order estimate of zero-ablation). `--verify-topk` compares it with exact ablation. |
| **VR-HH** (new) | `hadl/attention.py` | *Visual-Rebalanced* intervention. AD-HH zeroes a head's text attention, which shrinks the head's output when only 16-64 visual tokens are left. VR-HH instead moves the removed text mass onto the visual tokens, in proportion to the head's existing visual attention, so each row still sums to 1. |
| **Risk-adaptive budget** (new) | `hadl/runner.py` (`--adaptive LOW,HIGH`) | One cheap prefill at LOW tokens measures the text mass of the hallucination heads. The riskiest inputs (top 30% by default) are re-encoded with HIGH tokens. |
| **Compression-aware TF-HH** (new, optional) | `scripts/train_tfhh.py` | Paper 2's loss `CE + lambda * mean log A_text(h)`, trained while the model sees compressed tokens. Only the q/k rows of the selected heads are updated, so the learned delta covers about 20 heads. Training targets are COCO reference captions of images that are disjoint from the evaluation sets. |
| Metrics | `hadl/chair.py`, `scripts/pope_eval.py` | CHAIR_s / CHAIR_i / recall on COCO val2014 captions; POPE accuracy / F1 / yes-ratio from first-token Yes-vs-No logits. |
| Pipeline | `scripts/run_all.sh`, `scripts/launch.py`, `scripts/make_report_assets.py` | Resumable multi-GPU job runner that writes Markdown and LaTeX tables and PDF/PNG figures. |

## Results

The full numbers, figures and discussion are in **[RESULTS.md](RESULTS.md)**, and the raw outputs (metrics, captions,
attention statistics, head scores, logs) are in [`results_quick/`](results_quick). These come from a reduced run
(`QUICK=1`: 200 captions, 1,200 POPE-adversarial questions) on one H100.

Highlights:

- Uncompressed LLaVA-1.5-7B: CHAIR_s 53.0, POPE-adversarial accuracy 84.6.
- At 64 visual tokens the hybrid compressor beats both pooling and top-k selection on CHAIR and POPE.
- Compression-aware TF-HH at 64 tokens lowers CHAIR_s from 45.0 to 36.5 with almost no change in caption length,
  at a cost of 2.5 POPE points.
- VR-HH at 64 tokens gives the lowest CHAIR (27.5 / 16.8), but partly because captions get shorter. At 16 tokens
  it hurts, since there is little visual evidence left to move attention onto.
- AD-HH and VR-HH leave POPE essentially unchanged, because POPE is decided by a single yes/no token.

With 200 captions, CHAIR_s has a standard error of about 3.5 points, so gaps smaller than about 7 points are weak evidence.

## Setup

```bash
conda create -n hadl python=3.10 -y && conda activate hadl
# Install the torch build that matches your driver first (see the CUDA version in `nvidia-smi`), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/setup_data.sh data          # COCO val2014 + annotations + POPE (~7 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')"  # ~14 GB
```

Optional CPU sanity check (about 1 minute, no downloads):

```bash
python tests/make_tiny.py /tmp/tiny && python tests/test_pipeline.py /tmp/tiny
```

It builds a tiny random LLaVA and checks that the input path matches HF LLaVA logits, that the interventions behave
as intended, and that every compressor returns the requested number of tokens.

GPU preflight (about 5 minutes; run it before a long run):

```bash
python scripts/gpu_preflight.py          # add --skip-train on GPUs with less than 40 GB
```

It loads the real model and runs each GPU code path once (input path vs HF logits, greedy / AD-HH / VR-HH
captioning, the POPE forward pass, the attribution backward pass and one TF-HH step), printing PASS/FAIL, peak
memory and decoding speed for each.

## Running

```bash
GPUS=0,1,2,3 bash scripts/run_all.sh     # full run
QUICK=1 GPUS=0 bash scripts/run_all.sh   # reduced run (200 captions, 1,200 POPE questions), as in RESULTS.md
TRAIN=0 GPUS=0 bash scripts/run_all.sh   # skip the optional TF-HH stage
```

Finished jobs are skipped on a re-run, so the pipeline can be stopped and restarted safely. Logs go to `logs/`.
Use one job per GPU; repeating a GPU id (e.g. `GPUS=0,0`) is allowed but does not speed things up.

| Stage | Jobs | Approx. A100-80GB time (full run) |
|---|---|---|
| 1. Greedy captions + POPE for 576/144/64/16 hybrid, pool64, topk64 | 12 | ~3.5 h |
| 2. Hallucination-head attribution at each budget | 4 | ~0.7 h |
| 3. AD-HH / VR-HH at each budget, head transfer, adaptive budget | 21 | ~6 h |
| 4-5. TF-HH training at 64 tokens + evaluation (optional) | 5 | ~1.5 h |

The full run is about 12 GPU-hours (3-4 hours of wall time on 4 GPUs); `QUICK=1` cuts this by about 60%.
Inference needs about 16 GB of GPU memory and TF-HH about 30 GB. On a 16 GB card, add `EXTRA="--load-4bit"`
(inference only; needs `pip install bitsandbytes`). The attention kernel is eager because it needs the attention
weights, so FlashAttention is not used.

### Single experiments

```bash
python scripts/caption_eval.py --method hybrid --budget 64 --out results/caption/hybrid64_greedy
python scripts/find_heads.py --captions results/caption/hybrid64_greedy/captions.jsonl \
    --method hybrid --budget 64 --out results/heads/hybrid64.json --verify-topk 24
python scripts/caption_eval.py --method hybrid --budget 64 --decode vrhh --heads results/heads/hybrid64.json \
    --out results/caption/hybrid64_vrhh
python scripts/pope_eval.py --method hybrid --budget 64 --decode vrhh --heads results/heads/hybrid64.json \
    --out results/pope/hybrid64_vrhh
```

Main options:

- `--method {none,pool,topk,hybrid}` and `--budget N`: compressor and number of visual tokens.
- `--decode {greedy,adhh,vrhh}` with `--heads heads.json`: head intervention.
- `--topk-heads` (default 20, as in paper 2): number of hallucination heads to intervene on.
- `--tau` (default `auto`, the selected heads' mean text mass, so the intervention fires on above-average steps;
  paper 2 used 0.4).
- `--alpha`: fraction of the text mass that is removed (AD-HH) or moved (VR-HH).
- `--anchor-frac` (default 0.5): share of anchor tokens in the hybrid budget.
- `--include-bos`: count the BOS attention sink as text (off by default because it dominates).
- `--adaptive LOW,HIGH` with `--adaptive-heads`: risk-adaptive budget.
- `--delta tfhh_delta.pt`: evaluate with TF-HH weights.

## Outputs

`run_all.sh` writes everything under `results/` (the committed QUICK run is in `results_quick/` with the same layout):

```
tables/main_results.{md,tex}      CHAIR_s, CHAIR_i, recall, length, POPE acc/F1/yes%, tokens, latency
tables/compressors.{md,tex}       pool vs topk vs hybrid at 64 tokens
figures/chair_vs_tokens.pdf       hallucination vs visual-token budget for each decoding method
figures/pope_vs_tokens.pdf
figures/diagnosis_attention.pdf   per-layer text vs visual attention; text/visual ratio of hallucination heads
figures/head_scores.pdf           contrastive head scores per budget
figures/head_overlap.pdf          overlap of the top-20 hallucination heads across budgets
figures/attribution_check.pdf     first-order attribution vs exact ablation
figures/tfhh_training.pdf
summary.json                      all numbers, diagnosis ratios and the head-overlap matrix
caption/<config>/captions.jsonl   every caption with its CHAIR labels
```

Config names follow `<method><budget>_<decode>`, e.g. `hybrid64_vrhh`. `hybrid64_vrhh-transfer` uses heads found on
the *uncompressed* model, to test whether hallucination heads change under compression.

## Project layout

```
hadl/        model wrapper, compressors, attention kernel, attribution, CHAIR, shared CLI
scripts/     evaluation, head finding, TF-HH training, pipeline and report-asset scripts
tests/       tiny-model build and CPU pipeline test
results_quick/  outputs of the QUICK run reported in RESULTS.md
```

## Notes and caveats

- CHAIR is reimplemented in `hadl/chair.py` with a condensed synonym list. Absolute numbers can differ slightly from
  the original script, but every config uses the same scorer.
- Head attribution is a first-order estimate of zero-ablation. The `verify` block in `heads/hybrid64.json` reports
  how well it matches exact ablation; in the QUICK run, layer-0 heads disagree with it (see RESULTS.md).
- In the adaptive setting, VR-HH uses heads found at the HIGH budget for both budgets.
- Data: CHAIR uses a fixed random subset of COCO val2014 (seed 0; 500 images in the full run, 200 with `QUICK=1`).
  POPE uses the adversarial split (3,000 questions in the full run, 1,200 with `QUICK=1`). TF-HH training images
  exclude both.

## References

- J. Zhang et al. *Hybrid Token Compression for Vision-Language Models.* CVPR 2026. Code: github.com/jushengzhang/HybridToken-VLM
- *Mitigating Hallucination in Large Vision-Language Models via Modular Attribution and Intervention.* OpenReview Bjq4W7P2Us (ICLR 2025 submission).
- Y. Li et al. *Evaluating Object Hallucination in Large Vision-Language Models* (POPE). EMNLP 2023.
- A. Rohrbach et al. *Object Hallucination in Image Captioning* (CHAIR). EMNLP 2018.
- H. Liu et al. *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). CVPR 2024.
