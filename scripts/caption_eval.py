"""Detailed captioning on COCO val2014 + CHAIR, with compression and head interventions.

Example:
  python scripts/caption_eval.py --method hybrid --budget 64 --decode vrhh \
      --heads results/heads/hybrid64.json --out results/caption/hybrid64_vrhh
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hadl import attention as A
from hadl import model as M
from hadl.chair import Chair
from hadl.runner import add_common_args, adaptive_plan, set_decode, setup
from hadl.utils import CAPTION_PROMPT, dump_json, load_image, write_jsonl, coco_subset


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--coco-root", default="data/coco/val2014")
    p.add_argument("--ann-dir", default="data/coco/annotations")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--offset", type=int, default=0, help="skip the first OFFSET images of the fixed subset")
    p.add_argument("--prompt", default=CAPTION_PROMPT)
    args = p.parse_args()

    chair = Chair(os.path.join(args.ann_dir, "instances_val2014.json"),
                  os.path.join(args.ann_dir, "captions_val2014.json"))
    images = coco_subset(chair, args.n, args.seed, offset=args.offset)
    for im in images:
        im["_key"] = im["file_name"]
    model, proc, info = setup(args)
    eos = proc.tokenizer.eos_token_id
    img_path = lambda it: os.path.join(args.coco_root, it["file_name"])

    plan, thr = None, None
    if args.adaptive:
        plan, thr, risks = adaptive_plan(args, model, proc, images, lambda it: load_image(img_path(it)),
                                         lambda it: args.prompt)

    set_decode(args, info)
    A.CTRL.reset_stats()
    rows, scored, caps = [], [], []
    t_total, n_tok = 0.0, 0
    for i, im in enumerate(images):
        budget = plan[im["_key"]] if plan else args.budget
        img = load_image(img_path(im))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        s = M.build_sample(model, proc, img, args.prompt, args.method, budget, args.anchor_frac)
        toks = M.greedy_generate(model, s, args.max_new_tokens, eos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        t_total += dt; n_tok += len(toks)
        cap = proc.tokenizer.decode(toks, skip_special_tokens=True).strip()
        sc = chair.score_caption(im["id"], cap)
        scored.append(sc); caps.append(cap)
        rows.append({"image": im["file_name"], "image_id": im["id"], "caption": cap, "budget": budget,
                     "prompt_len": int(s.embeds.shape[1]), "latency_s": dt, **sc})
        if i % 25 == 0:
            print(f"[{i}/{len(images)}] budget={budget} {dt:.2f}s :: {cap[:100]}", flush=True)

    metrics = Chair.aggregate(scored, caps)
    metrics.update({"method": args.method, "budget": args.budget, "decode": args.decode,
                    "adaptive": args.adaptive, "risk_threshold": thr,
                    "avg_visual_tokens": sum(r["budget"] for r in rows) / len(rows),
                    "avg_prompt_len": sum(r["prompt_len"] for r in rows) / len(rows),
                    "sec_per_caption": t_total / len(rows), "tokens_per_sec": n_tok / max(t_total, 1e-9),
                    "tau": info["tau"], "n_heads": len(info["heads"]),
                    "trigger_rate": A.CTRL.trigger_count / max(A.CTRL.rows_seen, 1) if args.decode != "greedy" else None,
                    "delta": args.delta})
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "captions.jsonl"), rows)
    dump_json(os.path.join(args.out, "metrics.json"), metrics)
    dump_json(os.path.join(args.out, "attn_stats.json"), A.CTRL.summary())
    print(metrics)


if __name__ == "__main__":
    main()
