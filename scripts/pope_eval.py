"""POPE (yes/no object probing) with compression and head interventions.

The answer is read from the first-token logits (Yes vs No), so one forward pass per question.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hadl import attention as A
from hadl import model as M
from hadl.runner import add_common_args, adaptive_plan, set_decode, setup
from hadl.utils import POPE_SUFFIX, dump_json, load_image, read_jsonl, write_jsonl


def yes_no_ids(tok):
    vocab = tok.get_vocab()

    def ids(words):
        out = sorted({vocab[v] for w in words for v in ("▁" + w, w) if v in vocab})
        assert out, f"none of {words} found in vocabulary"
        return out
    return ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--coco-root", default="data/coco/val2014")
    p.add_argument("--pope-file", default="data/pope/coco_pope_adversarial.json")
    args = p.parse_args()

    qs = read_jsonl(args.pope_file)
    if args.n and args.n < len(qs):
        qs = sorted(random.Random(args.seed).sample(qs, args.n), key=lambda q: q["image"])
    for q in qs:
        q["_key"] = str(q["question_id"])
    model, proc, info = setup(args)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    img_path = lambda q: os.path.join(args.coco_root, q["image"])

    plan, thr = None, None
    if args.adaptive:
        plan, thr, _ = adaptive_plan(args, model, proc, qs, lambda q: load_image(img_path(q)),
                                     lambda q: q["text"] + POPE_SUFFIX)
    set_decode(args, info)
    A.CTRL.reset_stats()
    rows, cache_key, cache = [], None, None
    t_total = 0.0
    for i, q in enumerate(qs):
        if q["image"] != cache_key:
            cache = M.encode_image(model, proc, load_image(img_path(q)))
            cache_key = q["image"]
        budget = plan[q["_key"]] if plan else args.budget
        t0 = time.time()
        s = M.build_sample(model, proc, None, q["text"] + POPE_SUFFIX, args.method, budget, args.anchor_frac, cache)
        logits = M.next_token_logits(model, s)
        t_total += time.time() - t0
        y, n = logits[yes_ids].max().item(), logits[no_ids].max().item()
        pred = "yes" if y > n else "no"
        rows.append({"question_id": q["question_id"], "image": q["image"], "text": q["text"],
                     "label": q["label"], "pred": pred, "margin": y - n, "budget": budget})
        if i % 500 == 0:
            print(f"[{i}/{len(qs)}]", flush=True)

    tp = sum(r["pred"] == "yes" and r["label"] == "yes" for r in rows)
    fp = sum(r["pred"] == "yes" and r["label"] == "no" for r in rows)
    tn = sum(r["pred"] == "no" and r["label"] == "no" for r in rows)
    fn = sum(r["pred"] == "no" and r["label"] == "yes" for r in rows)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    metrics = {"n": len(rows), "accuracy": 100 * (tp + tn) / len(rows), "precision": 100 * prec,
               "recall": 100 * rec, "f1": 100 * 2 * prec * rec / max(prec + rec, 1e-9),
               "yes_ratio": 100 * (tp + fp) / len(rows), "method": args.method, "budget": args.budget,
               "decode": args.decode, "adaptive": args.adaptive, "risk_threshold": thr,
               "avg_visual_tokens": sum(r["budget"] for r in rows) / len(rows),
               "sec_per_question": t_total / len(rows), "tau": info["tau"],
               "pope_file": os.path.basename(args.pope_file), "delta": args.delta}
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "answers.jsonl"), rows)
    dump_json(os.path.join(args.out, "metrics.json"), metrics)
    dump_json(os.path.join(args.out, "attn_stats.json"), A.CTRL.summary())
    print(metrics)


if __name__ == "__main__":
    main()
