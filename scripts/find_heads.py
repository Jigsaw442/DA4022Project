"""Locate hallucination heads of the (compressed) model from its own captions.

Uses captions + CHAIR labels written by caption_eval.py (greedy decoding at the
same compression setting). Writes heads.json with
  scores_hall / scores_good / contrastive : [L][H]
  top          : heads sorted by contrastive score (most hallucination-prone first)
  text_mass / vis_mass / bos_mass : mean attention mass per head on caption rows
  verify       : (optional) exact zero-ablation check of the top heads
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from hadl import attention as A
from hadl import model as M
from hadl.attribution import HeadGrabber, attribute_one, verify_exact
from hadl.utils import CAPTION_PROMPT, dump_json, load_image, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=M.DEFAULT_MODEL)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--captions", required=True)
    p.add_argument("--coco-root", default="data/coco/val2014")
    p.add_argument("--method", default="hybrid")
    p.add_argument("--budget", type=int, default=576)
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.add_argument("--n", type=int, default=300, help="max captions used")
    p.add_argument("--top", type=int, default=64, help="length of the saved ranking")
    p.add_argument("--verify-topk", type=int, default=0)
    p.add_argument("--verify-n", type=int, default=30)
    p.add_argument("--include-bos", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model, proc = M.load(args.model, args.dtype, False, args.device)
    model.language_model.gradient_checkpointing_disable()
    tok = proc.tokenizer
    grab = HeadGrabber(model)
    A.CTRL.include_bos_as_text = args.include_bos
    A.CTRL.mode, A.CTRL.heads, A.CTRL.record = "off", {}, True
    A.CTRL.reset_stats()

    rows = [r for r in read_jsonl(args.captions) if r["hall_spans"] and r["good_spans"]][: args.n]
    print(f"{len(rows)} captions with both hallucinated and correct objects")
    SH, SG, used = 0, 0, 0
    for i, r in enumerate(rows):
        img = load_image(os.path.join(args.coco_root, r["image"]))
        s = M.build_sample(model, proc, img, CAPTION_PROMPT, args.method, args.budget, args.anchor_frac)
        res = attribute_one(model, grab, s, tok, r["caption"], r["hall_spans"], r["good_spans"])
        if res is None:
            continue
        SH = SH + res[0]; SG = SG + res[1]; used += 1
        if i % 25 == 0:
            print(f"[{i}/{len(rows)}]", flush=True)
    if used == 0:
        sys.exit("no usable captions: need captions with both hallucinated and correct objects")
    SH, SG = SH / used, SG / used
    con = SH - SG
    L, H = con.shape
    order = torch.argsort(con.flatten(), descending=True)
    top = [[int(k // H), int(k % H)] for k in order[: args.top]]
    stats = A.CTRL.summary()
    out = {"method": args.method, "budget": args.budget, "n_used": used,
           "scores_hall": SH.tolist(), "scores_good": SG.tolist(), "contrastive": con.tolist(), "top": top,
           "text_mass": stats["text"], "vis_mass": stats["vis"], "bos_mass": stats["bos"]}

    if args.verify_topk:
        A.CTRL.record = False
        cand = top[: args.verify_topk]
        est = [float(SH[l, h]) for l, h in cand]
        effs = []
        for r in rows[: args.verify_n]:
            img = load_image(os.path.join(args.coco_root, r["image"]))
            s = M.build_sample(model, proc, img, CAPTION_PROMPT, args.method, args.budget, args.anchor_frac)
            e = verify_exact(model, grab, s, tok, r["caption"], r["hall_spans"], cand)
            if e is not None:
                effs.append(e)
        exact = np.mean(effs, 0).tolist()
        out["verify"] = {"heads": cand, "estimate": est, "exact": exact,
                         "pearson": float(np.corrcoef(est, exact)[0, 1]) if len(cand) > 2 else None}
        print("verify pearson:", out["verify"]["pearson"])

    dump_json(args.out, out)
    print("top heads:", top[:20])


if __name__ == "__main__":
    main()
