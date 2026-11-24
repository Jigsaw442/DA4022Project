"""Shared CLI plumbing: model loading, decode-mode setup, adaptive budget."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from . import attention as A
from . import model as M


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--model", default=M.DEFAULT_MODEL)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--method", default="hybrid", choices=["none", "pool", "topk", "hybrid"])
    p.add_argument("--budget", type=int, default=576)
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.add_argument("--decode", default="greedy", choices=["greedy", "adhh", "vrhh"])
    p.add_argument("--heads", default=None, help="heads.json from find_heads.py")
    p.add_argument("--topk-heads", type=int, default=20)
    p.add_argument("--tau", default="auto", help="float, or 'auto' (= mean text mass of the chosen heads)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--renorm", action="store_true")
    p.add_argument("--include-bos", action="store_true", help="count BOS as a text token")
    p.add_argument("--adaptive", default=None,
                   help="'LOW,HIGH' budgets: start at LOW, escalate to HIGH when hallucination-head risk is high")
    p.add_argument("--adaptive-heads", default=None, help="heads.json measured at the LOW budget")
    p.add_argument("--risk-quantile", type=float, default=0.7,
                   help="fraction of samples kept at LOW budget (risk threshold quantile)")
    p.add_argument("--delta", default=None, help="TF-HH weights from train_tfhh.py")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")


def load_heads(path: str, k: int):
    d = json.load(open(path))
    pairs = d["top"][:k]
    text = np.array(d["text_mass"]) if "text_mass" in d else None
    tau_auto = float(np.mean([text[l][h] for l, h in pairs])) if text is not None else 0.4
    return A.heads_from_list(pairs), tau_auto, pairs


def setup(args):
    model, proc = M.load(args.model, args.dtype, args.load_4bit, args.device)
    if args.delta:
        M.load_delta(model, args.delta)
    A.CTRL.include_bos_as_text = args.include_bos
    A.CTRL.alpha, A.CTRL.renorm = args.alpha, args.renorm
    info = {"tau": None, "heads": []}
    if args.decode != "greedy":
        assert args.heads, "--heads is required for adhh / vrhh"
        heads, tau_auto, pairs = load_heads(args.heads, args.topk_heads)
        tau = tau_auto if args.tau == "auto" else float(args.tau)
        info.update(tau=tau, heads=pairs)
    return model, proc, info


def set_decode(args, info, heads_override=None):
    """Arm the controller for the decoding phase."""
    A.CTRL.record = True
    if args.decode == "greedy":
        A.CTRL.mode, A.CTRL.heads = "off", {}
    else:
        A.CTRL.mode = args.decode
        A.CTRL.heads = A.heads_from_list(info["heads"])
        A.CTRL.tau = info["tau"]


@torch.no_grad()
def prefill_risk(model, s: M.Sample, heads: Dict[int, List[int]]) -> float:
    """Mean text-attention mass of hallucination heads at the first decoding step."""
    saved = (A.CTRL.mode, A.CTRL.heads, A.CTRL.record)
    A.CTRL.mode, A.CTRL.heads, A.CTRL.record = "record", heads, False
    A.CTRL.head_text_trace.clear()
    M.next_token_logits(model, s)
    r = float(np.mean(A.CTRL.head_text_trace)) if A.CTRL.head_text_trace else 0.0
    A.CTRL.head_text_trace.clear()
    A.CTRL.mode, A.CTRL.heads, A.CTRL.record = saved
    return r


def adaptive_plan(args, model, proc, items, image_fn, question_fn):
    """Pass 1: risk of every item at the LOW budget; returns {key: budget} and the threshold."""
    low, high = [int(x) for x in args.adaptive.split(",")]
    heads, _, _ = load_heads(args.adaptive_heads or args.heads, args.topk_heads)
    risks = {}
    for it in items:
        feats = M.encode_image(model, proc, image_fn(it))
        s = M.build_sample(model, proc, None, question_fn(it), args.method, low, args.anchor_frac, feats)
        risks[it["_key"]] = prefill_risk(model, s, heads)
    thr = float(np.quantile(list(risks.values()), args.risk_quantile))
    plan = {k: (high if r > thr else low) for k, r in risks.items()}
    return plan, thr, risks
