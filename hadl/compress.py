"""Training-free visual token compressors for LLaVA-1.5.

All compressors work on CLIP patch features *before* the multimodal projector
(shape [N=576, C=1024]) and return [B, C] tokens plus metadata.

* ``none``   : keep all 576 tokens (upper bound).
* ``pool``   : 2-D average pooling to a sqrt(B) x sqrt(B) grid (continuous-only baseline).
* ``topk``   : keep the B patches with the highest CLS attention (semantic-only baseline).
* ``hybrid`` : HTC-style hybrid. ``n_anchor`` semantic anchors (top CLS attention,
               kept as-is) followed by ``B - n_anchor`` appearance tokens obtained by
               merging *all remaining* patches into evenly spaced centres by cosine
               similarity. Anchors first mirrors HTC's best "pre-fusion" order.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

N_PATCH = 576


def compress(feats: torch.Tensor, cls_attn: torch.Tensor, method: str, budget: int,
             anchor_frac: float = 0.5) -> Tuple[torch.Tensor, Dict]:
    """feats: [N, C] patch features; cls_attn: [N] CLS->patch attention."""
    n = feats.shape[0]
    if method == "none" or budget >= n:
        return feats, {"n_anchor": 0, "n_ctx": n}
    if method == "pool":
        side = int(round(math.sqrt(budget)))
        g = int(round(math.sqrt(n)))
        x = feats.T.reshape(1, -1, g, g)
        x = F.adaptive_avg_pool2d(x.float(), side).to(feats.dtype)
        return x.reshape(feats.shape[1], -1).T.contiguous(), {"n_anchor": 0, "n_ctx": side * side}
    if method == "topk":
        idx = cls_attn.topk(budget).indices.sort().values
        return feats[idx], {"n_anchor": budget, "n_ctx": 0, "anchor_idx": idx.tolist()}
    if method == "hybrid":
        n_anchor = max(1, min(budget - 1, int(round(budget * anchor_frac))))
        n_ctx = budget - n_anchor
        a_idx = cls_attn.topk(n_anchor).indices.sort().values
        mask = torch.ones(n, dtype=torch.bool, device=feats.device)
        mask[a_idx] = False
        rest = feats[mask]
        # evenly spaced centres among remaining tokens, then cosine assignment + mean merge
        c_pos = torch.linspace(0, rest.shape[0] - 1, n_ctx, device=feats.device).round().long()
        rn = F.normalize(rest.float(), dim=-1)
        sim = rn @ rn[c_pos].T                                      # [R, n_ctx]
        assign = sim.argmax(-1)
        assign[c_pos] = torch.arange(n_ctx, device=feats.device)    # centres keep their own slot
        ctx = torch.zeros(n_ctx, feats.shape[1], device=feats.device, dtype=torch.float32)
        ctx.index_add_(0, assign, rest.float())
        cnt = torch.bincount(assign, minlength=n_ctx).clamp_min(1).unsqueeze(-1)
        ctx = (ctx / cnt).to(feats.dtype)
        out = torch.cat([feats[a_idx], ctx], 0)
        return out, {"n_anchor": n_anchor, "n_ctx": n_ctx, "anchor_idx": a_idx.tolist()}
    raise ValueError(f"unknown compression method {method}")
