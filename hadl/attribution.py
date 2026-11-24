"""Hallucination-head attribution (paper 2, Eq. 1-2) under visual-token compression.

Exact zero-ablation of all 1024 heads needs ~1k forward passes per caption. We use
the first-order (attribution-patching) estimate of the same quantity:

    P_M(y_t) - P_{M\\c}(y_t)  ~=  grad_{a_c} log P(y_t) . a_c

where a_c is head c's output (the input slice of o_proj). One backward pass gives
the score of every head at once. ``verify_exact`` checks the estimate with real
zero-ablation on the top heads.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from . import attention as A
from .model import Sample


def _target_positions(tok, caption: str, spans: List[Dict]) -> Tuple[torch.Tensor, List[int]]:
    enc = tok(caption, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    pos = []
    for sp in spans:
        for i, (s, e) in enumerate(offs):
            if e > sp["start"] and s < sp["end"]:
                pos.append(i)
                break  # first sub-token of the object word
    return torch.tensor(ids), sorted(set(pos))


class HeadGrabber:
    def __init__(self, model):
        self.layers = model.language_model.model.layers
        self.acts: List[torch.Tensor] = [None] * len(self.layers)
        self.ablate: Dict[int, List[int]] = {}
        self.handles = []
        cfg = model.language_model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mk(i)))

    def _mk(self, i):
        def hook(mod, args):
            x = args[0]
            if i in self.ablate:
                x = x.clone()
                for h in self.ablate[i]:
                    x[..., h * self.head_dim:(h + 1) * self.head_dim] = 0
                return (x,)
            if x.requires_grad:
                x.retain_grad()
            self.acts[i] = x
        return hook

    def head_scores(self) -> torch.Tensor:
        """[L, H] grad . act summed over positions, then cleared."""
        out = []
        for x in self.acts:
            g = x.grad
            v = (g * x).float()[0].sum(0).view(self.n_heads, self.head_dim).sum(-1)
            out.append(v.detach().cpu())
            x.grad = None
        return torch.stack(out)

    def remove(self):
        for h in self.handles:
            h.remove()


def caption_logprobs(model, s: Sample, cap_ids: torch.Tensor, grad: bool = True):
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    ce = emb(cap_ids.to(dev)).unsqueeze(0)
    x = torch.cat([s.embeds, ce], 1).detach()
    if grad:
        x.requires_grad_(True)
    A.CTRL.set_layout(s.vis_start, s.vis_end, s.embeds.shape[1])
    with torch.set_grad_enabled(grad):
        logits = model.language_model(inputs_embeds=x, use_cache=False).logits[0].float()
        lp = torch.log_softmax(logits[s.embeds.shape[1] - 1:-1], -1)      # predicts caption tokens
        tok_lp = lp.gather(-1, cap_ids.to(dev).unsqueeze(-1)).squeeze(-1)  # [len(caption)]
    return tok_lp


def attribute_one(model, grabber: HeadGrabber, s: Sample, tok, caption: str, hall_spans, good_spans):
    cap_ids, hpos = _target_positions(tok, caption, hall_spans)
    _, gpos = _target_positions(tok, caption, good_spans)
    if not hpos or not gpos:
        return None
    tok_lp = caption_logprobs(model, s, cap_ids, grad=True)
    lh = tok_lp[hpos].mean()
    lh.backward(retain_graph=True)
    sh = grabber.head_scores()
    lg = tok_lp[gpos].mean()
    lg.backward()
    sg = grabber.head_scores()
    return sh, sg


@torch.no_grad()
def verify_exact(model, grabber: HeadGrabber, s: Sample, tok, caption, hall_spans, heads: List[Tuple[int, int]]):
    """True zero-ablation effect on mean hallucinated-token log-prob, per head."""
    cap_ids, hpos = _target_positions(tok, caption, hall_spans)
    if not hpos:
        return None
    base = caption_logprobs(model, s, cap_ids, grad=False)[hpos].mean().item()
    eff = []
    for l, h in heads:
        grabber.ablate = {l: [h]}
        v = caption_logprobs(model, s, cap_ids, grad=False)[hpos].mean().item()
        eff.append(base - v)
    grabber.ablate = {}
    return eff
