"""Head-level attention control for the LLaVA language model.

We register a custom attention function ("hadl") with transformers. It is
numerically identical to eager attention, but it lets a global controller
  * record how much attention every head puts on visual / text / BOS keys,
  * intervene on chosen "hallucination heads" (AD-HH or VR-HH), and
  * expose differentiable text-attention mass for the TF-HH regulariser.

Key layout for one sample (batch size 1, no padding):
  [BOS][system/prefix text][visual tokens (budget B)][instruction text][generated...]
"Text" keys are every key that is not visual and not BOS. BOS is excluded by
default because it is an attention sink that otherwise dominates the text mass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ATTN_NAME = "hadl"


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@dataclass
class AttnController:
    # layout of the current sample
    vis_start: int = 0
    vis_end: int = 0
    prompt_len: int = 0          # number of prompt tokens (rows >= prompt_len-1 are "generation" rows)
    include_bos_as_text: bool = False

    # mode: "off" | "record" | "adhh" | "vrhh" | "train"
    mode: str = "off"
    record: bool = False          # record stats in addition to the mode
    heads: Dict[int, List[int]] = field(default_factory=dict)  # layer -> head ids
    tau: float = 0.4              # trigger threshold on text mass
    alpha: float = 1.0            # fraction of text mass moved (VR-HH) / removed (AD-HH)
    renorm: bool = False          # AD-HH: renormalise rows after zeroing text attention

    # recorded statistics: layer -> tensor[H] sums, plus counts
    stats_text: Dict[int, torch.Tensor] = field(default_factory=dict)
    stats_vis: Dict[int, torch.Tensor] = field(default_factory=dict)
    stats_bos: Dict[int, torch.Tensor] = field(default_factory=dict)
    stats_count: Dict[int, int] = field(default_factory=dict)
    # per-step text mass of the selected heads (for tau calibration / adaptive budget)
    head_text_trace: List[float] = field(default_factory=list)
    trig_acc: Optional[torch.Tensor] = None   # number of triggered rows, kept on the GPU (no sync)
    rows_seen: int = 0
    # differentiable text mass for training: list of tensors [n_rows] per selected head
    train_terms: List[torch.Tensor] = field(default_factory=list)

    def set_layout(self, vis_start: int, vis_end: int, prompt_len: int):
        self.vis_start, self.vis_end, self.prompt_len = vis_start, vis_end, prompt_len

    @property
    def trigger_count(self) -> int:
        return int(self.trig_acc.sum()) if self.trig_acc is not None else 0

    def reset_stats(self):
        self.stats_text.clear(); self.stats_vis.clear(); self.stats_bos.clear(); self.stats_count.clear()
        self.head_text_trace.clear(); self.train_terms.clear()
        self.trig_acc = None; self.rows_seen = 0

    def summary(self) -> Dict[str, list]:
        """Mean attention mass per (layer, head) over recorded generation rows."""
        out = {"text": [], "vis": [], "bos": []}
        for layer in sorted(self.stats_count):
            c = max(self.stats_count[layer], 1)
            out["text"].append((self.stats_text[layer].float().cpu() / c).tolist())
            out["vis"].append((self.stats_vis[layer].float().cpu() / c).tolist())
            out["bos"].append((self.stats_bos[layer].float().cpu() / c).tolist())
        return out


CTRL = AttnController()


_MASKS: Dict[tuple, tuple] = {}
_HIDX: Dict[tuple, torch.Tensor] = {}


def _key_masks(k_len: int, device) -> tuple:
    """(vis, text, bos) float masks over keys; cached so decoding launches no extra work."""
    key = (k_len, CTRL.vis_start, CTRL.vis_end, CTRL.include_bos_as_text, str(device))
    m = _MASKS.get(key)
    if m is None:
        if len(_MASKS) > 4096:
            _MASKS.clear()
        vis = torch.zeros(k_len, dtype=torch.bool)
        vis[CTRL.vis_start:min(CTRL.vis_end, k_len)] = True
        bos = torch.zeros(k_len, dtype=torch.bool)
        bos[0] = True
        text = ~vis
        if not CTRL.include_bos_as_text:
            text = text & ~bos
        m = tuple(x.float().to(device) for x in (vis, text, bos))
        _MASKS[key] = m
    return m


def _head_index(layer: int, heads: List[int], device) -> torch.Tensor:
    key = (layer, tuple(heads), str(device))
    t = _HIDX.get(key)
    if t is None:
        t = _HIDX[key] = torch.tensor(heads, device=device)
    return t


def hadl_attention_forward(module: nn.Module, query, key, value, attention_mask, scaling: float,
                           dropout: float = 0.0, **kwargs):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn = attn + attention_mask[:, :, :, : key_states.shape[-2]]
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)

    layer = getattr(module, "layer_idx", -1)
    q_len, k_len = attn.shape[-2], attn.shape[-1]
    if CTRL.mode != "off" or CTRL.record:
        # absolute positions of the query rows (batch 1, no padding). Everything below is computed
        # from Python ints and float masks, so it never forces a GPU->CPU sync during decoding.
        first_row = k_len - q_len
        gen_from = max(CTRL.prompt_len - 1, 0)
        start = max(gen_from - first_row, 0)       # first selected row within this call's queries
        if start < q_len:
            vis_f, text_f, bos_f = _key_masks(k_len, attn.device)
            if CTRL.record:
                a = attn[0][:, start:].sum(1).detach()     # [H, K], summed over selected rows
                t, v, b = a @ text_f, a @ vis_f, a @ bos_f   # [H]
                n = q_len - start
                if layer not in CTRL.stats_count:
                    CTRL.stats_text[layer] = torch.zeros_like(t)
                    CTRL.stats_vis[layer] = torch.zeros_like(v)
                    CTRL.stats_bos[layer] = torch.zeros_like(b)
                    CTRL.stats_count[layer] = 0
                CTRL.stats_text[layer] += t; CTRL.stats_vis[layer] += v; CTRL.stats_bos[layer] += b
                CTRL.stats_count[layer] += n

            heads = CTRL.heads.get(layer)
            if heads:
                h_idx = _head_index(layer, heads, attn.device)
                sub = attn[:, h_idx, start:]                        # [B, h, rows, K]
                text_mass = sub @ text_f                            # [B, h, rows]
                if CTRL.mode == "train":
                    CTRL.train_terms.append(text_mass.reshape(-1))
                elif CTRL.mode == "record":
                    CTRL.head_text_trace.extend(text_mass[:, :, -1].reshape(-1).detach().float().cpu().tolist())
                if CTRL.mode in ("adhh", "vrhh"):
                    trig = (text_mass > CTRL.tau).unsqueeze(-1)       # [B, h, rows, 1]
                    cnt = trig.sum()
                    CTRL.trig_acc = cnt if CTRL.trig_acc is None else CTRL.trig_acc + cnt
                    CTRL.rows_seen += trig.numel()
                    tm = text_f.view(1, 1, 1, -1)
                    vm = vis_f.view(1, 1, 1, -1)
                    if CTRL.mode == "adhh":
                        new = sub * (1 - CTRL.alpha * tm)
                        if CTRL.renorm:
                            new = new / new.sum(-1, keepdim=True).clamp_min(1e-6)
                    else:  # VR-HH: move removed text mass onto visual keys, proportionally
                        removed = CTRL.alpha * text_mass.unsqueeze(-1)            # [B,h,rows,1]
                        vis_w = sub * vm
                        vis_mass = vis_w.sum(-1, keepdim=True)
                        n_vis = vm.sum().clamp_min(1)
                        uniform = vm / n_vis
                        share = torch.where(vis_mass > 1e-4, vis_w / vis_mass.clamp_min(1e-6), uniform)
                        new = sub * (1 - CTRL.alpha * tm) + removed * share
                    sub = torch.where(trig, new, sub)
                    attn = attn.clone()
                    attn[:, h_idx, start:] = sub

    attn = attn.to(query.dtype)
    attn = nn.functional.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
    return out, attn


def install(model) -> None:
    """Route the language model's attention through hadl_attention_forward."""
    ALL_ATTENTION_FUNCTIONS[ATTN_NAME] = hadl_attention_forward
    lm = model.language_model
    lm.config._attn_implementation = ATTN_NAME
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = ATTN_NAME


def heads_from_list(pairs) -> Dict[int, List[int]]:
    d: Dict[int, List[int]] = {}
    for layer, head in pairs:
        d.setdefault(int(layer), []).append(int(head))
    return d
