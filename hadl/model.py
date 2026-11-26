"""Loading LLaVA-1.5, building compressed multimodal inputs, and decoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from . import attention as A
from .compress import compress

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
PROMPT_TMPL = "USER: <image>\n{q} ASSISTANT:"


@dataclass
class Sample:
    embeds: torch.Tensor      # [1, T, d]
    input_ids: torch.Tensor   # [T] token ids, with image positions set to -1
    vis_start: int
    vis_end: int
    meta: Dict


def load(model_id: str = DEFAULT_MODEL, dtype: str = "float16", load_4bit: bool = False, device: str = "cuda"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Check `nvidia-smi` and install a torch build "
                           "matching your CUDA driver, or pass --device cpu (very slow).")
    kw = dict(torch_dtype=getattr(torch, dtype), attn_implementation="eager", low_cpu_mem_usage=True)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kw)
    if not load_4bit:
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    prepare(model)
    return model, processor


def prepare(model):
    model.vision_tower.config._attn_implementation = "eager"
    A.install(model)
    for p in model.parameters():
        p.requires_grad_(False)


def image_token_id(model, tokenizer) -> int:
    tid = getattr(model.config, "image_token_index", None)
    if tid is None:
        tid = tokenizer.convert_tokens_to_ids("<image>")
    return int(tid)


@torch.no_grad()
def encode_image(model, processor, image):
    """Returns CLIP patch features [576, C] (pre-projector) and CLS attention [576]."""
    pv = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
    dev = model.vision_tower.device
    pv = pv.to(dev, dtype=model.vision_tower.dtype)
    out = model.vision_tower(pv, output_hidden_states=True, output_attentions=True)
    layer = model.config.vision_feature_layer
    feats = out.hidden_states[layer][0, 1:]
    att = out.attentions[layer][0]          # [heads, 577, 577]
    cls_attn = att[:, 0, 1:].float().mean(0)
    return feats, cls_attn


def build_sample(model, processor, image, question: str, method: str = "none", budget: int = 576,
                 anchor_frac: float = 0.5, image_cache: Optional[tuple] = None) -> Sample:
    tok = processor.tokenizer
    feats, cls_attn = image_cache if image_cache is not None else encode_image(model, processor, image)
    cfeats, meta = compress(feats, cls_attn, method, budget, anchor_frac)
    with torch.no_grad():
        vis = model.multi_modal_projector(cfeats.unsqueeze(0))[0]          # [B, d]
    text = PROMPT_TMPL.format(q=question)
    ids = tok(text, return_tensors="pt").input_ids[0]
    img_id = image_token_id(model, tok)
    pos = (ids == img_id).nonzero()
    assert len(pos) == 1, "prompt must contain exactly one <image> token"
    p = int(pos[0])
    emb_layer = model.get_input_embeddings()
    dev = emb_layer.weight.device
    with torch.no_grad():
        pre = emb_layer(ids[:p].to(dev))
        post = emb_layer(ids[p + 1:].to(dev))
    embeds = torch.cat([pre, vis.to(pre.dtype), post], 0).unsqueeze(0)
    full_ids = torch.cat([ids[:p], torch.full((vis.shape[0],), -1, dtype=ids.dtype), ids[p + 1:]])
    meta.update({"budget": int(vis.shape[0]), "method": method})
    return Sample(embeds, full_ids, p, p + vis.shape[0], meta)


def _set_layout(s: Sample):
    A.CTRL.set_layout(s.vis_start, s.vis_end, s.embeds.shape[1])


@torch.no_grad()
def greedy_generate(model, s: Sample, max_new_tokens: int = 128, eos_id: Optional[int] = None) -> List[int]:
    lm = model.language_model
    emb_layer = model.get_input_embeddings()
    _set_layout(s)
    out = lm(inputs_embeds=s.embeds, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    toks: List[int] = []
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new_tokens):
        if eos_id is not None and nxt == eos_id:
            break
        toks.append(nxt)
        e = emb_layer(torch.tensor([[nxt]], device=emb_layer.weight.device))
        out = lm(inputs_embeds=e, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    return toks


@torch.no_grad()
def next_token_logits(model, s: Sample) -> torch.Tensor:
    _set_layout(s)
    out = model.language_model(inputs_embeds=s.embeds, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()


def load_delta(model, path: str):
    """Load q/k weights produced by scripts/train_tfhh.py."""
    sd = torch.load(path, map_location="cpu")
    params = dict(model.language_model.named_parameters())
    for k, v in sd["weights"].items():
        params[k].data.copy_(v.to(params[k].dtype))
    return sd.get("heads")
