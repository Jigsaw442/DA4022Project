"""Compression-aware TF-HH: fine-tune only the hallucination heads (paper 2, Eq. 3)
while the model sees *compressed* visual tokens.

    L = CE(caption | compressed image, prompt) + lambda * mean_{h in H_hall, t} log A_text_h(t)

Only the query/key rows that belong to the selected heads are updated (gradient
mask), so the delta is tiny and the rest of LLaVA is untouched.
Targets are the five COCO reference captions of each training image joined into
one paragraph (hallucination-free by construction). Training images are drawn
from val2014 and are disjoint from the CHAIR and POPE evaluation images.
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
from hadl.chair import Chair
from hadl.runner import load_heads
from hadl.utils import CAPTION_PROMPT, coco_subset, dump_json, load_image, read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=M.DEFAULT_MODEL)
    p.add_argument("--dtype", default="auto", help="auto = bfloat16 on GPUs that support it, else float32")
    p.add_argument("--heads", required=True)
    p.add_argument("--topk-heads", type=int, default=20)
    p.add_argument("--method", default="hybrid")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.add_argument("--coco-root", default="data/coco/val2014")
    p.add_argument("--ann-dir", default="data/coco/annotations")
    p.add_argument("--pope-file", default="data/pope/coco_pope_adversarial.json")
    p.add_argument("--eval-n", type=int, default=500, help="size of the CHAIR eval subset to exclude")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--max-cap-tokens", type=int, default=160)
    p.add_argument("--include-bos", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    chair = Chair(os.path.join(args.ann_dir, "instances_val2014.json"),
                  os.path.join(args.ann_dir, "captions_val2014.json"))
    caps = {}
    import json
    for a in json.load(open(os.path.join(args.ann_dir, "captions_val2014.json")))["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip().rstrip(".") + ".")
    exclude = {im["file_name"] for im in coco_subset(chair, args.eval_n, 0)}
    if os.path.exists(args.pope_file):
        exclude |= {q["image"] for q in read_jsonl(args.pope_file)}
    n_train = args.steps * args.grad_accum
    train = coco_subset(chair, n_train, seed=args.seed + 1, exclude=exclude)
    print(f"{len(train)} training images (excluded {len(exclude)} eval images)")

    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
    model, proc = M.load(args.model, dtype, False, args.device)
    tok = proc.tokenizer
    heads, _, pairs = load_heads(args.heads, args.topk_heads)
    cfg = model.language_model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads

    trainable, names = [], {}
    lm_named = dict(model.language_model.named_modules())
    for layer, hs in heads.items():
        attn = model.language_model.model.layers[layer].self_attn
        for proj_name in ("q_proj", "k_proj"):
            lin = getattr(attn, proj_name)
            lin.float()
            lin.weight.requires_grad_(True)
            rows = torch.zeros(lin.weight.shape[0], 1, device=lin.weight.device)
            for h in hs:
                hh = h if proj_name == "q_proj" else h // kv_groups
                rows[hh * hd:(hh + 1) * hd] = 1
            lin.weight.register_hook(lambda g, m=rows: g * m)
            trainable.append(lin.weight)
            names[f"model.layers.{layer}.self_attn.{proj_name}.weight"] = lin.weight
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"training {len(pairs)} heads in {len(heads)} layers")

    A.CTRL.include_bos_as_text = args.include_bos
    A.CTRL.mode, A.CTRL.heads, A.CTRL.record = "train", heads, False
    emb = model.get_input_embeddings()
    use_amp = dtype == "bfloat16"
    log = []
    model.language_model.train()
    t0 = time.time()
    it = iter(train)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        ce_acc, reg_acc = 0.0, 0.0
        for _ in range(args.grad_accum):
            im = next(it)
            target = " ".join(caps[im["id"]][:5])
            img = load_image(os.path.join(args.coco_root, im["file_name"]))
            s = M.build_sample(model, proc, img, CAPTION_PROMPT, args.method, args.budget, args.anchor_frac)
            ids = tok(target, add_special_tokens=False, return_tensors="pt").input_ids[0][: args.max_cap_tokens]
            ids = torch.cat([ids, torch.tensor([tok.eos_token_id])]).to(emb.weight.device)
            x = torch.cat([s.embeds, emb(ids).unsqueeze(0)], 1)
            A.CTRL.set_layout(s.vis_start, s.vis_end, s.embeds.shape[1])
            A.CTRL.train_terms.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.language_model(inputs_embeds=x, use_cache=False).logits[0]
            P = s.embeds.shape[1]
            ce = torch.nn.functional.cross_entropy(logits[P - 1:-1].float(), ids)
            reg = torch.cat(A.CTRL.train_terms).float().clamp_min(1e-6).log().mean()
            loss = (ce + args.lam * reg) / args.grad_accum
            loss.backward()
            ce_acc += ce.item() / args.grad_accum; reg_acc += reg.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        log.append({"step": step, "ce": ce_acc, "log_text_attn": reg_acc, "time": time.time() - t0})
        if step % 10 == 0:
            print(log[-1], flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"weights": {k: v.detach().half().cpu() for k, v in names.items()}, "heads": pairs,
                "args": vars(args)}, os.path.join(args.out, "tfhh_delta.pt"))
    dump_json(os.path.join(args.out, "train_log.json"), log)
    print("saved", os.path.join(args.out, "tfhh_delta.pt"))


if __name__ == "__main__":
    main()
