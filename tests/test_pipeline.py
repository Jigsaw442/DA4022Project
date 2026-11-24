"""Correctness checks on the tiny model (CPU):
1. our embedding splice with 576 tokens == HF LlavaForConditionalGeneration logits,
2. 'hadl' attention with mode off == eager attention,
3. AD-HH / VR-HH keep rows valid and actually change the selected heads,
4. compressors return the requested number of tokens.
Usage: python tests/test_pipeline.py /tmp/hadl_tiny
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hadl import attention as A
from hadl import model as M
from hadl.compress import compress
from hadl.utils import load_image


def main(root):
    model, proc = M.load(os.path.join(root, "model"), "float32", False, "cpu")
    img = load_image(os.path.join(root, "coco/val2014/COCO_val2014_000000000000.jpg"))
    q = "Please describe this image in detail."

    # 1. reference: HF native path (expands <image> to 576 tokens)
    text = M.PROMPT_TMPL.format(q=q)
    inputs = proc(images=img, text=text, return_tensors="pt")
    model.language_model.config._attn_implementation = "eager"
    with torch.no_grad():
        ref = model(**inputs).logits[0, -1]
    A.install(model)
    A.CTRL.mode, A.CTRL.record, A.CTRL.heads = "off", False, {}
    s = M.build_sample(model, proc, img, q, "none", 576)
    ours = M.next_token_logits(model, s)
    err = (ref - ours).abs().max().item()
    print("max |HF - ours| logits:", err)
    assert err < 1e-4, err

    # 2/3. interventions
    A.CTRL.mode, A.CTRL.heads, A.CTRL.tau = "adhh", {1: [0, 2], 3: [1]}, 0.0
    A.CTRL.reset_stats()
    l_ad = M.next_token_logits(model, s)
    assert A.CTRL.trigger_count > 0 and (l_ad - ours).abs().max() > 0
    A.CTRL.mode = "vrhh"
    l_vr = M.next_token_logits(model, s)
    assert (l_vr - ours).abs().max() > 0 and (l_vr - l_ad).abs().max() > 0
    A.CTRL.tau = 2.0  # never triggers -> identical
    A.CTRL.reset_stats()
    assert (M.next_token_logits(model, s) - ours).abs().max() < 1e-5 and A.CTRL.trigger_count == 0
    # VR-HH rows still sum to 1: check directly
    q_ = torch.randn(1, 4, 3, 8); k_ = torch.randn(1, 4, 10, 8); v_ = torch.randn(1, 4, 10, 8)

    class Dummy(torch.nn.Module):
        num_key_value_groups, layer_idx = 1, 1
    A.CTRL.set_layout(2, 6, 8); A.CTRL.mode = "vrhh"; A.CTRL.tau = 0.0; A.CTRL.heads = {1: [0, 3]}
    _, w = A.hadl_attention_forward(Dummy(), q_, k_, v_, None, 1.0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 4, 3), atol=1e-5)
    text_cols = [1, 6, 7, 8, 9]
    assert w[0, [0, 3]][:, :, text_cols].abs().max() < 1e-6  # alpha=1 moves all text mass
    A.CTRL.mode, A.CTRL.heads = "off", {}

    # 4. compressors
    feats, cls = M.encode_image(model, proc, img)
    for m in ("pool", "topk", "hybrid"):
        for b in (144, 64, 16):
            out, meta = compress(feats, cls, m, b)
            assert out.shape[0] == b, (m, b, out.shape)
    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
