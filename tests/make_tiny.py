"""Build a tiny random LLaVA + fake COCO/POPE data so the whole pipeline can be
smoke-tested on CPU in a couple of minutes (no downloads). Usage:
    python tests/make_tiny.py /tmp/hadl_tiny
"""
import json
import os
import random
import sys

import numpy as np
from PIL import Image


def main(root):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
    from transformers import (CLIPImageProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaProcessor, PreTrainedTokenizerFast)
    torch.manual_seed(0)
    os.makedirs(root, exist_ok=True)
    words = ("USER : ASSISTANT Please describe this image in detail . Is there a an the in ? Answer question "
             "using single word or phrase Yes No yes no dog cat person man table car bus pizza tree sky "
             "sitting on with and of is are two red blue next to").split()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<image>": 3, "<pad>": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tk.decoder = decoders.WordPiece(prefix="##")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  pad_token="<pad>", additional_special_tokens=["<image>"])
    vcfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                            image_size=336, patch_size=14, projection_dim=32)
    tcfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                       bos_token_id=1, eos_token_id=2, pad_token_id=4)
    cfg = LlavaConfig(vision_config=vcfg, text_config=tcfg, image_token_index=3, vision_feature_layer=-2,
                      vision_feature_select_strategy="default", image_seq_length=576)
    model = LlavaForConditionalGeneration(cfg)
    model.save_pretrained(os.path.join(root, "model"))
    ip = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336})
    LlavaProcessor(image_processor=ip, tokenizer=tok, patch_size=14,
                   vision_feature_select_strategy="default", num_additional_image_tokens=1).save_pretrained(os.path.join(root, "model"))

    # fake COCO val2014
    img_dir = os.path.join(root, "coco", "val2014"); os.makedirs(img_dir, exist_ok=True)
    ann_dir = os.path.join(root, "coco", "annotations"); os.makedirs(ann_dir, exist_ok=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(["person", "dog", "cat", "dining table", "car", "bus", "pizza"])]
    images, inst, capa = [], [], []
    rng = random.Random(0)
    for i in range(40):
        fn = f"COCO_val2014_{i:012d}.jpg"
        Image.fromarray((np.random.rand(240, 320, 3) * 255).astype("uint8")).save(os.path.join(img_dir, fn))
        images.append({"id": i, "file_name": fn})
        present = rng.sample(cats, 2)
        for c in present:
            inst.append({"image_id": i, "category_id": c["id"]})
        for k in range(5):
            capa.append({"image_id": i, "caption": f"a {present[0]['name']} next to a {present[1]['name']}"})
    json.dump({"images": images, "annotations": inst, "categories": cats}, open(os.path.join(ann_dir, "instances_val2014.json"), "w"))
    json.dump({"images": images, "annotations": capa}, open(os.path.join(ann_dir, "captions_val2014.json"), "w"))
    os.makedirs(os.path.join(root, "pope"), exist_ok=True)
    with open(os.path.join(root, "pope", "coco_pope_adversarial.json"), "w") as f:
        q = 0
        for im in images[:10]:
            for obj, lab in (("dog", "yes"), ("car", "no")):
                q += 1
                f.write(json.dumps({"question_id": q, "image": im["file_name"],
                                    "text": f"Is there a {obj} in the image?", "label": lab}) + "\n")
    # fake captions with both correct and hallucinated objects (for find_heads)
    with open(os.path.join(root, "fake_captions.jsonl"), "w") as f:
        for im in images[:6]:
            cap = "a man sitting on a table with a dog and a pizza ."
            hs = [{"cls": "pizza", "start": cap.index("pizza"), "end": cap.index("pizza") + 5}]
            gs = [{"cls": "person", "start": 2, "end": 5}]
            f.write(json.dumps({"image": im["file_name"], "image_id": im["id"], "caption": cap,
                                "hall_spans": hs, "good_spans": gs}) + "\n")
    print("tiny assets in", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hadl_tiny")
