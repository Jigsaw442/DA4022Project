from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List

from PIL import Image

CAPTION_PROMPT = "Please describe this image in detail."
POPE_SUFFIX = " Answer the question using a single word or phrase."


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def dump_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def coco_subset(chair, n: int, seed: int = 0, exclude: set | None = None, offset: int = 0) -> List[Dict]:
    """Deterministic subset of val2014 images that have instance annotations."""
    ims = [im for im in chair.images if im["id"] in chair.gt_inst]
    ims.sort(key=lambda im: im["id"])
    random.Random(seed).shuffle(ims)
    if exclude:
        ims = [im for im in ims if im["file_name"] not in exclude]
    return ims[offset: offset + n]


def config_name(method: str, budget: int, decode: str) -> str:
    return f"{method}{budget}_{decode}"
