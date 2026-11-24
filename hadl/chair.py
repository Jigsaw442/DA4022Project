"""Self-contained CHAIR metric (Rohrbach et al., 2018) for COCO captions.

Ground-truth objects of an image = its COCO instance categories plus objects
mentioned in its five reference captions (as in the original CHAIR).
Mentions are de-duplicated per caption by COCO class.
CHAIR_s = fraction of captions with >= 1 hallucinated object.
CHAIR_i = hallucinated object mentions / all object mentions.
Recall  = fraction of ground-truth instance classes that are mentioned.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# COCO class -> synonyms (lower case; multi-word allowed). Condensed from the CHAIR synonym list.
SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "girl", "boy", "man", "woman", "kid", "child", "chef", "baker", "people", "adult",
               "rider", "children", "baby", "worker", "passenger", "sister", "biker", "policeman", "cop",
               "officer", "lady", "cowboy", "bride", "groom", "male", "female", "guy", "traveler", "mother",
               "father", "gentleman", "pitcher", "player", "skier", "snowboarder", "skater", "skateboarder",
               "surfer", "batter", "catcher", "umpire", "teenager", "toddler", "pedestrian", "athlete",
               "tourist", "student", "teen", "infant", "spectator", "crowd", "daughter", "son", "brother",
               "friend", "couple", "family", "audience", "customer", "vendor", "soldier", "fireman"],
    "bicycle": ["bicycle", "bike", "unicycle", "minibike", "trike", "cycle"],
    "car": ["car", "automobile", "van", "minivan", "sedan", "suv", "hatchback", "cab", "jeep", "coupe",
            "taxicab", "limo", "taxi"],
    "motorcycle": ["motorcycle", "scooter", "motor bike", "motor cycle", "motorbike", "moped"],
    "airplane": ["airplane", "jetliner", "plane", "air plane", "monoplane", "aircraft", "jet", "airbus",
                 "biplane", "seaplane", "airliner"],
    "bus": ["bus", "minibus", "trolley"],
    "train": ["train", "locomotive", "tramway", "caboose", "tram"],
    "truck": ["truck", "pickup", "lorry", "hauler", "firetruck"],
    "boat": ["boat", "ship", "liner", "sailboat", "motorboat", "dinghy", "powerboat", "speedboat", "canoe",
             "skiff", "yacht", "kayak", "catamaran", "pontoon", "houseboat", "vessel", "rowboat", "trawler",
             "ferryboat", "watercraft", "tugboat", "schooner", "barge", "ferry", "sailboard", "paddleboat",
             "lifeboat", "freighter", "steamboat", "riverboat", "battleship", "steamship"],
    "traffic light": ["traffic light", "traffic signal", "stop light", "streetlight", "traffic lamp"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench", "pew"],
    "bird": ["bird", "ostrich", "owl", "seagull", "goose", "duck", "parakeet", "falcon", "robin", "pelican",
             "waterfowl", "heron", "hummingbird", "mallard", "finch", "pigeon", "sparrow", "seabird", "osprey",
             "blackbird", "fowl", "shorebird", "woodpecker", "egret", "chickadee", "quail", "bluebird",
             "kingfisher", "buzzard", "willet", "gull", "swan", "bluejay", "flamingo", "cormorant", "parrot",
             "loon", "gosling", "waterbird", "pheasant", "rooster", "sandpiper", "crow", "raven", "turkey",
             "oriole", "cowbird", "warbler", "magpie", "peacock", "cockatiel", "lorikeet", "puffin", "vulture",
             "condor", "macaw", "peafowl", "cockatoo", "songbird", "chicken", "hen", "eagle", "hawk"],
    "cat": ["cat", "kitten", "feline", "tabby", "kitty"],
    "dog": ["dog", "puppy", "beagle", "pup", "chihuahua", "schnauzer", "dachshund", "rottweiler", "canine",
            "pitbull", "collie", "pug", "terrier", "poodle", "labrador", "doggie", "doberman", "mutt",
            "doggy", "spaniel", "bulldog", "sheepdog", "weimaraner", "corgi", "cocker", "greyhound",
            "retriever", "brindle", "hound", "whippet", "husky"],
    "horse": ["horse", "colt", "pony", "racehorse", "stallion", "equine", "mare", "foal", "palomino",
              "mustang", "clydesdale", "bronc", "bronco"],
    "sheep": ["sheep", "lamb", "ram", "goat", "ewe"],
    "cow": ["cow", "cattle", "oxen", "ox", "calf", "holstein", "heifer", "buffalo", "bull", "zebu", "bison"],
    "elephant": ["elephant"],
    "bear": ["bear", "grizzly"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "wallet", "purse", "briefcase"],
    "tie": ["tie", "necktie", "bow tie"],
    "suitcase": ["suitcase", "suit case", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball", "baseball", "football", "soccer", "basketball", "softball",
                    "volleyball", "pinball", "fastball", "racquetball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "longboard", "skimboard", "shortboard", "wakeboard", "surf board"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "wineglass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife", "pocketknife", "knives"],
    "spoon": ["spoon"],
    "bowl": ["bowl", "container"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "sub", "cheeseburger", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut", "bagel"],
    "cake": ["cake", "cheesecake", "cupcake", "shortcake", "coffeecake", "pancake"],
    "chair": ["chair", "seat", "stool"],
    "couch": ["couch", "sofa", "recliner", "futon", "loveseat", "settee", "chesterfield"],
    "potted plant": ["potted plant", "houseplant", "plant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet", "urinal", "commode", "lavatory", "potty"],
    "tv": ["tv", "monitor", "televison", "television"],
    "laptop": ["laptop", "computer", "notebook", "netbook", "lenovo", "macbook", "laptop computer"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "mobile phone", "phone", "cellphone", "telephone", "phon", "smartphone",
                   "iphone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stovetop", "stove", "stove top oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book": ["book", "novel"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddybear"],
    "hair drier": ["hair drier", "hairdryer", "hair dryer", "blow dryer", "blowdryer", "blow drier", "dryer"],
    "toothbrush": ["toothbrush"],
}

IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child", "mice": "mouse",
             "geese": "goose", "knives": "knife", "teeth": "tooth", "feet": "foot", "oxen": "ox",
             "skis": "skis", "scissors": "scissors", "buses": "bus", "glasses": "glass",
             "benches": "bench", "sandwiches": "sandwich", "couches": "couch", "boxes": "box",
             "dishes": "dish", "peaches": "peach", "ponies": "pony", "puppies": "puppy", "kitties": "kitty",
             "babies": "baby", "ladies": "lady", "cherries": "cherry", "calves": "calf", "wolves": "wolf",
             "leaves": "leaf", "shelves": "shelf"}

# phrases where a class word does not denote that object
BLOCK = {"baby bird": "bird", "baby elephant": "elephant",
         "baby giraffe": "giraffe", "baby zebra": "zebra", "baby bear": "bear", "baby cow": "cow",
         "baby horse": "horse", "baby sheep": "sheep", "baby cat": "cat", "baby dog": "dog",
         "adult giraffe": "giraffe", "adult elephant": "elephant", "adult zebra": "zebra",
         "bow tie": "tie", "toilet seat": "toilet", "wine glass": "wine glass", "cell phone": "cell phone",
         "dining table": "dining table", "computer mouse": "mouse", "tennis ball": "sports ball",
         "passenger train": "train", "train car": "train", "fire truck": "truck", "car seat": None,
         "orange juice": None, "apple juice": None}

WORD2CLS: Dict[str, str] = {}
for _cls, _syns in SYNONYMS.items():
    for _s in _syns:
        WORD2CLS.setdefault(_s, _cls)
for _ph, _cls in BLOCK.items():
    WORD2CLS[_ph] = _cls  # None means "not an object mention"


def singular(w: str) -> str:
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in WORD2CLS:
        return w
    for suf, rep in (("ies", "y"), ("ves", "f"), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            cand = w[: -len(suf)] + rep
            if cand in WORD2CLS:
                return cand
    return w


_tok_re = re.compile(r"[a-zA-Z]+")


def mentions(caption: str) -> List[Tuple[str, int, int]]:
    """Return list of (coco_class, char_start, char_end) for object mentions in the caption."""
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in _tok_re.finditer(caption)]
    out = []
    i = 0
    while i < len(toks):
        hit = None
        for n in (3, 2):  # multi-word phrases first
            if i + n <= len(toks):
                ph = " ".join(singular(t[0]) if k == n - 1 else t[0] for k, t in enumerate(toks[i:i + n]))
                if ph in WORD2CLS:
                    hit = (WORD2CLS[ph], toks[i][1], toks[i + n - 1][2], n)
                    break
        if hit is None:
            w = singular(toks[i][0])
            if w in WORD2CLS:
                hit = (WORD2CLS[w], toks[i][1], toks[i][2], 1)
        if hit is not None:
            if hit[0] is not None:
                out.append(hit[:3])
            i += hit[3]
        else:
            i += 1
    return out


class Chair:
    def __init__(self, instances_json: str, captions_json: str):
        inst = json.load(open(instances_json))
        caps = json.load(open(captions_json))
        id2name = {c["id"]: c["name"] for c in inst["categories"]}
        self.gt_inst: Dict[int, Set[str]] = defaultdict(set)
        for a in inst["annotations"]:
            self.gt_inst[a["image_id"]].add(id2name[a["category_id"]])
        self.gt_all: Dict[int, Set[str]] = defaultdict(set)
        for k, v in self.gt_inst.items():
            self.gt_all[k] |= v
        for a in caps["annotations"]:
            for c, _, _ in mentions(a["caption"]):
                self.gt_all[a["image_id"]].add(c)
        self.file2id = {im["file_name"]: im["id"] for im in inst["images"]}
        self.images = inst["images"]

    def score_caption(self, image_id: int, caption: str) -> Dict:
        ms = mentions(caption)
        gt = self.gt_all.get(image_id, set())
        seen, objs, hall = set(), [], []
        for c, s, e in ms:
            if c in seen:
                continue
            seen.add(c)
            objs.append(c)
            if c not in gt:
                hall.append({"cls": c, "start": s, "end": e, "word": caption[s:e]})
        # spans of all non-hallucinated mentions (used for contrastive attribution)
        good_spans = [{"cls": c, "start": s, "end": e} for c, s, e in ms if c in gt]
        hall_spans = [{"cls": c, "start": s, "end": e} for c, s, e in ms if c not in gt]
        return {"objects": objs, "hallucinated": [h["cls"] for h in hall], "hall_spans": hall_spans,
                "good_spans": good_spans, "gt_inst": sorted(self.gt_inst.get(image_id, set()))}

    @staticmethod
    def aggregate(scored: List[Dict], captions: List[str]) -> Dict:
        n = len(scored)
        n_obj = sum(len(s["objects"]) for s in scored)
        n_hal = sum(len(s["hallucinated"]) for s in scored)
        n_cap_h = sum(1 for s in scored if s["hallucinated"])
        rec_hit = sum(len(set(s["objects"]) & set(s["gt_inst"])) for s in scored)
        rec_tot = sum(len(s["gt_inst"]) for s in scored)
        return {"n": n, "CHAIR_s": 100 * n_cap_h / max(n, 1), "CHAIR_i": 100 * n_hal / max(n_obj, 1),
                "recall": 100 * rec_hit / max(rec_tot, 1),
                "avg_len_words": sum(len(c.split()) for c in captions) / max(n, 1),
                "objs_per_cap": n_obj / max(n, 1)}
