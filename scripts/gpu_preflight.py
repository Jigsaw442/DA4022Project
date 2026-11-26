"""5-minute check on the real GPU machine, before the long run.

Exercises every GPU code path once with the real LLaVA-1.5-7B and prints PASS/FAIL per step,
peak memory and timings:
  1. CUDA, GPU, bf16, library versions
  2. model load (fp16, eager) + our compressed input path == HF LLaVA logits (576 tokens)
  3. caption generation: greedy, AD-HH and VR-HH at 64 tokens
  4. POPE-style single forward
  5. attribution backward (find_heads path)
  6. one TF-HH optimisation step (train_tfhh.py, bf16), unless --skip-train
Usage (after scripts/setup_data.sh):
  python scripts/gpu_preflight.py
"""
import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

RESULTS = []


def step(name):
    def deco(fn):
        def run(*a, **k):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                out = fn(*a, **k)
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                RESULTS.append((name, "PASS", f"{time.time() - t0:.1f}s, peak {mem:.1f} GB"))
                print(f"[PASS] {name} ({time.time() - t0:.1f}s, peak {mem:.1f} GB)", flush=True)
                return out
            except Exception as e:
                traceback.print_exc()
                RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"[:300]))
                print(f"[FAIL] {name}: {e}", flush=True)
                return None
        return run
    return deco


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--data", default="data")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)  # cpu only for self-tests
    args = p.parse_args()

    import transformers
    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), transformers {transformers.__version__}")
    dev = args.device
    dtype = "float16" if dev == "cuda" else "float32"
    if dev == "cuda" and not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False. Install a torch build that matches your driver "
              "(see `nvidia-smi`), e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {pr.name}, {pr.total_memory / 2**30:.0f} GB")
    if dev == "cuda":
        print("bf16 supported:", torch.cuda.is_bf16_supported())
    if transformers.__version__ != "4.51.3":
        print("WARNING: tested with transformers==4.51.3; other versions may change LLaVA internals")

    imgs = sorted(glob.glob(os.path.join(args.data, "coco/val2014/*.jpg")))[:2]
    assert imgs, f"no images in {args.data}/coco/val2014 - run scripts/setup_data.sh first"

    from hadl import attention as A
    from hadl import model as M
    from hadl.attribution import HeadGrabber, attribute_one
    from hadl.utils import CAPTION_PROMPT, load_image

    st = {}

    @step("load model (fp16, eager) on cuda")
    def s_load():
        st["model"], st["proc"] = M.load(args.model, dtype, False, dev)
    s_load()
    if "model" not in st:
        return report()
    model, proc = st["model"], st["proc"]
    img = load_image(imgs[0])

    @step("our input path matches HF LLaVA logits (576 tokens)")
    def s_equiv():
        inputs = proc(images=img, text=M.PROMPT_TMPL.format(q=CAPTION_PROMPT), return_tensors="pt").to(dev)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        model.language_model.config._attn_implementation = "eager"
        with torch.no_grad():
            ref = model(**inputs).logits[0, -1].float()
        A.install(model)
        A.CTRL.mode, A.CTRL.record, A.CTRL.heads = "off", False, {}
        ours = M.next_token_logits(model, M.build_sample(model, proc, img, CAPTION_PROMPT, "none", 576))
        err = (ref - ours).abs().max().item()
        same = ref.argmax().item() == ours.argmax().item()
        print(f"    max |logit diff| = {err:.4f}, same top token: {same}")
        assert same and err < 0.5, "logits differ too much"
    s_equiv()

    nl = model.language_model.config.num_hidden_layers
    heads = {nl // 2: [0, 1, 2, 3], (3 * nl) // 4: [1, 2]}

    @step("caption generation: greedy / AD-HH / VR-HH at 64 tokens")
    def s_gen():
        eos = proc.tokenizer.eos_token_id
        s = M.build_sample(model, proc, img, CAPTION_PROMPT, "hybrid", 64)
        for mode in ("off", "adhh", "vrhh"):
            A.CTRL.mode, A.CTRL.heads, A.CTRL.tau, A.CTRL.record = mode, heads if mode != "off" else {}, 0.05, True
            A.CTRL.reset_stats()
            t0 = time.time()
            toks = M.greedy_generate(model, s, 64, eos)
            cap = proc.tokenizer.decode(toks, skip_special_tokens=True)
            dt = time.time() - t0
            print(f"    {mode:5s} {len(toks)} tok in {dt:.1f}s ({len(toks) / dt:.1f} tok/s): {cap[:110]}")
            assert len(toks) > 5 and cap.strip(), "empty caption"
            st["cap"] = cap.strip() if mode == "off" else st.get("cap")
            st["tok_s"] = len(toks) / dt
        A.CTRL.mode, A.CTRL.heads = "off", {}
    s_gen()

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from pope_eval import yes_no_ids

        @step("POPE single forward (Yes/No logits)")
        def s_pope2():
            y, n = yes_no_ids(proc.tokenizer)
            s = M.build_sample(model, proc, img, "Is there a person in the image? Answer the question using a single word or phrase.", "hybrid", 64)
            lg = M.next_token_logits(model, s)
            top = proc.tokenizer.decode([int(lg.argmax())])
            print(f"    yes ids {y}, no ids {n}, top token {top!r}")
            assert top.strip().lower() in ("yes", "no"), f"model did not answer yes/no: {top!r}"
        s_pope2()
    except ImportError as e:
        RESULTS.append(("POPE single forward", "FAIL", str(e)))

    @step("attribution backward (find_heads path)")
    def s_attr():
        grab = HeadGrabber(model)
        cap = st.get("cap") or "A man is standing next to a table with a dog."
        words = [w for w in cap.split() if w.isalpha()]
        a, b = words[1], words[3]
        hs = [{"cls": "x", "start": cap.index(a), "end": cap.index(a) + len(a)}]
        gs = [{"cls": "y", "start": cap.index(b), "end": cap.index(b) + len(b)}]
        s = M.build_sample(model, proc, img, CAPTION_PROMPT, "hybrid", 64)
        A.CTRL.record = True
        res = attribute_one(model, grab, s, proc.tokenizer, cap, hs, gs)
        grab.remove()
        assert res is not None and torch.isfinite(res[0]).all() and res[0].abs().sum() > 0, "bad head scores"
        print(f"    head score tensor {tuple(res[0].shape)}, finite")
    s_attr()

    del model, st["model"]
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    if not args.skip_train:
        @step("one TF-HH training step (train_tfhh.py, bf16)")
        def s_train():
            hj = "/tmp/hadl_preflight_heads.json"
            json.dump({"top": [[l, h] for l, hs in heads.items() for h in hs][:3], "text_mass": [[0.0] * 64 for _ in range(64)]}, open(hj, "w"))
            cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train_tfhh.py"), "--model", args.model,
                   "--heads", hj, "--topk-heads", "3", "--steps", "1", "--grad-accum", "2",
                   "--coco-root", f"{args.data}/coco/val2014", "--ann-dir", f"{args.data}/coco/annotations",
                   "--pope-file", f"{args.data}/pope/coco_pope_adversarial.json", "--out", "/tmp/hadl_preflight_tfhh",
                   "--device", dev] + ([] if dev == "cuda" else ["--eval-n", "5"])
            r = subprocess.run(cmd, capture_output=True, text=True)
            print("    " + "\n    ".join(r.stdout.strip().splitlines()[-3:]))
            assert r.returncode == 0, r.stderr[-1500:]
        s_train()
    report(st.get("tok_s"))


def report(tok_s=None):
    print("\n==== PREFLIGHT SUMMARY ====")
    for n, s, d in RESULTS:
        print(f"{s:4s}  {n}  [{d}]")
    if tok_s:
        print(f"\nDecoding speed ~{tok_s:.0f} tok/s at 64 tokens -> ~{128 / tok_s:.1f}s per caption")
    ok = all(s == "PASS" for _, s, _ in RESULTS)
    print("\nALL PASSED - safe to start scripts/run_all.sh" if ok else "\nSOME CHECKS FAILED - send this output")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
