"""Run a list of shell commands on a pool of GPUs (one job per GPU at a time).

Each line of the jobs file is:  <output-marker-file> | <command>
A job is skipped when its marker file already exists, so the pipeline is resumable.
    python scripts/launch.py jobs.txt --gpus 0,1,2,3
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    p.add_argument("--logdir", default="logs")
    args = p.parse_args()
    gpus = [g for g in args.gpus.split(",") if g != ""]
    os.makedirs(args.logdir, exist_ok=True)
    jobs = []
    for line in open(args.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        marker, cmd = [x.strip() for x in line.split("|", 1)]
        if os.path.exists(marker):
            print(f"skip (done): {marker}")
            continue
        jobs.append((marker, cmd))
    running, failed = {}, []
    slots = list(enumerate(gpus))  # GPUS=0,0,0 runs 3 jobs at once on GPU 0
    while jobs or running:
        for slot, g in slots:
            if slot not in running and jobs:
                marker, cmd = jobs.pop(0)
                name = marker.replace("/", "_").strip("_")
                log = open(os.path.join(args.logdir, name + ".log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
                print(f"[gpu {g}] start {marker}", flush=True)
                running[slot] = (subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT),
                              marker, time.time())
        for slot, (proc, marker, t0) in list(running.items()):
            g = gpus[slot]
            if proc.poll() is not None:
                ok = proc.returncode == 0 and os.path.exists(marker)
                print(f"[gpu {g}] {'done' if ok else 'FAILED'} {marker} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if not ok:
                    failed.append(marker)
                del running[slot]
        time.sleep(2)
    if failed:
        print("FAILED jobs (see logs/):", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
