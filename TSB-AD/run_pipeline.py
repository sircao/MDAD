"""run_pipeline.py — One-stop: ① score generation → ② Tuning learns parameters → ③ Eva zero-shot evaluation.

Usage:
    python run_pipeline.py --workers 2 --max-len 25000

The three steps run in sequence, each writing dashboard data (score_progress / route_progress) in real time,
and tpfp_report.html refreshes and stays visible throughout. Any step failing aborts the run.

Note: step ① (score generation) is the longest (remote CHARM + Eva 350×4, possibly hours);
②③ are fast (minutes each). If scores are already generated, add --skip-scores to skip ①.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(name, argv):
    print(f"\n{'=' * 62}\n  {name}\n{'=' * 62}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + argv, cwd=str(ROOT))
    print(f"  [{name}] done, took {time.time() - t0:.0f}s, exit code {r.returncode}", flush=True)
    if r.returncode != 0:
        print(f"  !! {name} failed (exit code {r.returncode}), pipeline aborted.", flush=True)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="multiprocessing count per step (default 2, to avoid OOM")
    ap.add_argument("--max-len", type=int, default=25000,
                    help="skip files exceeding this sample-point count (VUS-PR is slow")
    ap.add_argument("--skip-scores", action="store_true",
                    help="scores already generated, skip step ①")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_scores:
        step("① score generation (local first, resume)",
             ["gen_local_scores.py", "--workers", str(args.workers)])
    step("② Tuning learns parameters (morphology+source two-level routing)",
         ["eval_loop_route.py", "--split", "tuning",
          "--max-len", str(args.max_len), "--workers", str(args.workers)])
    step("③ Eva zero-shot evaluation (no label leakage)",
         ["eval_loop_route.py", "--split", "eva",
          "--max-len", str(args.max_len), "--workers", str(args.workers)])

    print(f"\n✅ all done, total time {time.time() - t0:.0f}s. Refresh tpfp_report.html to see the final results.", flush=True)


if __name__ == "__main__":
    main()
