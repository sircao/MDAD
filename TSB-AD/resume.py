"""resume.py —— One-click resume of score generation + start EVA live display.

Two processes run in parallel:
  ① gen_local_scores.py —— resume (already-computed files are skipped, oversized files are auto-downsampled)
  ② eval_loop_route.py --split eva --watch —— compute EVA VUS-PR in real time while scores are generated

Usage:
    python resume.py

Open tpfp_report.html in a browser to see live progress (summary cards + compute progress bar + per-file table).
Ctrl+C stops both processes.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print("① starting score generation (resume + oversized-file downsampling)", flush=True)
    score = subprocess.Popen(
        [sys.executable, "gen_local_scores.py", "--workers", "4", "--max-len", "25000"],
        cwd=str(ROOT))

    time.sleep(2)

    print("② starting EVA live display (watch)", flush=True)
    eva = subprocess.Popen(
        [sys.executable, "eval_loop_route.py", "--split", "eva", "--watch",
         "--interval", "10", "--workers", "2"],
        cwd=str(ROOT))

    print("\ntwo processes started. Open tpfp_report.html in a browser to see live progress. Ctrl+C to stop.", flush=True)

    try:
        score.wait()
        print("\nscore generation finished. EVA watch continues evaluating remaining files, Ctrl+C to stop.", flush=True)
        eva.wait()
    except KeyboardInterrupt:
        print("\nstopping all processes.", flush=True)
        score.terminate()
        eva.terminate()


if __name__ == "__main__":
    main()
