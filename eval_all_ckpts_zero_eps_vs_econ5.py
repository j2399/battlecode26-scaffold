#!/usr/bin/env python3
"""All learner_rl_ckpt* (now EPSILON=0, deterministic argmax) vs econ5,
both team orders, across all TRAIN_MAPS."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, PARALLEL_LOGS, PROJECT_ROOT
from rl_train import TRAIN_MAPS

GAMES_PER_ORDER = 1


def run_matchup(subject, opponent, maps=None, games=GAMES_PER_ORDER):
    maps = maps or TRAIN_MAPS
    matchups_file = PROJECT_ROOT / "rl_eval_matchups.txt"
    matchups_file.write_text(f"{subject},{opponent}\n{opponent},{subject}\n")
    PARALLEL_LOGS.mkdir(parents=True, exist_ok=True)
    for f in PARALLEL_LOGS.glob("*.log"):
        f.unlink()
    for m in maps:
        cmd = [sys.executable, "parallel_run.py", "--matchups", str(matchups_file),
               "--maps", m, "--games", str(games), "--skip-compile"]
        subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    wins, losses = 0, 0
    for log in PARALLEL_LOGS.glob("*.log"):
        _trajs, outcome, _round = collect_from_log(log, subject)
        if outcome is None:
            continue
        if outcome > 0:
            wins += 1
        else:
            losses += 1
    return wins, losses, wins + losses


def main():
    ckpt_dirs = sorted(
        (PROJECT_ROOT / "src").glob("learner_rl_ckpt*"),
        key=lambda p: int(p.name.replace("learner_rl_ckpt", "")),
    )
    checkpoints = [p.name for p in ckpt_dirs]
    print(f"Evaluating {len(checkpoints)} checkpoints (EPSILON=0) vs econ5 "
          f"({len(TRAIN_MAPS)} maps x 2 orders x {GAMES_PER_ORDER} games = "
          f"{len(TRAIN_MAPS)*2*GAMES_PER_ORDER} games each)...\n", flush=True)

    results = []
    for i, ckpt in enumerate(checkpoints, 1):
        wins, losses, decided = run_matchup(ckpt, "econ5")
        rate = wins / decided if decided else 0.0
        results.append((ckpt, wins, losses, decided, rate))
        print(f"[{i}/{len(checkpoints)}] {ckpt} vs econ5: {wins}W {losses}L ({decided} decided) winrate={rate:.3f}", flush=True)

    results.sort(key=lambda r: -r[4])
    print("\n=== Ranked by win rate vs econ5 (EPSILON=0) ===")
    for ckpt, wins, losses, decided, rate in results:
        print(f"  {ckpt}: {wins}W {losses}L winrate={rate:.3f}")

    tw = sum(r[1] for r in results)
    tl = sum(r[2] for r in results)
    print(f"\nOverall across all checkpoints: {tw}W {tl}L winrate={tw/(tw+tl):.3f}")


if __name__ == "__main__":
    main()
