#!/usr/bin/env python3
"""Runs a fixed list of checkpoints vs learner_rl (final) and vs econ5,
both team orders, across all TRAIN_MAPS, and reports win rates."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, PARALLEL_LOGS, PROJECT_ROOT
from rl_train import TRAIN_MAPS

GAMES_PER_ORDER = 1
CHECKPOINTS = ["learner_rl_ckpt45", "learner_rl_ckpt80", "learner_rl_ckpt120", "learner_rl_ckpt160", "learner_rl_ckpt200"]
OPPONENTS = ["learner_rl", "econ5"]


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
    for ckpt in CHECKPOINTS:
        for opp in OPPONENTS:
            wins, losses, decided = run_matchup(ckpt, opp)
            rate = wins / decided if decided else 0.0
            print(f"{ckpt} vs {opp}: {wins}W {losses}L ({decided} decided) winrate={rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
