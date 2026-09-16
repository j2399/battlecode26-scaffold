#!/usr/bin/env python3
"""
Screens every learner_rl_ckpt* self-checkpoint against the full econ5 bot
(both team orders, across all 8 TRAIN_MAPS), ranks them by win rate, then
runs the top 2 head-to-head against the final trained checkpoint
(learner_rl) -- same map/order coverage -- to see whether the run ended
on its strongest point or peaked somewhere earlier along the way.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, PARALLEL_LOGS, PROJECT_ROOT
from rl_train import TRAIN_MAPS

GAMES_PER_ORDER = 1


def run_matchup(subject: str, opponent: str, maps=None, games: int = GAMES_PER_ORDER):
    """subject vs opponent, both team orders, across maps. Returns
    (wins, losses, decided) from subject's perspective."""
    maps = maps or TRAIN_MAPS
    matchups_file = PROJECT_ROOT / "rl_eval_matchups.txt"
    matchups_file.write_text(f"{subject},{opponent}\n{opponent},{subject}\n")

    PARALLEL_LOGS.mkdir(parents=True, exist_ok=True)
    for f in PARALLEL_LOGS.glob("*.log"):
        f.unlink()

    for m in maps:
        cmd = [
            sys.executable, "parallel_run.py",
            "--matchups", str(matchups_file),
            "--maps", m,
            "--games", str(games),
            "--skip-compile",
        ]
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
    print(f"Screening {len(checkpoints)} self-checkpoints vs econ5 "
          f"({len(TRAIN_MAPS)} maps x 2 orders x {GAMES_PER_ORDER} games = "
          f"{len(TRAIN_MAPS)*2*GAMES_PER_ORDER} games each)...\n", flush=True)

    results = []
    for i, ckpt in enumerate(checkpoints, 1):
        wins, losses, decided = run_matchup(ckpt, "econ5")
        rate = wins / decided if decided else 0.0
        results.append((ckpt, wins, losses, decided, rate))
        print(f"[{i}/{len(checkpoints)}] {ckpt} vs econ5: {wins}W {losses}L ({decided} decided) winrate={rate:.3f}", flush=True)

    results.sort(key=lambda r: -r[4])
    print("\n=== Ranked by win rate vs econ5 ===")
    for ckpt, wins, losses, decided, rate in results:
        print(f"  {ckpt}: {wins}W {losses}L winrate={rate:.3f}")

    top2 = results[:2]
    print(f"\n=== Top 2: {top2[0][0]} and {top2[1][0]} ===\n")

    print("Running top 2 vs learner_rl (final trained checkpoint), full map coverage...\n", flush=True)
    for ckpt, _w, _l, _d, rate in top2:
        wins, losses, decided = run_matchup(ckpt, "learner_rl")
        rate2 = wins / decided if decided else 0.0
        print(f"{ckpt} (winrate {rate:.3f} vs econ5) vs learner_rl (final): "
              f"{wins}W {losses}L ({decided} decided) winrate={rate2:.3f}", flush=True)


if __name__ == "__main__":
    main()
