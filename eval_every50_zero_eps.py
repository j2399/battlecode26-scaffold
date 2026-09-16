#!/usr/bin/env python3
"""Every 50th checkpoint (50-1950) plus the final one (1965), all at
EPSILON=0, vs econ5 and vs the last two checkpoints (1965, 1960, also
EPSILON=0), both team orders, across all TRAIN_MAPS."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, PARALLEL_LOGS, PROJECT_ROOT
from rl_train import TRAIN_MAPS

GAMES_PER_ORDER = 1
TEST_SET = [f"learner_rl_ckpt{n}" for n in range(400, 2000, 400)] + ["learner_rl_ckpt1965"]
REFERENCE_OPPONENTS = ["econ5", "learner_rl_ckpt1965", "learner_rl_ckpt1960"]


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
    results = {opp: [] for opp in REFERENCE_OPPONENTS}
    for i, ckpt in enumerate(TEST_SET, 1):
        for opp in REFERENCE_OPPONENTS:
            if ckpt == opp:
                continue
            wins, losses, decided = run_matchup(ckpt, opp)
            rate = wins / decided if decided else 0.0
            results[opp].append((ckpt, wins, losses, decided, rate))
            print(f"[{i}/{len(TEST_SET)}] {ckpt} vs {opp}: {wins}W {losses}L ({decided} decided) winrate={rate:.3f}", flush=True)

    print("\n=== Summary per opponent ===")
    for opp, rows in results.items():
        tw = sum(r[1] for r in rows)
        tl = sum(r[2] for r in rows)
        print(f"\nvs {opp}: overall {tw}W {tl}L winrate={tw/(tw+tl):.3f}")
        for ckpt, w, l, d, r in rows:
            print(f"  {ckpt}: {w}W {l}L winrate={r:.3f}")


if __name__ == "__main__":
    main()
