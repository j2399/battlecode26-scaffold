#!/usr/bin/env python3
"""
Alternative to rl_evaluate.py's win/loss-based evaluation. That approach
gets at most 5 usable data points per opponent (one per map), and often
far fewer after filtering for order-independence (positional bias) --
several evaluations this session came back with only 0-2 trustworthy
readings per opponent, which is too little signal to distinguish real
improvement from noise.

This instead looks at every one of the subject's own robots' whole
per-turn trajectory and scores it on damage done vs. health lost (or
death) -- reusing the exact same "[traj]" logs already collected for RL
training (rl_collect.py's collect_from_log, which already correctly
filters to only the subject's own side -- see the team-mislabeling bug it
was built to fix). "Damage done" is RobotPlayer.java's actionValue, which
folds bites, thrown-rat collisions, and capturing an enemy into one
damage-equivalent number, not just plain bites. A single match easily
contains a dozen+ robot trajectories, so this gives far more independent
samples per match than one win/loss bit, and answers a more direct
question: "how did each rat actually trade," not "who won the race to the
enemy camp" (which is exactly the question contaminated by this engine's
positional bias).

Only works when the subject is a learner_rl-family package (it and every
self-checkpoint print "[traj]" lines); the opponent can be anything, since
only the subject's own trajectory data is needed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, LEARNER, PARALLEL_LOGS, PROJECT_ROOT

MAPS = ["DefaultMedium", "DefaultSmall", "dirtpassageway", "dirtfulcat", "micromap"]
HELD_OUT_OPPONENTS = ["econ5", "weighted_micro", "micro_move_imitator", "learner"]

OWN_HEALTH_INDEX = 18   # state[18] = ownHealth -- see RobotPlayer.java's buildState(); requires STATE_DIM >= 19
DEATH_HEALTH_THRESHOLD = 15.0  # last observed health at/below this -> likely died shortly after its last logged turn


def robot_efficiency(steps):
    """One robot's whole trajectory -> {damage_done, health_lost, died}.
    damage_done sums actionValue across every turn -- a unified damage-
    equivalent figure covering bites, thrown-rat collisions, and
    capturing an enemy (see RobotPlayer.java's autoActions()), not just
    plain bites. A robot's trajectory just stops when it dies (no more
    turns to log), so we can't see the literal fatal blow -- last observed
    health is used as a proxy: at/below DEATH_HEALTH_THRESHOLD counts as
    having died, in which case health_lost is counted as its full starting
    health (it eventually lost everything it had, even if the exact last
    few points of damage weren't captured in a logged turn)."""
    if not steps:
        return None
    first_health = steps[0][1][OWN_HEALTH_INDEX] * 100.0
    last_health = steps[-1][1][OWN_HEALTH_INDEX] * 100.0
    damage_done = sum(s[5] for s in steps)  # s[5] = actionValue (bites+throws+captures, unified)
    died = last_health <= DEATH_HEALTH_THRESHOLD
    health_lost = first_health if died else max(0.0, first_health - last_health)
    return {"damage_done": damage_done, "health_lost": health_lost, "died": died}


def run_combat_eval(subject: str, opponents=None, maps=None, games: int = 1):
    opponents = opponents or HELD_OUT_OPPONENTS
    maps = maps or MAPS

    per_opponent = {}
    for opponent in opponents:
        matchups_file = PROJECT_ROOT / "rl_combat_eval_matchups.txt"
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

        robot_stats = []
        for log in PARALLEL_LOGS.glob("*.log"):
            trajectories, _outcome, _round = collect_from_log(log, subject)
            for traj in trajectories:
                stat = robot_efficiency(traj["steps"])
                if stat:
                    robot_stats.append(stat)

        n = len(robot_stats)
        total_damage = sum(s["damage_done"] for s in robot_stats)
        total_health_lost = sum(s["health_lost"] for s in robot_stats)
        deaths = sum(1 for s in robot_stats if s["died"])
        per_opponent[opponent] = {
            "n_robots": n,
            "mean_damage_done": total_damage / n if n else 0.0,
            "mean_health_lost": total_health_lost / n if n else 0.0,
            "trade_ratio": (total_damage / total_health_lost) if total_health_lost > 0 else (float("inf") if total_damage > 0 else 0.0),
            "death_rate": deaths / n if n else 0.0,
        }
    return per_opponent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="learner_rl", help="learner_rl-family package to evaluate (default learner_rl)")
    parser.add_argument("--games", type=int, default=1, help="Games per order per map (default 1)")
    args = parser.parse_args()

    results = run_combat_eval(args.subject, games=args.games)

    print(f"Per-turn combat efficiency for {args.subject} (damage done vs. health lost, per robot-trajectory):\n")
    total_n, total_damage, total_health_lost, total_deaths = 0, 0.0, 0.0, 0
    for opponent, r in results.items():
        print(f"  vs {opponent}: {r['n_robots']} robot-trajectories -- "
              f"mean damage done={r['mean_damage_done']:.1f}, mean health lost={r['mean_health_lost']:.1f}, "
              f"trade ratio={r['trade_ratio']:.2f}, death rate={100 * r['death_rate']:.0f}%")
        total_n += r["n_robots"]
        total_damage += r["mean_damage_done"] * r["n_robots"]
        total_health_lost += r["mean_health_lost"] * r["n_robots"]
        total_deaths += r["death_rate"] * r["n_robots"]

    if total_n:
        overall_ratio = total_damage / total_health_lost if total_health_lost > 0 else float("inf")
        print(f"\nOverall: {total_n} robot-trajectories, trade ratio={overall_ratio:.2f}, "
              f"death rate={100 * total_deaths / total_n:.0f}%")


if __name__ == "__main__":
    main()
