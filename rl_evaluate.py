#!/usr/bin/env python3
"""
Milestone 6 of the learner_rl RL pipeline: benchmarks a subject package
(default learner_rl) against a fixed held-out opponent set, both orders,
across the 5-map set, and reports only order-independent (trustworthy)
results -- same methodology established earlier this session for the evo
loop, since raw win totals on this engine are contaminated by real
positional bias on several maps (see evo_round.py's MAPS comment).

This is the metric to track across training, not the training loop's own
self-reported policy/value loss -- those describe optimization progress
against whatever opponent that iteration happened to sample, not actual
strength against a fixed yardstick.
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MATCHES_DIR = PROJECT_ROOT / "matches"
PARALLEL_LOGS = MATCHES_DIR / "parallel_logs"

MAPS = ["DefaultMedium", "DefaultSmall", "dirtpassageway", "dirtfulcat", "micromap"]

# Deliberately NOT including any learner_rl_ckptN self-checkpoint: those
# are part of what training itself selects against, not an independent
# yardstick.
HELD_OUT_OPPONENTS = ["econ5", "weighted_micro", "micro_move_imitator", "learner"]

SUMMARY_RE = re.compile(
    r"^(?P<a>\S+) vs (?P<b>\S+) on (?P<map>\S+): "
    r"\S+=(?P<a_wins>\d+), \S+=(?P<b_wins>\d+), "
    r"unclear=(?P<unclear>\d+), timeouts=(?P<timeouts>\d+), total=(?P<total>\d+)$"
)


def run_eval(subject: str, opponents=None, games: int = 1):
    opponents = opponents or HELD_OUT_OPPONENTS

    # winner_by[(map, a, b)] = winning package name, so both orders of the
    # same pairing on the same map can be compared for order-independence.
    winner_by = {}

    for opponent in opponents:
        matchups_file = PROJECT_ROOT / "rl_eval_matchups.txt"
        matchups_file.write_text(f"{subject},{opponent}\n{opponent},{subject}\n")

        for m in MAPS:
            cmd = [
                sys.executable, "parallel_run.py",
                "--matchups", str(matchups_file),
                "--maps", m,
                "--games", str(games),
                "--skip-compile",
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            output = result.stdout + result.stderr
            for line in output.splitlines():
                sm = SUMMARY_RE.match(line.strip())
                if not sm:
                    continue
                a, b = sm.group("a"), sm.group("b")
                a_wins = int(sm.group("a_wins"))
                winner = a if a_wins > 0 else b
                winner_by[(m, a, b)] = winner

    results = {}  # opponent -> {"trustworthy": n, "subject_wins": n, "total_trustworthy": n}
    for opponent in opponents:
        trustworthy = 0
        subject_wins = 0
        for m in MAPS:
            w1 = winner_by.get((m, subject, opponent))
            w2 = winner_by.get((m, opponent, subject))
            if w1 is not None and w2 is not None and w1 == w2:
                trustworthy += 1
                if w1 == subject:
                    subject_wins += 1
        results[opponent] = {"trustworthy": trustworthy, "subject_wins": subject_wins, "maps_checked": len(MAPS)}

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="learner_rl", help="Package to evaluate (default learner_rl)")
    parser.add_argument("--games", type=int, default=1, help="Games per order per map (default 1)")
    args = parser.parse_args()

    results = run_eval(args.subject, games=args.games)

    print(f"Evaluating {args.subject} against the held-out opponent set (order-independent results only):\n")
    total_trustworthy = 0
    total_wins = 0
    for opponent, r in results.items():
        print(f"  vs {opponent}: {r['subject_wins']}/{r['trustworthy']} trustworthy wins "
              f"({r['trustworthy']}/{r['maps_checked']} maps gave a clean signal)")
        total_trustworthy += r["trustworthy"]
        total_wins += r["subject_wins"]

    print(f"\nOverall: {total_wins}/{total_trustworthy} trustworthy wins across all held-out opponents "
          f"({total_trustworthy}/{len(MAPS) * len(results)} maps gave a clean signal)")


if __name__ == "__main__":
    main()
