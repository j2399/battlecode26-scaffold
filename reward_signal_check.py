#!/usr/bin/env python3
"""Checks whether the dense per-turn shaped reward actually correlates
with winning matches, and with a better final cheese/health margin --
computed two ways per match so the newly-added proximity shaping term
can be compared directly against the combat-only reward it's layered on
top of: `combat_reward` (action_value*ACTION_VALUE_SCALE +
hp_delta*HP_DELTA_SCALE, the original formula) and `combined_reward`
(combat_reward + the same potential-difference proximity term
build_batch now adds). Neither includes the terminal win/loss bonus.
Runs learner_rl vs a mix of opponents across TRAIN_MAPS, both orders,
keeping each match's trajectories grouped (unlike run_batch, which
flattens them) so reward can be summed per-match and matched against
that match's own outcome/margin."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_collect import collect_from_log, final_team_stats, team_sides_from_filename, PARALLEL_LOGS, PROJECT_ROOT, LEARNER
from rl_train import (TRAIN_MAPS, ACTION_VALUE_SCALE, HP_DELTA_SCALE, CAPTURE_DAMAGE_EQUIVALENT,
                       GAMMA, PROXIMITY_SHAPING_SCALE, enemy_potential)

OPPONENTS = ["econ5", "weighted_micro", "micro_move_imitator"]
GAMES = 1


def rewards_for_match(log_path):
    """Sums both reward formulas (dense terms only, no terminal bonus)
    across every one of our robots in this one match."""
    trajectories, outcome, round_num = collect_from_log(log_path, LEARNER)
    combat_total = 0.0
    proximity_total = 0.0
    for traj in trajectories:
        steps = traj["steps"]
        for t, (turn, state, action, exploratory, just_captured, action_value, hp_delta, logits) in enumerate(steps):
            if t == 0:
                continue
            if just_captured:
                combat_total -= CAPTURE_DAMAGE_EQUIVALENT * HP_DELTA_SCALE
            combat_total += action_value * ACTION_VALUE_SCALE
            combat_total += hp_delta * HP_DELTA_SCALE
            proximity_total += PROXIMITY_SHAPING_SCALE * (
                GAMMA * enemy_potential(state) - enemy_potential(steps[t - 1][1])
            )
    return combat_total, combat_total + proximity_total, outcome, round_num


def main():
    matchups_file = PROJECT_ROOT / "reward_check_matchups.txt"
    lines = []
    for opp in OPPONENTS:
        lines.append(f"{LEARNER},{opp}\n{opp},{LEARNER}\n")
    matchups_file.write_text("".join(lines))

    PARALLEL_LOGS.mkdir(parents=True, exist_ok=True)
    for f in PARALLEL_LOGS.glob("*.log"):
        f.unlink()

    for m in TRAIN_MAPS:
        cmd = [sys.executable, "parallel_run.py", "--matchups", str(matchups_file),
               "--maps", m, "--games", str(GAMES), "--skip-compile"]
        subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    rows = []  # (combat_reward, combined_reward, win(1/0), margin, opponent, map)
    for log in sorted(PARALLEL_LOGS.glob("*.log")):
        team_a, team_b = team_sides_from_filename(log)
        if LEARNER not in (team_a, team_b):
            continue
        opp = team_b if team_a == LEARNER else team_a
        combat_reward, combined_reward, outcome, round_num = rewards_for_match(log)
        if outcome is None:
            continue
        our_side = "A" if team_a == LEARNER else "B"
        opp_side = "B" if our_side == "A" else "A"
        stats = final_team_stats(log)
        margin = None
        if our_side in stats and opp_side in stats:
            our_cheese, our_health = stats[our_side]
            opp_cheese, opp_health = stats[opp_side]
            margin = (our_cheese * 2 + our_health) - (opp_cheese * 2 + opp_health)
        win = 1 if outcome > 0 else 0
        rows.append((combat_reward, combined_reward, win, margin, opp, log.stem))
        print(f"{log.stem}: combat_reward={combat_reward:.2f} combined_reward={combined_reward:.2f} win={win} margin={margin}", flush=True)

    print(f"\n=== {len(rows)} matches with usable data ===")

    import statistics

    def pearson(xs, ys):
        if len(xs) <= 2:
            return float("nan")
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")

    for label, idx in [("combat_reward (original)", 0), ("combined_reward (+ proximity shaping)", 1)]:
        vals = [r[idx] for r in rows]
        wins_r = [r[idx] for r in rows if r[2] == 1]
        losses_r = [r[idx] for r in rows if r[2] == 0]
        print(f"\n--- {label} ---")
        print(f"mean when WON:  {sum(wins_r)/len(wins_r):.3f}  (n={len(wins_r)})" if wins_r else "no wins")
        print(f"mean when LOST: {sum(losses_r)/len(losses_r):.3f}  (n={len(losses_r)})" if losses_r else "no losses")
        corr_outcome = pearson(vals, [r[2] for r in rows])
        print(f"Pearson correlation(reward, win): {corr_outcome:.3f}")
        margin_rows = [r for r in rows if r[3] is not None]
        if margin_rows:
            corr_margin = pearson([r[idx] for r in margin_rows], [r[3] for r in margin_rows])
            print(f"Pearson correlation(reward, final cheese+health margin): {corr_margin:.3f}  (n={len(margin_rows)})")


if __name__ == "__main__":
    main()
