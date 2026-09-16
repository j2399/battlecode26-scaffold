#!/usr/bin/env python3
"""
Milestone 1 of the learner_rl RL pipeline: runs a batch of learner_rl vs
opponent matches, parses the "[traj]" debug lines learner_rl's robots print
each turn into clean per-robot trajectories, and ties each trajectory to
that match's actual win/loss outcome via the "[server] ... wins" line.

This is a pure plumbing/correctness check -- no training happens here.
See rl_train.py (milestone 2+) for the actual policy-gradient loop, which
imports run_batch()/collect_from_log() from this file rather than
reimplementing collection.
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MATCHES_DIR = PROJECT_ROOT / "matches"
PARALLEL_LOGS = MATCHES_DIR / "parallel_logs"

LEARNER = "learner_rl"

MAPS = [
    "DefaultMedium", "DefaultSmall",
    "dirtpassageway", "dirtfulcat", "micromap",
]

# Group 4 (the action segment) is captured as a whole comma-list and split
# in Python rather than one regex group per field -- robust to adding more
# shaped-reward fields later (e.g. didAttack, hpDelta) without touching
# the regex, and hpDelta is a signed float so a digit-only capture group
# wouldn't have matched it anyway.
TRAJ_RE = re.compile(r"\[traj\] (\d+),(\d+),([^|]+)\|([^|]+)\|(.+)")
WIN_RE = re.compile(r"\[server\]\s+(\S+) \([AB]\) wins \(round (\d+)\)")

# Every rat (baby and king) prints "[stats] id,round,cheese,health" every
# turn -- see RobotPlayer.java/RatKing.java. Deliberately its own format,
# not folded into "[traj]": [traj] is only ever collected for one side at
# a time (see collect_from_log's team-mislabeling-bug fix below), while
# "[stats]" exists specifically so final_team_stats() can read BOTH sides.
STATS_RE = re.compile(r"\[stats\] (\d+),(\d+),(\d+),(\d+)")

# RatKing's own "[stats]" line carries a trailing ",K" (see RatKing.java)
# so king_and_cheese_by_round can tell it apart from a baby rat's
# otherwise-identical line. STATS_RE.search still matches either (it
# doesn't anchor past the 4th number), so this only needs its own check.
KING_MARKER = ",K"


def parse_traj_line(line: str):
    """Returns (robot_id, turn, state, action, exploratory, just_captured,
    action_value, hp_delta, logits). action_value folds bites, thrown-rat
    collisions, and capturing an enemy into one damage-equivalent float
    (see learner_rl/RobotPlayer.java's autoActions()) -- replaces the
    older separate did_attack(bool)/just_captured_enemy(bool) fields."""
    m = TRAJ_RE.search(line)
    if not m:
        return None
    robot_id = int(m.group(1))
    turn = int(m.group(2))
    state = [float(x) for x in m.group(3).split(",")]
    action_fields = m.group(4).split(",")
    action = int(action_fields[0])
    exploratory = bool(int(action_fields[1]))
    just_captured = bool(int(action_fields[2]))
    action_value = float(action_fields[3])
    hp_delta = float(action_fields[4])
    logits = [float(x) for x in m.group(5).split(",")]
    return robot_id, turn, state, action, exploratory, just_captured, action_value, hp_delta, logits


def team_sides_from_filename(log_path: Path):
    """Parallel_run.py names worker logs '<team_a>-vs-<team_b>-on-<map>-workerN.log'.
    Returns (team_a, team_b). Splitting on literal '-vs-'/'-on-' is safe
    here since none of our package names contain hyphens."""
    stem = log_path.stem
    team_a, rest = stem.split("-vs-", 1)
    team_b, _rest = rest.split("-on-", 1)
    return team_a, team_b


def opponent_hp_delta_by_round(log_path: Path, opponent_side: str):
    """Returns {round: total_hp_delta} for `opponent_side` ('A' or 'B'),
    computed from PER-ROBOT health deltas between consecutive rounds --
    deliberately not a raw total-health difference. Checked directly
    against a real log: a team's total health climbs steadily round over
    round almost entirely from reinforcements spawning (each a fresh
    ~100hp baby rat with a brand new robot id -- e.g. ids 12708, 12349,
    11984, 12766 each first appearing at consecutive rounds in one real
    match), which would get counted as "the opponent doing great" even
    in rounds where every one of their EXISTING units is actively losing
    a fight. Restricting to the same robot id seen in back-to-back
    rounds isolates actual combat damage from roster-size changes: a
    robot's first appearance contributes nothing (no prior round to
    diff against), and a gap (robot missing a round, e.g. from dying)
    isn't bridged into a delta either.

    See final_team_stats for the same per-side "[stats]" reading
    pattern (and its caveat: a package that doesn't print "[stats]" at
    all, e.g. econ5_no_cheese or weighted_micro, yields an empty dict
    here). Used for the zero-sum reward term in rl_train.py's
    build_batch (OPPONENT_ZERO_SUM_SCALE) -- catches outcomes our own
    hp_delta/action_value never could, since those only ever measure
    events on OUR side (e.g. their king starving from a cheese shortage
    our combat pressure caused shows up nowhere in what we logged, only
    in their own health dropping)."""
    text = log_path.read_text(errors="replace")
    prefix = f"[{opponent_side}:"
    health_by_robot_round = defaultdict(dict)
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        m = STATS_RE.search(line)
        if not m:
            continue
        robot_id, round_num_s, _cheese, health = (int(x) for x in m.groups())
        health_by_robot_round[robot_id][round_num_s] = health

    delta_by_round = defaultdict(float)
    for robot_id, by_round in health_by_robot_round.items():
        rounds = sorted(by_round.keys())
        for prev_r, cur_r in zip(rounds, rounds[1:]):
            if cur_r == prev_r + 1:
                delta_by_round[cur_r] += by_round[cur_r] - by_round[prev_r]
    return dict(delta_by_round)


def king_and_cheese_by_round(log_path: Path, side: str):
    """Returns {round: (king_health, cheese)} for `side` ('A' or 'B'),
    read from that side's RatKing "[stats]" line (identified by the
    trailing KING_MARKER -- see RatKing.java). Cheese is team-wide
    (rc.getGlobalCheese()) so any one robot's reading of it is already
    the team's whole bank; king health specifically needs the marker
    since a plain "[stats]" line can't otherwise be told apart from a
    baby rat's. Empty dict if this side's package doesn't print "[stats]"
    at all (same caveat as opponent_hp_delta_by_round/final_team_stats),
    or if it prints "[stats]" but not the king-marked variant (e.g. an
    older, un-rebuilt package snapshot from before this marker existed --
    the marker was added mid-session, so any self-checkpoint package
    frozen before that point won't have it when playing as either side).

    Used to give rl_train.py's (training-time-only) ValueNet direct
    visibility into both kings' health and both sides' cheese -- signals
    close to the actual win condition itself, which no single robot's own
    buildState() carries (king health is deliberately excluded from the
    policy's own combat-balance feature, see buildState()'s comment) and
    which the existing team_avg/opponent_hp_delta value-net features only
    reflect indirectly, through a noisy combat-reward proxy."""
    text = log_path.read_text(errors="replace")
    prefix = f"[{side}:"
    result = {}
    for line in text.splitlines():
        if not line.startswith(prefix) or KING_MARKER not in line:
            continue
        m = STATS_RE.search(line)
        if not m:
            continue
        _robot_id, round_num_s, cheese, health = (int(x) for x in m.groups())
        result[round_num_s] = (health, cheese)
    return result


def collect_from_log(log_path: Path, our_team: str = LEARNER):
    """Returns (list of per-robot trajectory dicts, outcome for our_team or
    None, the match's round count or None). Any package sharing
    learner_rl's RobotPlayer.java (every self-checkpoint, and learner_rl
    itself) prints identical "[traj]" lines tagged with the engine's own
    "[A: ...]"/"[B: ...]" prefix -- so when the opponent is a self-
    checkpoint, its lines are ALSO present in this same log and must be
    excluded, or its actions get collected and mislabeled with our_team's
    outcome instead of its own. Filters strictly to whichever side (A or
    B) our_team actually played as this match, determined from the log
    filename, not just from grepping every "[traj]" line in the file.

    Each trajectory dict also carries match_id (the log file's stem,
    unique per match -- lets build_batch regroup this otherwise-flat
    list back into "which robots fought in the same match together",
    needed for team-spirit reward blending) and
    opponent_hp_delta_by_round (see opponent_hp_delta_by_round() above,
    identical for every one of our robots in the same match, used for
    the zero-sum reward term)."""
    team_a, team_b = team_sides_from_filename(log_path)
    if team_a == our_team:
        our_side = "A"
    elif team_b == our_team:
        our_side = "B"
    else:
        return [], None, None  # our_team wasn't actually in this match
    opp_side = "B" if our_side == "A" else "A"

    text = log_path.read_text(errors="replace")

    win_match = WIN_RE.search(text)
    outcome = None
    round_num = None
    if win_match:
        winner = win_match.group(1).strip()
        round_num = int(win_match.group(2))
        outcome = 1.0 if winner == our_team else -1.0

    by_robot = defaultdict(list)
    prefix = f"[{our_side}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        parsed = parse_traj_line(line)
        if parsed:
            robot_id, turn, state, action, exploratory, just_captured, action_value, hp_delta, logits = parsed
            by_robot[robot_id].append((turn, state, action, exploratory, just_captured, action_value, hp_delta, logits))

    match_id = log_path.stem
    opp_hp_delta = opponent_hp_delta_by_round(log_path, opp_side)
    our_king_cheese = king_and_cheese_by_round(log_path, our_side)
    opp_king_cheese = king_and_cheese_by_round(log_path, opp_side)

    trajectories = []
    for robot_id, steps in by_robot.items():
        steps.sort(key=lambda s: s[0])
        trajectories.append({
            "robot_id": robot_id, "steps": steps, "outcome": outcome,
            "match_id": match_id, "opponent_hp_delta_by_round": opp_hp_delta,
            "our_king_cheese_by_round": our_king_cheese, "opp_king_cheese_by_round": opp_king_cheese,
        })
    return trajectories, outcome, round_num


def final_team_stats(log_path: Path):
    """Returns {'A': (cheese, total_health), 'B': (cheese, total_health)}
    for this match, computed from the "[stats]" line every rat prints
    every turn. Unlike collect_from_log (which intentionally discards the
    non-subject side's "[traj]" lines -- see its team-mislabeling-bug fix
    above), this reads BOTH sides on purpose: it's only ever used for a
    cross-side comparison (the cheese+health tiebreak in rl_train.py's
    iteration_result), never fed into RL training data, so there's no
    mislabeling risk to guard against here.

    A robot's last-reported round decides whether it counts as alive at
    match end (its last known health) or dead (contributes 0): a robot
    that stops appearing before the match's actual final round is presumed
    to have died in between (its last logged health was from while it was
    still alive, not 0). A side whose package doesn't print "[stats]" at
    all (e.g. econ5_no_cheese, weighted_micro -- as opposed to learner_rl
    and every self-checkpoint, learner, and micro_move_imitator, which
    do) is simply absent from the returned dict."""
    text = log_path.read_text(errors="replace")
    win_match = WIN_RE.search(text)
    final_round = int(win_match.group(2)) if win_match else None

    stats = {}
    for side in ("A", "B"):
        prefix = f"[{side}:"
        cheese = None
        last_round_by_robot = {}
        last_health_by_robot = {}
        for line in text.splitlines():
            if not line.startswith(prefix):
                continue
            m = STATS_RE.search(line)
            if not m:
                continue
            robot_id, round_num_s, c, h = (int(x) for x in m.groups())
            cheese = c
            last_round_by_robot[robot_id] = round_num_s
            last_health_by_robot[robot_id] = h
        if cheese is None:
            continue  # this side's package doesn't print "[stats]"
        alive_cutoff = final_round if final_round is not None else max(last_round_by_robot.values(), default=0)
        total_health = sum(
            h for rid, h in last_health_by_robot.items()
            if last_round_by_robot[rid] >= alive_cutoff
        )
        stats[side] = (cheese, total_health)
    return stats


def run_batch(opponent: str, maps=None, games: int = 1, subject: str = LEARNER):
    """Runs `subject` (learner_rl by default -- see train_exploiter in
    rl_train.py for why this needs to be overridable: it trains a
    separate in-memory model that has to be exported to its own package
    and played as-is, not as learner_rl) vs opponent (both orders) across
    maps, returns (all_trajectories, list_of_outcomes_for_subject,
    list_of_round_counts, list_of_team_scores) -- the latter two parallel
    to outcomes, one per decided match. Each team_scores entry is
    (our_score, opponent_score) from final_team_stats (cheese*2 + total_
    health), or None if that match's opponent package doesn't print
    "[stats]" and so no score could be computed for it."""
    maps = maps or MAPS

    matchups_file = PROJECT_ROOT / "rl_collect_matchups.txt"
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

    all_trajectories = []
    outcomes = []
    round_nums = []
    team_scores = []
    for log in PARALLEL_LOGS.glob("*.log"):
        trajs, outcome, round_num = collect_from_log(log, subject)
        all_trajectories.extend(trajs)
        if outcome is not None:
            outcomes.append(outcome)
            round_nums.append(round_num)

            team_a, team_b = team_sides_from_filename(log)
            our_side = "A" if team_a == subject else "B"
            opp_side = "B" if our_side == "A" else "A"
            stats = final_team_stats(log)
            if our_side in stats and opp_side in stats:
                team_scores.append((stats[our_side], stats[opp_side]))  # each is (cheese, total_health)
            else:
                team_scores.append(None)

    return all_trajectories, outcomes, round_nums, team_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default="weighted_micro", help="Opponent package to play against (default weighted_micro)")
    parser.add_argument("--games", type=int, default=1, help="Games per pairing per map (default 1)")
    args = parser.parse_args()

    trajectories, outcomes, round_nums, team_scores = run_batch(args.opponent, games=args.games)

    n_steps = sum(len(t["steps"]) for t in trajectories)
    n_exploratory = sum(1 for t in trajectories for s in t["steps"] if s[3])
    n_captured = sum(1 for t in trajectories for s in t["steps"] if s[4])
    n_action_events = sum(1 for t in trajectories for s in t["steps"] if s[5] > 0)
    total_action_value = sum(s[5] for t in trajectories for s in t["steps"])
    action_counts = defaultdict(int)
    for t in trajectories:
        for s in t["steps"]:
            action_counts[s[2]] += 1
    wins = sum(1 for o in outcomes if o > 0)

    print(f"Opponent: {args.opponent}")
    print(f"Matches with a decided outcome: {len(outcomes)} ({wins} wins for {LEARNER}, {len(outcomes) - wins} losses)")
    print(f"Trajectories (robot-match pairs): {len(trajectories)}")
    print(f"Total (state, action) steps: {n_steps}")
    print(f"Exploratory steps: {n_exploratory} ({100 * n_exploratory / max(n_steps, 1):.1f}%)")
    print(f"Got captured: {n_captured} times. Damage-equivalent actions (bites/throws/captures): "
          f"{n_action_events} events, {total_action_value:.0f} total damage-equivalent value")
    print(f"Action distribution: {dict(sorted(action_counts.items()))}")
    if trajectories:
        lengths = [len(t["steps"]) for t in trajectories]
        print(f"Trajectory length: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
