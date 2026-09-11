#!/usr/bin/env python3
"""
Runs ONE generation of random-mutation tournament selection for
weighted_micro's combat scoring constants -- no AI reasoning involved,
just mutate-and-select, as an alternative to tune_round.py/tune_loop.sh's
AI-driven approach.

Each generation:
  1. Reads the current 2 baseline packages from evo_state.json (both
     default to "weighted_micro" the first time this runs).
  2. Fields 7 competitors: the 2 baselines themselves, unmutated
     (elitism -- see below), plus 5 mutated candidates (3 mutated from
     baseline 1, 2 from baseline 2). Each mutant copies its parent
     package verbatim, with System.out.println calls stripped (this is
     a pure win/loss tournament -- no AI reads these logs, and printing
     has real bytecode cost at this scale), except CombatTileScore.java
     also gets every one of its named `private static final int`
     constants scaled by its own independent random factor in
     [0.8, 1.2] (+-20%), rounded, floored at 1.
  3. Compiles once, then round-robins all 7 competitors against each
     other -- every ordered pair (so both orders of every matchup),
     across a fixed 5-map set (evileye/thunderdome/knifefight excluded
     -- see MAPS below), --games games per pairing.
  4. Ranks all 7 by total wins across the whole tournament. The top 2
     become next generation's baselines, persisted back to
     evo_state.json. Because the unmutated baselines are in the field
     too, a generation where every mutation is worse simply re-selects
     the same baselines -- next round's pair can only ever be at least
     as good as this round's, never regress from a bad mutation batch.

Prints a JSON summary to stdout on success for evo_loop.sh to report.
Exits nonzero (before touching evo_state.json) if compilation fails, so
a broken mutation never becomes a baseline.

All candidate packages are kept on disk (never deleted) as a full
history of the run, same policy as tune_round.py's weighted_micro_v*
snapshots.
"""

import argparse
import glob
import itertools
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
MATCHES_DIR = PROJECT_ROOT / "matches"
PARALLEL_LOGS = MATCHES_DIR / "parallel_logs"
EVO_DIR = MATCHES_DIR / "evo_rounds"
STATE_FILE = PROJECT_ROOT / "evo_state.json"

LIVE_BOT = "weighted_micro"
CONST_RE = re.compile(r"(private static final int (\w+) = )(-?\d+)(;)")

MAPS = [
    # evileye, thunderdome, and knifefight are excluded: in the most
    # thorough test run (a 6-candidate round-robin shootout), literally
    # every pairing on those 3 maps flipped winner when team order
    # flipped -- 0/15 order-independent results each, pure positional
    # bias with no bot-quality signal at all. The maps kept here at
    # least sometimes showed a trustworthy (order-independent) result.
    "DefaultMedium", "DefaultSmall",
    "dirtpassageway", "dirtfulcat", "micromap",
]

MUTATE_LOW, MUTATE_HIGH = 0.8, 1.2

PRINTLN_RE = re.compile(r"System\.out\.println\([\s\S]*?\);\s*\n?")

SUMMARY_RE = re.compile(
    r"^(?P<a>\S+) vs (?P<b>\S+) on (?P<map>\S+): "
    r"\S+=(?P<a_wins>\d+), \S+=(?P<b_wins>\d+), "
    r"unclear=(?P<unclear>\d+), timeouts=(?P<timeouts>\d+), total=(?P<total>\d+)$"
)


def compile_project():
    print("[evo_round] compiling...", flush=True)
    result = subprocess.run(["./gradlew", "compileJava", "--console=plain"], cwd=PROJECT_ROOT)
    return result.returncode == 0


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"generation": 0, "baselines": [LIVE_BOT, LIVE_BOT]}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def mutate_combat_tile_score(text: str, rng: random.Random):
    """Every named constant gets its own independent +-20% roll -- no
    subset sampling, the mutation is applied uniformly across all of
    them."""
    all_matches = list(CONST_RE.finditer(text))
    if not all_matches:
        return text, []

    changes = []
    pieces = []
    last_end = 0
    for m in all_matches:
        prefix, name, value, suffix = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        factor = rng.uniform(MUTATE_LOW, MUTATE_HIGH)
        new_value = max(1, round(value * factor))
        pieces.append(text[last_end:m.start()])
        pieces.append(f"{prefix}{new_value}{suffix}")
        last_end = m.end()
        pct = round((factor - 1) * 100)
        changes.append(f"{name}: {value} -> {new_value} ({pct:+d}%)")
    pieces.append(text[last_end:])
    return "".join(pieces), changes


def strip_prints(text: str) -> str:
    return PRINTLN_RE.sub("", text)


def create_candidate(parent: str, candidate: str, rng: random.Random):
    parent_dir = SRC / parent
    dest_dir = SRC / candidate
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    changes = []
    for java_file in parent_dir.glob("*.java"):
        text = java_file.read_text()
        text = re.sub(
            rf"^package {re.escape(parent)};",
            f"package {candidate};",
            text, count=1, flags=re.MULTILINE,
        )
        text = strip_prints(text)
        if java_file.name == "CombatTileScore.java":
            text, changes = mutate_combat_tile_score(text, rng)
        (dest_dir / java_file.name).write_text(text)
    return changes


def run_tournament(candidates: list, games: int, round_dir: Path) -> dict:
    matchups_dir = round_dir / "matchups"
    matchups_dir.mkdir()

    wins = {c: 0 for c in candidates}
    pairs = list(itertools.permutations(candidates, 2))  # every ordered pair -> both orders of every matchup

    for map_name in MAPS:
        matchups_file = matchups_dir / f"matchups_{map_name}.txt"
        matchups_file.write_text("\n".join(f"{a},{b}" for a, b in pairs) + "\n")

        cmd = [
            sys.executable, "parallel_run.py",
            "--matchups", str(matchups_file),
            "--maps", map_name,
            "--games", str(games),
            "--skip-compile",
        ]
        print(f"[evo_round] tournament on {map_name}: {len(pairs)} pairing(s) x {games} game(s)...", flush=True)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        output = result.stdout + result.stderr
        print(output, flush=True)
        (round_dir / f"summary_{map_name}.txt").write_text(output)

        for line in output.splitlines():
            m = SUMMARY_RE.match(line.strip())
            if not m:
                continue
            a, b = m.group("a"), m.group("b")
            wins[a] = wins.get(a, 0) + int(m.group("a_wins"))
            wins[b] = wins.get(b, 0) + int(m.group("b_wins"))

    return wins


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=1, help="Games per pairing per map (default 1; matches are deterministic given fixed code/map, so repeats of the same exact pairing only help if a bot uses its own randomness)")
    args = parser.parse_args()

    state = load_state()
    generation = state["generation"] + 1
    b1, b2 = state["baselines"][0], state["baselines"][1]

    rng = random.Random()

    candidates = []
    candidate_info = {}

    # Elitism: the current baselines compete unmutated alongside the
    # mutants, so a bad generation of mutations can never lose ground --
    # next round's baselines can only ever be at least as good as this
    # round's best. (b1 == b2 only happens on a freshly reset state.)
    elites = [b1] if b1 == b2 else [b1, b2]
    for elite in elites:
        candidates.append(elite)
        candidate_info[elite] = {"parent": None, "changes": ["(unchanged, carried over as an elite baseline)"]}

    mutant_count = 7 - len(elites)
    from_b1 = (mutant_count + 1) // 2
    for i in range(1, from_b1 + 1):
        name = f"weighted_micro_evo{generation}_c{i}"
        changes = create_candidate(b1, name, rng)
        candidates.append(name)
        candidate_info[name] = {"parent": b1, "changes": changes}
    for i in range(from_b1 + 1, mutant_count + 1):
        name = f"weighted_micro_evo{generation}_c{i}"
        changes = create_candidate(b2, name, rng)
        candidates.append(name)
        candidate_info[name] = {"parent": b2, "changes": changes}

    if not compile_project():
        print(json.dumps({"ok": False, "reason": "compile_failed", "generation": generation}))
        sys.exit(1)

    EVO_DIR.mkdir(parents=True, exist_ok=True)
    round_dir = EVO_DIR / f"gen_{generation}"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "candidates.json").write_text(json.dumps(candidate_info, indent=2))

    PARALLEL_LOGS.mkdir(parents=True, exist_ok=True)
    for f in PARALLEL_LOGS.glob("*.log"):
        f.unlink()

    wins = run_tournament(candidates, args.games, round_dir)

    archived_logs = round_dir / "worker_logs"
    if PARALLEL_LOGS.exists():
        shutil.move(str(PARALLEL_LOGS), str(archived_logs))

    ranked = sorted(candidates, key=lambda c: wins.get(c, 0), reverse=True)
    top2 = ranked[:2]

    (round_dir / "results.json").write_text(json.dumps(
        {"wins": wins, "ranked": ranked, "top2": top2}, indent=2))

    save_state({"generation": generation, "baselines": top2})

    print(json.dumps({
        "ok": True,
        "generation": generation,
        "candidates": candidates,
        "candidate_info": candidate_info,
        "wins": wins,
        "ranked": ranked,
        "new_baselines": top2,
        "round_dir": str(round_dir.relative_to(PROJECT_ROOT)),
    }))


if __name__ == "__main__":
    main()
