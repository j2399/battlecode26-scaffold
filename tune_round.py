#!/usr/bin/env python3
"""
Runs ONE round of the weighted_micro tuning benchmark:

  1. Compile once.
  2. Find the last (up to) 5 versioned history opponents
     (src/weighted_micro_v{N}, highest N first).
  3. For each of the 8 fixed maps, play weighted_micro vs each history
     opponent in both orders (weighted_micro as team A, then as team B),
     --games games per pairing, via parallel_run.py.
  4. Archive that round's worker logs (which contain the [combat] debug
     prints from CombatState.java) and a win/loss summary under
     matches/tune_rounds/round_{N}/.
  5. Snapshot the *current* src/weighted_micro/ into a new
     src/weighted_micro_v{N+1}/ history entry (mechanical copy + package
     rename only -- no judgment calls, so it's done here rather than by
     the AI tuning step).

Prints a small JSON blob to stdout on success (round dir, snapshot
version, opponents faced, log dir) for tune_loop.sh to build the next
prompt from. Exits nonzero (before any snapshot) if compilation fails,
so a broken build never gets benchmarked or frozen into history.
"""

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
MATCHES_DIR = PROJECT_ROOT / "matches"
PARALLEL_LOGS = MATCHES_DIR / "parallel_logs"
ROUNDS_DIR = MATCHES_DIR / "tune_rounds"

LIVE_BOT = "weighted_micro"
HISTORY_PREFIX = "weighted_micro_v"
HISTORY_RE = re.compile(rf"^{HISTORY_PREFIX}(\d+)$")

MAPS = [
    "evileye", "DefaultMedium", "DefaultSmall", "thunderdome",
    "dirtpassageway", "dirtfulcat", "micromap", "knifefight",
]


def history_versions():
    versions = []
    for p in SRC.glob(f"{HISTORY_PREFIX}*"):
        if not p.is_dir():
            continue
        m = HISTORY_RE.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions)


def compile_project():
    print("[tune_round] compiling...", flush=True)
    result = subprocess.run(
        ["./gradlew", "compileJava", "--console=plain"],
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0


def run_map_benchmark(map_name: str, opponents: list, games: int, matchups_dir: Path) -> str:
    """Runs weighted_micro vs each opponent, both orders, on one map.
    Returns the captured stdout (includes parallel_run.py's summary)."""
    pairs = []
    for v in opponents:
        opp = f"{HISTORY_PREFIX}{v}"
        pairs.append((LIVE_BOT, opp))
        pairs.append((opp, LIVE_BOT))

    matchups_file = matchups_dir / f"matchups_{map_name}.txt"
    matchups_file.write_text("\n".join(f"{a},{b}" for a, b in pairs) + "\n")

    cmd = [
        sys.executable, "parallel_run.py",
        "--matchups", str(matchups_file),
        "--maps", map_name,
        "--games", str(games),
        "--skip-compile",
    ]
    print(f"[tune_round] running {len(pairs)} pairing(s) x {games} game(s) on {map_name}...", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, flush=True)
    return output


def snapshot_live_bot_as_next_version(next_version: int):
    dest = SRC / f"{HISTORY_PREFIX}{next_version}"
    src = SRC / LIVE_BOT
    if dest.exists():
        sys.exit(f"[tune_round] refusing to overwrite existing history dir {dest}")
    dest.mkdir()
    new_pkg = f"{HISTORY_PREFIX}{next_version}"
    for java_file in src.glob("*.java"):
        text = java_file.read_text()
        text = re.sub(
            rf"^package {re.escape(LIVE_BOT)};",
            f"package {new_pkg};",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        (dest / java_file.name).write_text(text)
    print(f"[tune_round] snapshotted current {LIVE_BOT} -> {dest.relative_to(PROJECT_ROOT)}", flush=True)


def next_round_number() -> int:
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[1]) for p in ROUNDS_DIR.glob("round_*") if p.name.split("_")[1].isdigit()]
    return (max(existing) + 1) if existing else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2, help="Games per pairing per map (default 2)")
    parser.add_argument("--history-count", type=int, default=5, help="How many recent history versions to benchmark against (default 5)")
    args = parser.parse_args()

    if not compile_project():
        print(json.dumps({"ok": False, "reason": "compile_failed"}))
        sys.exit(1)

    versions = history_versions()
    if not versions:
        sys.exit("[tune_round] no history versions found under src/weighted_micro_v* -- nothing to benchmark against")
    opponents = versions[-args.history_count:]

    round_num = next_round_number()
    round_dir = ROUNDS_DIR / f"round_{round_num}"
    round_dir.mkdir(parents=True)
    matchups_dir = round_dir / "matchups"
    matchups_dir.mkdir()

    PARALLEL_LOGS.mkdir(parents=True, exist_ok=True)
    # Clear stale logs from a prior round so this round's archive step only
    # picks up what this round actually produced.
    for f in PARALLEL_LOGS.glob("*.log"):
        f.unlink()

    summary_lines = []
    for map_name in MAPS:
        output = run_map_benchmark(map_name, opponents, args.games, matchups_dir)
        summary_lines.append(f"--- {map_name} ---\n{output}")

    (round_dir / "summary.txt").write_text("\n".join(summary_lines))

    # Archive this round's per-match debug-print logs before the next round
    # overwrites matches/parallel_logs.
    archived_logs = round_dir / "worker_logs"
    if PARALLEL_LOGS.exists():
        shutil.move(str(PARALLEL_LOGS), str(archived_logs))

    next_version = opponents[-1] + 1 if opponents[-1] == max(versions) else max(versions) + 1
    snapshot_live_bot_as_next_version(next_version)

    result = {
        "ok": True,
        "round": round_num,
        "round_dir": str(round_dir.relative_to(PROJECT_ROOT)),
        "summary_file": str((round_dir / "summary.txt").relative_to(PROJECT_ROOT)),
        "worker_logs_dir": str(archived_logs.relative_to(PROJECT_ROOT)) if archived_logs.exists() else None,
        "opponents_faced": [f"{HISTORY_PREFIX}{v}" for v in opponents],
        "maps": MAPS,
        "games_per_pairing": args.games,
        "snapshotted_as": f"{HISTORY_PREFIX}{next_version}",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
