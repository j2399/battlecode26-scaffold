#!/usr/bin/env python3
"""
Runs many Battlecode matches in parallel as efficiently as possible.

Two overhead sources this avoids, layered on top of each other:

  1. Gradle overhead: `./gradlew run` pays Gradle's daemon/task-graph cost
     on top of the actual match, and N concurrent `./gradlew` invocations
     tend to serialize on the daemon instead of truly parallelizing. Fixed
     by compiling once, then driving the engine directly.

  2. JVM-per-match overhead: even a direct `java ... battlecode.server.Main`
     process pays JVM bootstrap + cold-JIT cost on *every single match*.
     The engine's own Server class holds a BlockingQueue<GameInfo> and its
     run() loop already supports taking many games off that queue in one
     process (this is clearly how the real tournament server avoids
     restarting between matches) -- battlecode.server.Main just never
     queues more than one before exiting. bc-overrides-src/BatchServer.java
     is a small driver that queues a whole *slice* of matches into one
     Server and calls run() once, so JVM startup is paid once per worker
     *process*, not once per match, while still running `--workers`
     processes in true OS-level parallel.

  Also removes the per-turn bytecode limit (see bc-overrides-src/battlecode/
  common/UnitType.java) by shadowing battlecode.common.UnitType earlier on
  the classpath than the official jar -- there's no config flag for this,
  it's hardcoded per-unit-type in the engine.

Usage examples:
  # Run weighted vs weighted_micro on micromap, 20 times, using all cores
  python3 parallel_run.py --team-a weighted --team-b weighted_micro --maps micromap --games 20

  # Sweep several matchups across several maps, 8 workers
  python3 parallel_run.py --matchups matchups.txt --maps micromap,DefaultSmall --workers 8

  # matchups.txt format: one "teamA,teamB" pair per line
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_CLASSES = PROJECT_ROOT / "build" / "classes"
GRADLE_CACHE = Path.home() / ".gradle" / "caches" / "modules-2" / "files-2.1"
MATCHES_DIR = PROJECT_ROOT / "matches"
LOGS_DIR = MATCHES_DIR / "parallel_logs"
# A recompiled battlecode.common.UnitType with each unit's bytecodeLimit
# raised to 100,000,000 (effectively unlimited), shadowing the official
# class -- see bc-overrides-src/battlecode/common/UnitType.java. Must be
# placed *before* the real jar on the classpath since the JVM resolves a
# class to whichever definition it finds first.
BC_OVERRIDES = PROJECT_ROOT / "bc-overrides"

WIN_LINE_RE = re.compile(r"\[server\]\s+(.+?)\s+\([AB]\)\s+wins \(round (\d+)\)")
REASON_LINE_RE = re.compile(r"\[server\] Reason: (.+)")


def find_jar(group_dir_name: str, jar_name_prefix: str) -> Path:
    """Find a jar under the Gradle module cache by artifact name prefix,
    skipping any '-sources'/'-javadoc' variants."""
    base = GRADLE_CACHE / group_dir_name
    if not base.exists():
        sys.exit(f"Could not find Gradle cache dir for {group_dir_name} at {base}")
    candidates = [
        p for p in base.rglob(f"{jar_name_prefix}*.jar")
        if "-sources" not in p.name and "-javadoc" not in p.name
    ]
    if not candidates:
        sys.exit(f"Could not find a jar matching {jar_name_prefix}*.jar under {base}")
    # Prefer the newest by mtime if there are multiple versions cached.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def default_worker_count() -> int:
    """
    Picks a default worker count. Plain os.cpu_count() overcounts on Apple
    Silicon: those chips mix full-speed performance cores with much slower
    efficiency cores, and treating all of them as equally fast leads to
    oversubscribing the P-cores. Measured on an M2 (4P+4E) running this
    exact workload: 4 workers (P-cores only) and 8 workers (all logical
    cores) both landed around 46s wall-clock for 50 matches, while 6
    workers (P-cores + half the E-cores) finished in 38s -- noticeably
    faster than either extreme. So: default to P-cores + half the E-cores,
    rounded, when that info is available; otherwise fall back to
    os.cpu_count() as before.
    """
    try:
        p_cores = int(subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        e_cores = int(subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return p_cores + round(e_cores / 2)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return os.cpu_count() or 4


def build_classpath() -> str:
    battlecode_jar = find_jar("org.battlecode/battlecode26-java", "battlecode26-java")
    scala_jar = find_jar("org.scala-lang/scala-library", "scala-library")
    entries = [str(BUILD_CLASSES)]
    if BC_OVERRIDES.exists():
        entries.append(str(BC_OVERRIDES))  # must precede the jar below
    entries += [str(battlecode_jar), str(scala_jar)]
    return os.pathsep.join(entries)


def compile_once():
    print("Compiling (./gradlew compileJava)...", flush=True)
    result = subprocess.run(
        ["./gradlew", "compileJava", "--console=plain"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        sys.exit("Compile failed -- fix compile errors before running matches.")


def truncate(name: str, limit: int = 80) -> str:
    return name if len(name) <= limit else name[:limit]


class WorkerBatch:
    """A slice of `num_games` identical matchups assigned to one worker
    process, which runs them all inside a single JVM via BatchServer."""
    __slots__ = ("team_a", "team_b", "maps", "start_index", "num_games", "worker_id")

    def __init__(self, team_a, team_b, maps, start_index, num_games, worker_id):
        self.team_a = team_a
        self.team_b = team_b
        self.maps = maps
        self.start_index = start_index
        self.num_games = num_games
        self.worker_id = worker_id

    @property
    def label(self) -> str:
        base = truncate(f"{self.team_a}-vs-{self.team_b}-on-{self.maps}", 60)
        return f"{base}-worker{self.worker_id}"


def run_worker_batch(batch: WorkerBatch, classpath: str, heap_mb: int, timeout_s: int, validate_maps: bool) -> list:
    log_file = LOGS_DIR / f"{batch.label}.log"

    jvm_args = [
        "java",
        f"-Xmx{heap_mb}m",
        "-XX:+UseSerialGC",  # one lightweight GC thread per worker, not several
        # C2 (the JVM's aggressive JIT tier) has real compile-time cost.
        # With many matches now sharing one process the JIT has more chance
        # to pay that cost back, but tier 1 (C1) still starts useful sooner
        # for a process that's done in single-digit seconds either way.
        "-XX:TieredStopAtLevel=1",
        "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.math=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.util=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.access=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        # No client will ever connect to a headless batch run -- skip
        # starting the websocket NetServer thread entirely.
        "-Dbc.server.websocket=false",
        "-Dbc.server.wait-for-client=false",
        "-Dbc.server.mode=headless",
        "-Dbc.server.map-path=maps",
        "-Dbc.server.robot-player-to-system-out=true",
        "-Dbc.server.debug=false",
        "-Dbc.engine.debug-methods=false",
        "-Dbc.engine.enable-profiler=false",
        "-Dbc.engine.show-indicators=false",
        "-Dbc.server.alternate-order=false",
        # Re-validating the same already-valid map file on every one of N
        # repeated runs is pure repeated work -- skip it by default here.
        f"-Dbc.server.validate-maps={'true' if validate_maps else 'false'}",
        "-cp", classpath,
        "BatchServer",
        batch.team_a, batch.team_b, batch.maps,
        str(batch.num_games), str(batch.start_index),
        str(MATCHES_DIR), str(BUILD_CLASSES),
    ]

    start = time.monotonic()
    try:
        with open(log_file, "w") as log:
            proc = subprocess.run(
                jvm_args, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout_s * batch.num_games,
            )
        timed_out = False
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
    elapsed = time.monotonic() - start
    per_match_elapsed = round(elapsed / batch.num_games, 2)

    text = log_file.read_text(errors="replace") if log_file.exists() else ""
    wins = WIN_LINE_RE.findall(text)  # one match per queued match, in order

    results = []
    for i in range(batch.num_games):
        run_index = batch.start_index + i
        if i < len(wins):
            winner, round_num = wins[i][0].strip(), int(wins[i][1])
        else:
            winner, round_num = None, None
        results.append({
            "label": f"{batch.team_a}-vs-{batch.team_b}-on-{batch.maps}-run{run_index}",
            "team_a": batch.team_a,
            "team_b": batch.team_b,
            "maps": batch.maps,
            "winner": winner,
            "round": round_num,
            "elapsed_s": per_match_elapsed,
            # A timeout kills the whole worker mid-batch; only the matches
            # that hadn't reported a winner yet are the ones actually lost.
            "timed_out": timed_out and i >= len(wins),
            "returncode": returncode,
        })
    return results


def parse_matchups_file(path: str) -> list:
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b = [x.strip() for x in line.split(",", 1)]
        pairs.append((a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team-a", help="Team A package name (single matchup mode)")
    parser.add_argument("--team-b", help="Team B package name (single matchup mode)")
    parser.add_argument("--matchups", help="Path to a file of 'teamA,teamB' pairs, one per line (sweep mode)")
    parser.add_argument("--maps", default="DefaultSmall", help="Comma-separated map name(s), same as gradle.properties 'maps'")
    parser.add_argument("--games", type=int, default=1, help="How many times to repeat each matchup (default 1)")
    parser.add_argument("--workers", type=int, default=None,
                         help="Max parallel matches (default: P-cores + half of E-cores on Apple Silicon, else CPU count)")
    parser.add_argument("--heap-mb", type=int, default=512, help="Per-match JVM heap cap in MB (default 512)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-match timeout in seconds (default 300)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip the initial ./gradlew compileJava step")
    parser.add_argument("--validate-maps", action="store_true",
                         help="Re-validate the map file on every run instead of skipping it (slower; use if you haven't already confirmed the map is valid)")
    args = parser.parse_args()

    if args.matchups:
        pairs = parse_matchups_file(args.matchups)
    elif args.team_a and args.team_b:
        pairs = [(args.team_a, args.team_b)]
    else:
        parser.error("Provide either --team-a/--team-b or --matchups")

    MATCHES_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    if not args.skip_compile:
        compile_once()

    classpath = build_classpath()
    workers = args.workers or default_worker_count()

    # Split each matchup's games as evenly as possible across the worker
    # pool -- e.g. 50 games / 6 workers = slices of [9,9,8,8,8,8]. Each
    # slice becomes one BatchServer process handling that many matches
    # sequentially, so JVM startup is paid once per slice, not once per
    # match, while the slices themselves still run in true parallel.
    batches = []
    for (a, b) in pairs:
        base, remainder = divmod(args.games, workers)
        start = 1
        for worker_id in range(workers):
            n = base + (1 if worker_id < remainder else 0)
            if n == 0:
                continue
            batches.append(WorkerBatch(a, b, args.maps, start, n, worker_id))
            start += n

    total_games = sum(b.num_games for b in batches)
    print(f"Running {total_games} match(es) across {len(batches)} worker process(es) "
          f"(heap {args.heap_mb}MB each)...", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_worker_batch, batch, classpath, args.heap_mb, args.timeout, args.validate_maps): batch
            for batch in batches
        }
        done_workers = 0
        for future in concurrent.futures.as_completed(futures):
            done_workers += 1
            batch_results = future.result()
            results.extend(batch_results)
            wins = sum(1 for r in batch_results if not r["timed_out"] and r["winner"])
            print(f"[worker {done_workers}/{len(batches)}] {futures[future].label}: "
                  f"{len(batch_results)} matches, {wins} with a clear winner, "
                  f"~{batch_results[0]['elapsed_s']}s/match", flush=True)

    print_summary(results)


def print_summary(results: list):
    print("\n=== Summary ===")
    tally = {}
    for r in results:
        key = (r["team_a"], r["team_b"], r["maps"])
        tally.setdefault(key, {"a_wins": 0, "b_wins": 0, "other": 0, "timeouts": 0, "total": 0})
        bucket = tally[key]
        bucket["total"] += 1
        if r["timed_out"]:
            bucket["timeouts"] += 1
        elif r["winner"] == r["team_a"]:
            bucket["a_wins"] += 1
        elif r["winner"] == r["team_b"]:
            bucket["b_wins"] += 1
        else:
            bucket["other"] += 1

    for (a, b, maps), bucket in tally.items():
        print(f"{a} vs {b} on {maps}: "
              f"{a}={bucket['a_wins']}, {b}={bucket['b_wins']}, "
              f"unclear={bucket['other']}, timeouts={bucket['timeouts']}, "
              f"total={bucket['total']}")

    total_elapsed = sum(r["elapsed_s"] for r in results)
    print(f"\nTotal match-seconds: {total_elapsed:.1f}s across {len(results)} matches "
          f"(logs in {LOGS_DIR})")


if __name__ == "__main__":
    main()
