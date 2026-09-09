#!/usr/bin/env bash
# Autonomous weighted_micro tuning loop.
#
# Each round:
#   1. tune_round.py compiles, benchmarks the current weighted_micro against
#      its last (up to) 5 history versions in both orders on 8 maps, archives
#      the [combat] debug-print logs, and snapshots the just-benchmarked
#      build into a new src/weighted_micro_vN history entry.
#   2. A headless `claude -p` call reads that round's logs/summary, looks
#      for tile scores that seem wrong given what actually happened, and
#      makes one targeted edit to src/weighted_micro/ (CombatTileScore.java
#      and/or CombatState.java).
#   3. Repeat, up to --rounds times.
#
# Stop early at any time: `touch TUNE_STOP` in the project root (checked at
# the top of every round), or Ctrl-C.
#
# Usage: ./tune_loop.sh [--rounds N] [--games N] [--budget USD] [--model NAME]

set -uo pipefail
cd "$(dirname "$0")"

ROUNDS=10
GAMES=2
BUDGET=1.50
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --games) GAMES="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# macOS ships neither GNU `timeout` nor `gtimeout` by default, so implement
# a portable equivalent instead of depending on either being installed.
run_with_timeout() {
    local secs="$1"; shift
    "$@" &
    local pid=$!
    ( sleep "$secs"; kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
}

STOP_FILE="TUNE_STOP"
rm -f "$STOP_FILE"

echo "[tune_loop] starting: up to $ROUNDS round(s), $GAMES game(s)/pairing, \$$BUDGET/round budget for the AI tuning step"
echo "[tune_loop] touch $STOP_FILE at any time to stop cleanly after the current round"

ROUNDS_COMPLETED=0
for ((i = 1; i <= ROUNDS; i++)); do
    if [[ -f "$STOP_FILE" ]]; then
        echo "[tune_loop] $STOP_FILE present, stopping before round $i"
        rm -f "$STOP_FILE"
        break
    fi

    echo ""
    echo "==================== round $i/$ROUNDS ===================="

    ROUND_OUTPUT=$(python3 tune_round.py --games "$GAMES")
    ROUND_EXIT=$?
    ROUND_JSON=$(echo "$ROUND_OUTPUT" | tail -1)

    if [[ $ROUND_EXIT -ne 0 ]]; then
        echo "[tune_loop] tune_round.py failed (compile error most likely) -- asking claude to fix the build"
        PROMPT="The Battlecode project at $(pwd) currently fails to compile (./gradlew compileJava). \
This is almost certainly from an edit made to src/weighted_micro/ in a previous automated tuning round. \
Read the compiler error, fix ONLY files under src/weighted_micro/ (do not touch src/weighted_micro_v*/, \
which is frozen history), and verify with './gradlew compileJava' that it compiles cleanly before you stop. \
Keep the fix minimal -- restore correctness, don't redesign anything."
    else
        ROUND_DIR=$(echo "$ROUND_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['round_dir'])")
        SUMMARY_FILE=$(echo "$ROUND_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['summary_file'])")
        LOGS_DIR=$(echo "$ROUND_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['worker_logs_dir'])")
        OPPONENTS=$(echo "$ROUND_JSON" | python3 -c "import json,sys; print(', '.join(json.load(sys.stdin)['opponents_faced']))")
        SNAPSHOT=$(echo "$ROUND_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['snapshotted_as'])")

        echo "[tune_loop] round $i benchmark done. Faced: $OPPONENTS. Snapshotted as $SNAPSHOT."

        PROMPT="Tuning round $i for the Battlecode bot at $(pwd), package src/weighted_micro/.

This round's build was just benchmarked (as $SNAPSHOT) against its last 5 predecessor \
versions (weighted_micro_v*), in both orders, on 8 maps, $GAMES game(s) per pairing. Results:
- Win/loss summary: $SUMMARY_FILE
- Per-match worker logs (contain '[combat] hp=... tileScoresNoAct=[...] tileScoresAct=[...] \
actionScores(attack,trap,throw,carry)=... action=... damageDealt=...' debug prints from every \
combat turn, plus '[server] ... wins' lines): $LOGS_DIR/*.log

Your job:
1. Skim $SUMMARY_FILE first for the win-rate picture, then sample several worker logs (grep for \
'[combat]' and '[server]') -- you don't need to read every line of every log, just enough to spot \
patterns.
2. Look for turns where a tile's score in tileScoresAct/tileScoresNoAct looks wrong given the \
actual situation and what happened next (e.g. the highest-scored tile walks into a bad trade, a \
clearly safe/aggressive tile is scored too low, carry/throw/trap/attack are picked when another \
action was obviously better, HP drops sharply right after a move that scored well, etc).
3. Trace that back to the specific constant, weight, or piece of logic in \
src/weighted_micro/CombatTileScore.java (primarily) or src/weighted_micro/CombatState.java that's \
responsible.
4. Make ONE targeted, well-reasoned change to fix it. Don't refactor, don't touch multiple unrelated \
things, don't touch anything under src/weighted_micro_v*/ (that's frozen history, never edit it).
5. Run './gradlew compileJava' to confirm it still compiles.
6. Append a short note to $ROUND_DIR/notes.md: what looked wrong, what you changed, and why. Keep it \
to a few sentences.

Then stop -- the outer loop will benchmark your change next round."
    fi

    if [[ -n "$MODEL" ]]; then
        MODEL_ARGS=(--model "$MODEL")
    else
        MODEL_ARGS=()
    fi

    run_with_timeout 1800 claude -p "$PROMPT" \
        --permission-mode acceptEdits \
        --allowedTools "Read" "Edit" "Grep" "Glob" "Bash(./gradlew compileJava*)" \
        --max-budget-usd "$BUDGET" \
        --output-format text \
        "${MODEL_ARGS[@]}"
    CLAUDE_EXIT=$?
    if [[ $CLAUDE_EXIT -ne 0 ]]; then
        echo "[tune_loop] claude -p exited nonzero ($CLAUDE_EXIT) this round -- continuing to next round anyway"
    fi

    ROUNDS_COMPLETED=$((ROUNDS_COMPLETED + 1))
done

echo ""
echo "[tune_loop] done ($ROUNDS_COMPLETED round(s) completed)."

if [[ $ROUNDS_COMPLETED -gt 0 ]]; then
    git add src/weighted_micro src/weighted_micro_v* 2>/dev/null
    if ! git diff --cached --quiet; then
        git commit -m "$(cat <<EOF
Autonomous weighted_micro tuning: $ROUNDS_COMPLETED round(s)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
        echo "[tune_loop] committed tuning changes."
    else
        echo "[tune_loop] nothing to commit (no changes under src/weighted_micro or src/weighted_micro_v*)."
    fi
fi
