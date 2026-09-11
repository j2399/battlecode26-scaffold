#!/usr/bin/env bash
# Evolutionary (random-mutation) tuning loop for weighted_micro's combat
# scoring constants. No AI reasoning and no external `claude` CLI call --
# just mutate-and-select, each generation:
#
#   1. evo_round.py takes the current 2 baseline packages and fields them
#      unmutated alongside 5 mutated candidates (3 from baseline 1, 2
#      from baseline 2 -- every named constant in CombatTileScore.java,
#      each scaled by its own independent +-20% roll) -- 7 competitors
#      total. Compiles, then round-robins all 7 against each other
#      (every ordered pair, so both orders of every matchup, across 5
#      maps -- evileye/thunderdome/knifefight excluded as pure
#      positional-bias noise, see evo_round.py).
#   2. The top-2 by total wins become next generation's baselines (state
#      persisted in evo_state.json, so this resumes correctly across
#      separate runs of this script). Since the unmutated baselines are
#      in the field, a generation of worse mutations can't regress --
#      the baselines just get reselected.
#   3. Repeat, up to --generations times.
#
# Stop early at any time: `touch EVO_STOP` in the project root (checked at
# the top of every generation), or Ctrl-C.
#
# Usage: ./evo_loop.sh [--generations N] [--games N]

set -uo pipefail
cd "$(dirname "$0")"

GENERATIONS=50
GAMES=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --generations) GENERATIONS="$2"; shift 2 ;;
        --games) GAMES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

STOP_FILE="EVO_STOP"
rm -f "$STOP_FILE"

echo "[evo_loop] starting: up to $GENERATIONS generation(s), $GAMES game(s)/pairing"
echo "[evo_loop] touch $STOP_FILE at any time to stop cleanly after the current generation"

for ((i = 1; i <= GENERATIONS; i++)); do
    if [[ -f "$STOP_FILE" ]]; then
        echo "[evo_loop] $STOP_FILE present, stopping"
        rm -f "$STOP_FILE"
        break
    fi

    echo ""
    echo "==================== generation $i/$GENERATIONS ===================="

    GEN_OUTPUT=$(python3 evo_round.py --games "$GAMES")
    GEN_EXIT=$?
    GEN_JSON=$(echo "$GEN_OUTPUT" | tail -1)

    if [[ $GEN_EXIT -ne 0 ]]; then
        echo "[evo_loop] evo_round.py failed -- stopping"
        echo "$GEN_OUTPUT"
        break
    fi

    echo "$GEN_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"[evo_loop] generation {d['generation']} done.\")
for c in d['ranked']:
    info = d['candidate_info'][c]
    changes = ', '.join(info['changes']) if info['changes'] else '(no changes recorded)'
    print(f\"  {c} (from {info['parent']}): {d['wins'].get(c, 0)} wins -- {changes}\")
print(f\"[evo_loop] new baselines: {d['new_baselines']}\")
"
done

echo ""
echo "[evo_loop] done."
