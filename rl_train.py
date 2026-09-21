#!/usr/bin/env python3
"""
Milestone 2+ of the learner_rl RL pipeline: REINFORCE (with an importance-
sampling correction for the epsilon-greedy behavior policy, and a simple
mean-return baseline) using trajectories collected via rl_collect.py.

The 9 raw outputs learner's net produces (originally trained via MSE
regression to imitate econ5's tile scores) are reinterpreted here as
categorical policy logits: pi_theta(a|s) = softmax(net(s)). Epsilon-greedy
at inference (argmax most of the time, uniform-random with prob EPSILON
otherwise -- see learner_rl/RobotPlayer.java) is exploration around that
same categorical policy's mode, so no architecture change is needed, only
a different loss and a different interpretation of the same 9 numbers.

Each iteration: pick an opponent (diverse roster -- see STATIC_OPPONENTS
and the self-checkpoint mechanism below; this is the DLT fix for the
self-play overfitting the evo loop already demonstrated), collect a batch
of learner_rl vs that opponent, turn the result into (state, action,
return, behavior_prob) tuples, take one REINFORCE gradient step, export
the updated weights back into learner_rl's Java source, recompile, and
repeat. Every --checkpoint-every iterations, snapshot the current weights
into a new learner_rl_ckptN package and add it to the opponent roster
permanently, so later iterations occasionally face any past version of
itself, not just recent ones -- self-play is still only ever sampled with
probability (1 - STATIC_OPPONENT_PROB), so a growing self-checkpoint count
can't crowd the static opponents out of the roster by sheer numbers the
way it did before that fix (a 200-iteration run on the unbounded roster
made held-out win rate *worse*, not better -- see the comment on
STATIC_OPPONENT_PROB below); a separate, later fix removed a rolling
recency window that had been layered on top of that one, after it made
old self-checkpoints permanently unreachable and let the policy regress
against them unnoticed (see the comment above STATIC_OPPONENT_PROB).

Collection is restricted to a small set of combat-focused maps (TRAIN_MAPS),
both orders per map, per iteration -- see the comment above TRAIN_MAPS below.
"""

import argparse
import json
import math
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import StudentTileNet, TeacherTileNet, export_neuralnet_java  # reuse existing arch + Java exporter
from rl_collect import run_batch, LEARNER

# Collection was originally cut down to just "micromap" (both orders) to
# get a clean, cheap per-iteration result instead of spreading each
# iteration's batch across all 5 of rl_collect.MAPS, several of which
# involve enough map traversal/economy that the RL-trained policy's own
# combat/movement decisions aren't the main thing determining who wins --
# diluting the training signal with outcomes the policy barely
# influences. But collecting on only one map cut fresh trajectories per
# iteration from ~8000-18000 (the old 5-map batch) down to ~1500-3500,
# which slowed REPLAY_CAPACITY's 40000-step buffer from turning over every
# ~3-4 iterations to every ~15-20 -- since build_batch's GAE advantages/
# value-targets are computed once at collection time and never refreshed
# as training continues (see its docstring), most gradient steps were
# being taken against advantages computed relative to a policy from 15-20
# iterations ago. That's a plausible cause of mean_advantage oscillating
# instead of settling after the switch to micromap-only collection. Fix:
# these additional small, combat-focused maps (trainingmap1-7) restore a
# similar volume of fresh data per iteration to the original 5-map setup,
# while keeping every map one where combat/movement -- not economy or
# long-range navigation -- decides the outcome.
TRAIN_MAPS = [
    "micromap",
    "trainingmap1", "trainingmap2", "trainingmap3", "trainingmap4",
    "trainingmap5", "trainingmap6",
    # trainingmap7 deliberately excluded: every match on it this session --
    # in both training-era logs and the ckpt round-robin -- ran far longer
    # than any other map (900-1000+ rounds vs. 100-400 elsewhere), with no
    # clear winner in many cases even after that. Some structural property
    # of this specific map (likely a stalemate-prone layout) rather than
    # anything model-specific, since it showed up across every checkpoint
    # tested. Including it was diluting each iteration's batch with mostly
    # uninformative (no decided outcome) data and skewing per-map win-rate
    # comparisons.
]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "learner_rl.pth"
VALUE_MODEL_PATH = PROJECT_ROOT / "learner_rl_value.pth"  # training-time-only baseline net, never exported to Java
REFERENCE_MODEL_PATH = PROJECT_ROOT / "learner_rl_reference.pth"  # frozen DAPO snapshot, refreshed periodically
IL_MODEL_PATH = PROJECT_ROOT / "qnet.pth"  # original imitation-learned student weights, used to warm-start
TEACHER_IL_MODEL_PATH = PROJECT_ROOT / "teacher.pth"  # IL-trained TeacherTileNet weights (train.py) -- RL now trains THIS network, not the small student directly (see build_warm_start_model)
DISTILLED_STUDENT_PATH = PROJECT_ROOT / "learner_rl_distilled_student.pth"  # final KL-distilled small student (see distill_to_student_kl) -- what's actually exported into learner_rl for deployment at the end of a run. Deliberately separate from IL_MODEL_PATH (qnet.pth), which train_exploiter's reset-to-baseline still needs as the original, structurally distinct pre-RL policy to diverge from -- not this run's own end result.
STATE_PATH = PROJECT_ROOT / "rl_state.json"  # persists the global iteration count + self-checkpoint roster across separate invocations
REPLAY_BUFFER_PATH = PROJECT_ROOT / "rl_replay_buffer.pt"  # persists collected steps across iterations/invocations

# 18 original + ownHealth + nearest-ally(distance,bearing,health,direction)
# + local health advantage + 2nd/3rd-nearest ally -- see RobotPlayer.java's
# buildState(). Every enemy/ally slot's first two fields are (distance,
# bearing-from-north) rather than raw (dx,dy) offsets -- see
# SYMMETRY_ANGLE_FIELDS below for the transform this implies under the 4x
# symmetry augmentation, and buildState()'s own comment for the atan2
# convention. Own and enemy orientations were already present in the
# original 18 (dirToInt(rc.getDirection()) and e1d/e2d/e3d); the ally's
# facing direction was the one orientation missing. Field 24 (added after
# reports of overly defensive play even with a clear numbers/health edge)
# sums health across every currently-sensed ally (incl. self) minus every
# currently-sensed enemy: previously only the *nearest* ally/enemies were
# individually visible, giving the policy no way to distinguish "we
# outnumber them here" from "it's me alone against three of them" even
# when both look identical from the nearest-enemy fields alone. Fields
# 25-32 (added while investigating this engine's first-move advantage
# swarming a lone/first-move-disadvantaged rat) extend ally tracking from
# 1 to 3 nearest, mirroring the enemy fields -- a single nearest-ally
# reading can't tell "I have one ally right behind me" from "I'm in the
# middle of a group," which matters for judging whether a fight (or
# getting swarmed) is actually winnable. Fields 33-40 are one flag per
# compass direction for whether rc.canMove() currently allows moving that
# way -- reusing that call (already made every turn in the movement
# fallback cascade, so known to be bytecode-affordable) instead of
# rc.senseMapInfo() (constructs a whole wall/dirt/trap/cheese object,
# pricier per call for a question that doesn't need that detail); it
# deliberately reflects transient occupancy too, not just permanent
# terrain, since "blocked by a wall" and "blocked by an ally standing
# there" both mean the same thing for a movement decision made this turn.
# qnet.pth was trained with the original 18, so warm-starting now needs a
# partial weight transfer (see build_warm_start_model) rather than a
# plain load_state_dict.
STATE_DIM = 64
NUM_TILES = 9

# Bump any time a state field's SEMANTICS change without STATE_DIM itself
# changing -- e.g. this session's dx,dy -> (distance, bearing) change kept
# every field count identical, so a shape-only staleness check (see
# load_replay_buffer) can't detect it on its own. 2 = polar-coordinate
# enemy/ally location fields (was 1 = raw dx,dy offsets).
STATE_FORMAT_VERSION = 2

# ValueNet-only input width: the policy's own STATE_DIM plus 2 team/
# opponent summary scalars (this round's team-average raw reward, this
# round's opponent hp_delta -- see TEAM_SPIRIT/OPPONENT_ZERO_SUM_SCALE
# below) that the reward now depends on but no single robot's own state
# encodes, PLUS 4 more: both sides' king health and cheese (see
# KING_HEALTH_SCALE/CHEESE_SCALE and rl_collect.py's
# king_and_cheese_by_round). Found via direct measurement: R^2 of the
# value net's predictions against actual value_targets, computed on the
# live replay buffer, was only 0.059 -- barely better than predicting the
# mean -- after team-spirit/zero-sum made a real chunk of the reward's
# variance depend on teammates'/the opponent's situation, information the
# value net (state-only input) structurally can't see. The team_avg/
# opponent_hp_delta fix helped but didn't close the gap (R^2 still
# negative after hundreds of iterations in an earlier run) -- king
# health/cheese are added because they're close to the actual win
# condition itself, and the value net previously had to infer "are we
# winning this match" only indirectly through a noisy combat-reward
# proxy, never seeing either king's health or either side's cheese
# directly. ValueNet is training-time-only (never exported to Java --
# the deployed policy still only ever sees its own STATE_DIM-wide state,
# unchanged, and deliberately never sees king health at all -- see
# buildState()'s comment on why it's excluded from the combat-balance
# feature), so this only affects how well training's own credit
# assignment works, not what the deployed bot can act on.
VALUE_STATE_DIM = STATE_DIM + 6

# King health caps at 600 (vs. a baby rat's 100 -- see buildState()'s
# comment on why king health is excluded from the policy's own combat-
# balance signal). Cheese has no fixed cap; measured directly against a
# real training-map log: climbed to ~2500 by round 223 of a ~300-round
# match. 500 keeps typical mid-match values in roughly the same [0, a
# few] order of magnitude as the rest of this (mostly [-1,1]-normalized)
# extra-feature block, rather than letting an unbounded, still-growing
# quantity dominate early gradients the way the unnormalized opp_hp_delta
# once did (see OPPONENT_ZERO_SUM_SCALE's /100 fix for the same class of
# bug).
KING_HEALTH_SCALE = 600.0
CHEESE_SCALE = 500.0

# Hidden-layer width. Was doubled 24->48 earlier for state capacity, but
# that turned out to have a real, previously-unmeasured cost: profiling
# real matches at the true bytecode limit (17,500/turn for BABY_RAT --
# this pipeline's own training/eval harness runs with that limit removed
# entirely, see bc-overrides-src/UnitType.java) found nn.forward() alone
# -- fully unrolled into literal Java multiply-adds by export_neuralnet_
# java, so its cost scales directly with parameter count -- consuming
# ~14,000 of the 17,500 budget on every turn with an enemy visible, on
# every map tested. At 48 hidden units that's ~80% of the *entire*
# per-turn budget gone to the forward pass alone, before buildState(),
# autoActions(), or movement get a chance to run, and every real match
# examined was landing within a few bytecode of the hard cap. Shrunk to
# 32 to buy back real headroom. Every place that builds a
# StudentTileNet/ValueNet or exports one to Java must pass this
# explicitly (see the dimension-mismatch checks in main() and
# build_warm_start_model below) -- export_neuralnet_java has no way to
# detect a mismatch between the `hidden` it's told and the model's
# actual weight shape; passing the wrong value there silently generates
# truncated or out-of-range code instead of failing loudly.
#
# No longer the main policy's own hidden width -- RL now trains
# TeacherTileNet directly (fixed 256/128/64, see build_warm_start_model),
# which never runs under the real bytecode limit at all (only ever
# exported via export_teacher_neuralnet_java, used solely for training-
# time self-play under bc-overrides' unlimited budget). This constant now
# governs ValueNet (also training-time-only, so the bytecode rationale
# above doesn't really apply to it either -- kept the same width mainly
# for convenience) and the FINAL distilled student (distill_to_student_kl)
# -- the one network that still actually needs to fit the real 17,500
# budget, since it's what gets exported into learner_rl for deployment.
HIDDEN_DIM = 32
EPSILON = 0.33  # must match learner_rl/RobotPlayer.java's EPSILON -- raised from 0.1, see that file's comment
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
PPO_CLIP_EPS = 0.2

# Clip-Higher (DAPO, Yu et al. 2025): standard PPO clips the ratio
# symmetrically around 1, which caps how far a low-probability action's
# ratio can rise even when the update is genuinely good news for it --
# the same "minority actions structurally suppressed" mechanism DUAL_CLIP_C
# addresses on the downside (negative advantage), just on the upside
# (positive advantage) instead. DAPO's own writeup: symmetric clipping
# "restricts the exploration of low-probability tokens" and entropy
# collapses even with dual-clip's downside floor already in place unless
# the upside is also widened. PPO_CLIP_EPS_HIGH is the paper's own value
# (they report tuning it well above the standard 0.2 low side).
PPO_CLIP_EPS_HIGH = 0.28

# Dual-clip PPO (Ye et al. 2019): standard vanilla-PPO clipping only bounds
# the *optimistic* side (large ratio, positive advantage); for negative
# advantage a large ratio is deliberately left unbounded -- fine for
# vanilla PPO, where pi_old is a recent snapshot of pi_theta so ratio
# rarely strays far from 1, but this pipeline's behavior policy is a
# persistent epsilon-greedy mixture whose denominator for a non-argmax
# action can be as small as EPSILON/9=0.011, so ratio routinely reaches
# several multiples of 1 even for perfectly ordinary samples. Confirmed on
# the actual replay buffer: for actions the current model *doesn't* favor,
# the mean unclipped surrogate on negative-advantage samples was -0.538 vs.
# a fairly-clipped -0.271 for the same samples -- roughly double the
# effective penalty a favored action's negative-advantage samples get
# (-0.205 vs -0.265, i.e. no such penalty at all). That asymmetry is a
# structural ratchet: whichever action currently leads keeps facing a
# lighter downside than every alternative, which is a fully sufficient
# explanation for the collapse-onto-one-action pattern observed twice now
# (first CENTER, then -- after a warm-start bias fix specifically for
# CENTER -- EAST/WEST) regardless of which action started with the edge.
# DUAL_CLIP_C floors the negative-advantage loss the same way clip_eps
# already floors the positive-advantage one, at the paper's standard value.
DUAL_CLIP_C = 3.0

# Experience replay: collection (real matches) is the expensive, slow part
# of this pipeline (~50s/iteration); gradient steps on already-collected
# data are nearly free. Keeping a rolling buffer of recent steps and
# training PPO epochs over that instead of only the freshly-collected
# batch extracts several iterations' worth of gradient signal per match
# actually played, instead of throwing each batch away after one use.
# PPO's existing behavior-probability ratio already makes this sound: the
# ratio corrects for however far pi_theta has drifted from whatever
# policy actually generated a given sample, however old that sample is.
REPLAY_CAPACITY = 40000  # steps; roughly 4-5 iterations' worth of fresh data
REPLAY_TRAIN_STEPS = 20000  # how many steps to sample from the buffer for each update

# Denser per-turn shaping, both on the same "HP of damage" footing as the
# +-1 terminal win/loss reward (a full-health swing is worth about +-1):
# hpDelta is already raw HP (0-100 range), and actionValue (bites, thrown-
# rat collisions, capturing an enemy -- see RobotPlayer.java's
# autoActions()) is logged in the same raw-HP units instead of a separate
# ad hoc bonus constant per action type.
#
# ACTION_VALUE_SCALE (dealing damage) and HP_DELTA_SCALE (taking damage)
# used to be the same constant, which made a perfectly even trade (deal X,
# take X back) net to exactly zero reward -- engaging only ever paid off
# by actually winning the exchange, with no positive baseline for trading
# at all. Combined with the terminal win/loss reward being sparse and, at
# GAMMA=0.99, considerably decayed by the time GAE propagates it back
# through a 100-300+ round match, an even-or-uncertain trade could easily
# look worse than never engaging at all -- a plausible cause of observed
# overly defensive play (declining to attack even with a numbers/health
# edge). Split so dealing damage is worth slightly more than the same
# amount of damage taken costs, giving engagement a small built-in
# positive baseline instead of requiring a decisive win to be worthwhile.
ACTION_VALUE_SCALE = 0.011
HP_DELTA_SCALE = 0.01

# Getting captured takes this robot fully out of the fight -- symmetric
# with actionValue's CAPTURE_DAMAGE_EQUIVALENT for capturing an enemy
# (RobotPlayer.java), just on the losing end instead of the winning one.
CAPTURE_DAMAGE_EQUIVALENT = 25.0

# Proximity shaping (potential-based, Ng/Harada/Russell 1999): a dense,
# every-turn reward for closing distance to the nearest tracked enemy,
# on top of the existing hp_delta/action_value terms. Motivated by a
# real, measured gap: GAMMA=0.99 discounts the terminal win/loss bonus
# to near-nothing by the time GAE propagates it back through a 150-300
# round match, and a reward_signal_check.py run found the existing dense
# reward correlates only weakly with actually winning (r=0.315) --
# unsurprising, since it only measures combat trades, and matches are
# decided by whether either king actually gets destroyed, not by the
# aggregate skirmish record along the way.
#
# Phi(state) = -distance_to_nearest_tracked_enemy, or -PROXIMITY_FAR_
# SENTINEL when none is tracked (e1h==0 is the same "no enemy" padding
# value used elsewhere in this state). Expressed as gamma*Phi(s') -
# Phi(s), not a raw "+bonus for being close", specifically because a raw
# bonus is the textbook reward-hacking shape (Coast Runners: farm the
# green-block bonus by circling instead of finishing the race) -- an
# agent could hover just inside a raw bonus's radius forever. The
# potential-difference form doesn't have that hole: shaping rewards
# telescope to exactly zero over any round trip in Phi-space, so e.g. a
# BABY_RAT spinning in place to sweep an enemy in and out of its narrow
# 90-degree vision cone (toggling e1h, and so Phi, without any real
# distance change) nets to zero once it's out of view again -- only
# genuine, lasting progress pays out.
#
# PROXIMITY_SHAPING_SCALE keeps this comparable in magnitude to the
# existing terms rather than dominating them: a full "enemy newly
# spotted" jump (sentinel 64 -> a real in-vision-range distance, at most
# ~4.5 given BABY_RAT's vision radius) is roughly PROXIMITY_FAR_SENTINEL
# * this scale =~ 0.6-0.9, similar order of magnitude to a single big
# hit (hp_delta*HP_DELTA_SCALE) or a capture penalty
# (CAPTURE_DAMAGE_EQUIVALENT*HP_DELTA_SCALE=0.25), not larger.
PROXIMITY_SHAPING_SCALE = 0.01
PROXIMITY_FAR_SENTINEL = 64.0


def enemy_potential(state) -> float:
    # state[2] is now distance directly (see the state-layout comment on
    # SYMMETRY_ANGLE_FIELDS below) -- no sqrt needed anymore.
    e1dist, e1h = state[2] * 64.0, state[4] * 100.0
    if e1h <= 0:
        return -PROXIMITY_FAR_SENTINEL
    return -e1dist


# Team-spirit reward blending + zero-sum opponent normalization (OpenAI
# Five, Dota 2): with many of our own robots on the field at once, each
# one's raw reward so far only reflects its OWN combat/proximity events,
# with no credit for helping a teammate win ITS fight, and no penalty
# tied to how the OPPONENT team is actually doing overall -- two
# specific gaps in a genuinely multi-agent setting that potential-based
# shaping alone doesn't address (it's still a single-robot signal).
#
# TEAM_SPIRIT blends each robot's own per-round reward with that round's
# average across every one of our robots in the same match:
#   blended_i = TEAM_SPIRIT * team_avg + (1 - TEAM_SPIRIT) * individual_i
# matching OpenAI Five's hero_rewards[i] = tau*mean(hero_rewards) +
# (1-tau)*hero_rewards[i] exactly. At 0, every robot only ever sees its
# own trades (today's behavior); at 1, all of a team's robots share one
# fully pooled reward. Started low (their own tau was annealed 0.2->0.97
# over a vastly longer training run than this project's) since an
# under-trained team-pooled signal can just as easily teach every robot
# to freeride on whichever teammate happens to be trading well.
#
# OPPONENT_ZERO_SUM_SCALE subtracts a scaled version of the opponent
# team's own per-round hp_delta (see rl_collect.py's
# opponent_hp_delta_by_round -- reuses the already-present "[stats]"
# line every rat prints, no new logging needed) from our reward each
# round -- OpenAI Five's "subtract the opposing team's average reward"
# step, adapted to what's actually observable about an opponent here
# (we can see their health over time via "[stats]" even for packages
# that never print our RL-specific "[traj]" lines, but not their own
# per-robot reward breakdown). Deliberately a per-robot delta between
# consecutive rounds, not a raw total-health difference: checked
# directly against a real log and found a team's total health climbs
# steadily just from reinforcements spawning (each a fresh ~100hp baby
# rat, a brand new robot id with no prior round to diff against), which
# would otherwise get counted as "the opponent doing great" even in
# rounds where every one of their EXISTING units is actively losing a
# fight. This catches outcomes our own hp_delta/action_value never could
# -- e.g. their king starving from a cheese shortage our combat pressure
# caused shows up nowhere in what WE logged, only in THEIR health
# dropping -- while keeping the same HP-based scale as the rest of the
# reward (HP_DELTA_SCALE) so it doesn't dominate. Falls back to 0
# contribution for opponents that don't print "[stats]" at all
# (econ5_no_cheese, weighted_micro -- see final_team_stats) or for a
# round with no robot seen in back-to-back rounds.
TEAM_SPIRIT = 0.3
OPPONENT_ZERO_SUM_SCALE = HP_DELTA_SCALE

# DAPO: KL penalty against a reference snapshot -- discourages the policy
# thrashing between iterations. Originally a periodic hard refresh (every
# --dapo-refresh-every iterations, the reference became an exact copy of
# whatever the policy currently was). That bounds how far the policy can
# move in any single refresh window, but does nothing to stop a slow net
# decline over many windows: each refresh just locks in wherever drift
# had taken the policy, good or bad, so the "stability" term couldn't
# actually resist a sustained regression. A 30-iteration validation batch
# after fixing the replay-buffer-staleness issue (see TRAIN_MAPS above)
# confirmed mean_advantage's *oscillation pattern* was healthy again (no
# more multi-iteration sustained drift), yet overall win rate against the
# self-checkpoint roster was still only 39.6% -- the anchor not resisting
# decline is a plausible piece of that. Replaced with a per-iteration EMA
# (see DAPO_EMA_DECAY / update_dapo_reference below): the anchor still
# moves, but far more slowly than the policy itself, so it resists
# sustained drift instead of resetting its baseline every refresh.
DAPO_KL_COEF = 0.1

# IL teacher anchor: a second, FIXED KL-stability term against the
# original IL-trained teacher (teacher.pth, loaded once at the start of
# main() and never updated) -- distinct from DAPO_KL_COEF's reference,
# which is an EMA that drifts along with the policy itself and so cannot
# resist a slow, sustained decline (by design: it's meant to smooth
# iteration-to-iteration thrashing, not anchor against multi-hundred-
# iteration drift). Added after a full round-robin tournament across 29
# checkpoints from a 599-iteration run found a strong negative correlation
# (-0.90) between iteration number and win rate: the policy peaked around
# iteration 80-100 and degraded almost monotonically for the rest of the
# run, confirmed head-to-head (early checkpoints beat late ones ~80% of
# the time), not just an aggregate artifact. This mirrors AlphaStar's own
# documented fix for the identical symptom ("the agent continuously
# forgot how to win against a previous version of itself"): weaving
# imitation-learning distillation continuously through training, not just
# once at warm-start. Implemented as compute_dapo_kl(model, il_teacher,
# states) -- reuses that function as-is, since it's already generic over
# *which* reference model it compares against, not DAPO-specific despite
# the name.
#
# 0.02, a fifth of DAPO_KL_COEF: deliberately gentle and UNTUNED (no A/B
# comparison was run before this training job) -- meant as a soft pull
# back toward sound fundamentals, not a hard constraint that would
# prevent the policy from ever improving past the IL teacher's own
# (mediocre) play. If degradation still shows up in the next round-robin
# check, raise this; if the policy seems unable to improve past early
# checkpoints at all, lower it.
IL_ANCHOR_COEF = 0.02

# How much weight the reference snapshot keeps of itself each iteration
# (vs. blending in the current policy) -- reference = DECAY*reference +
# (1-DECAY)*current, applied every iteration. 0.99 means the reference's
# effective averaging window is roughly 1/(1-0.99) = 100 iterations: slow
# enough to meaningfully resist a multi-iteration decline (unlike the old
# every-5-iterations hard refresh, which had zero memory beyond that
# window), while still eventually following genuine, sustained
# improvement rather than anchoring forever to some early point.
DAPO_EMA_DECAY = 0.99

# RGPS: NOT a port of weighted_micro's CombatTileScore -- that scores full
# MapLocation/RobotInfo/terrain context this 18-dim state vector doesn't
# carry (no walls, no ally positions, no map bounds). This is a much
# smaller, verifiably-correct proxy for the same idea (inject one
# high-confidence rule instead of making RL rediscover it from scratch):
# when the nearest enemy is adjacent and already low on health, softly
# prefer whichever move direction points toward it. Direction deltas below
# were read directly off the real battlecode.common.Direction enum
# (dx/dy), not assumed, to avoid a silent sign-convention bug -- order
# matches allDirections in RobotPlayer.java: CENTER,N,NE,E,SE,S,SW,W,NW.
#
# CONF_DIST_SQ widened from 2.0 (~1.4 tiles, essentially already-adjacent)
# to 16.0 (~4 tiles, well inside vision cone radius 20): found via replay-
# buffer analysis that CENTER's dominance (argmax==CENTER for 99.97% of
# 40000 states, even after a 5x ENTROPY_COEF bump did nothing) isn't from
# movement being actively penalized turn-by-turn -- mean advantage for
# move vs. stay was statistically the same (-0.021 vs -0.021 just outside
# the old 1.4-tile trigger, -0.094 vs -0.090 right at it). It's a
# self-reinforcing fixed point instead: once CENTER edges ahead, on-policy
# collection samples it ~90% of the time, starving PPO's advantage
# estimator of the alternative-action data it'd need to push back out.
# RGPS sidesteps that entirely -- it's a supervised cross-entropy pull
# toward the correct direction, independent of advantage/exploration --
# but its old 1.4-tile radius only ever engaged once basically adjacent,
# too late to influence the actual approach decision. Coefficient also
# raised 0.05->0.1 since it now needs to out-pull a much more entrenched
# CENTER bias than when it was originally tuned.
RGPS_COEF = 0.1
RGPS_CONF_DIST_SQ = 16.0   # enemy within ~4 tiles
RGPS_CONF_HEALTH = 30.0   # out of 100
# True bite-adjacency for a BABY_RAT (GameConstants.ATTACK_DISTANCE_SQUARED
# in the engine source, confirmed directly) -- distinct from the much
# looser RGPS_CONF_DIST_SQ above ("worth closing the gap at all"). Used
# to stop rgps_loss unconditionally zeroing CENTER's target weight even
# when the robot is already close enough to bite, where closing further
# is actually wrong. See rgps_loss's docstring.
ATTACK_DISTANCE_SQUARED = 2.0
DIRECTION_DELTAS = torch.tensor([
    [0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1],
], dtype=torch.float32)
# Each direction's own bearing-from-north, in radians -- same atan2(dx,dy)
# convention as buildState()'s e1x/e1y (see SYMMETRY_ANGLE_FIELDS below).
# CENTER's (0,0) gives atan2(0,0)=0 here but is never used: rgps_loss zeros
# its alignment column explicitly, matching the old dirs_norm behavior.
DIRECTION_BEARINGS = torch.atan2(DIRECTION_DELTAS[:, 0], DIRECTION_DELTAS[:, 1])

# Entropy bonus: subtracted from the policy loss (i.e. added to what's
# being maximized) to penalize the policy for collapsing toward a
# near-deterministic distribution. The clipped PPO surrogate alone has
# nothing discouraging that collapse -- standard PPO implementations
# (OpenAI Baselines, Stable-Baselines3, etc.) always pair it with an
# entropy term for exactly this reason, and this pipeline never had one.
# Consequence, found by inspecting the actual trained policy after ~900
# iterations: argmax(q) == CENTER (tile 0, "don't move") for literally
# 100% of 40000 sampled replay-buffer states, and the action distribution
# in the buffer was 91.1% CENTER with the rest split ~evenly across the
# other 8 directions -- exactly what pure EPSILON-random exploration
# alone would produce, meaning the *learned* policy had zero remaining
# movement preference; every bit of apparent positioning behavior in
# matches was coming from the random 10% exploration branch. An earlier
# checkpoint (iteration 125) was already biased toward CENTER (69.6%) but
# still showed real state-dependent variation (one direction fired far
# above its random-exploration baseline), so this was a *progressive*
# collapse over training, not a bug introduced by one change -- plausibly
# because "don't move" reliably avoids hpDelta penalties while engaging
# carries risk the noisy value function hadn't learned to credit
# properly, and nothing was pushing back against PPO reinforcing that
# every iteration until it took over completely. 0.01 matches the
# commonly-used PPO default; not tuned further yet.
#
# Update after retraining from scratch with this fix active from
# iteration 0 (462 iterations in): checked again and CENTER is still
# argmax for 99.97% of 40000 replay-buffer states. The raw confidence
# collapse is gone -- per-state CENTER probability is ~0.66-0.70 (not
# ~1.0), and the other 8 actions show real, state-varying spread in the
# remaining mass -- so 0.01 did stop the policy from going fully
# deterministic and kept a PPO gradient signal alive. But it's not
# enough weight to unseat CENTER as the top-ranked action anywhere.
# RGPS_COEF*rgps_loss (~0.05*3.15=0.16) still dwarfs
# ENTROPY_COEF*entropy (~0.01*1.31=0.013) in the policy_loss sum, so the
# entropy term has very little leverage relative to the other terms.
# Raised 5x to put it in the same range as RGPS's contribution.
ENTROPY_COEF = 0.05

# Targets a DIFFERENT failure mode than ENTROPY_COEF above: per-sample
# entropy measures how sharp each individual state's prediction is, but
# says nothing about which action wins argmax *in aggregate* across many
# states -- a policy can stay confidently spread per-state (healthy
# entropy) while still picking the same 1-2 actions almost every time
# (collapsed argmax), since entropy is a property of one softmax, not of
# the population of argmax winners. Concretely observed: at iteration 100
# of a run with a verified-NEUTRAL warm start (uniform-calibrated anchor,
# balanced IL labels, balanced state-feature bearings -- ruled out
# inherited bias directly), argmax(E)+argmax(W) reached 95.3% of all
# states while entropy was still a moderate 1.67 (down from a fresh-start
# 1.86, not collapsed). This is a self-reinforcing on-policy dynamic
# (documented earlier for CENTER, then E/W/diagonals) that per-sample
# entropy structurally cannot see, let alone resist -- it needs its own
# term. See argmax_share_loss below.
ARGMAX_SHARE_COEF = 0.1

# Count-based exploration bonus: unlike ARGMAX_SHARE_COEF/ENTROPY_COEF
# above (both correct the LOSS after imbalanced data has already been
# collected), this attacks the self-reinforcing collapse at its actual
# source -- the data itself. An under-sampled action's advantage
# estimates stay noisy simply because there's little data to estimate
# them from, which is *why* PPO's gradient barely reinforces it even
# when it'd otherwise be reasonable; standard count-based/UCB-style
# intrinsic motivation (add a bonus to actions taken less often recently)
# directly compensates. Deliberately one-sided (bonus only, never a
# penalty on over-represented actions) -- RGPS/argmax-share/entropy
# already push back against over-concentration; this only needs to make
# rare actions look more worth reinforcing, not punish common ones that
# may be common because they're genuinely good.
ACTION_FREQ_EMA_DECAY = 0.95   # slower than OPPONENT_WIN_EMA_DECAY (0.9) -- a stabler read on "how has the policy actually been behaving lately," not reactive to one noisy iteration
COUNT_BONUS_COEF = 2.0         # roughly comparable scale to mean_advantage (typically 0.5-2 in the logs) so the bonus can actually compete for a collapsed action, not get lost in the noise


def policy_log_probs(raw_output: torch.Tensor) -> torch.Tensor:
    """The net's 9 outputs were trained via MSE regression to match
    econ5's own tile scores, which have no bounded scale -- real logged
    values range in the hundreds (e.g. 372, 445, 520 for one state).
    Feeding that directly into softmax saturates it into a near-
    deterministic one-hot almost everywhere, which then makes any
    cross-entropy-style loss against a target that doesn't match that
    saturated one-hot blow up (observed: RGPS loss in the hundreds,
    correct scale is 0-log(9)=2.2). argmax is invariant to any monotonic
    per-sample rescaling, so this doesn't change deployed behavior (the
    Java side still just argmaxes the raw output) -- it only fixes how
    training interprets these outputs as a stochastic policy."""
    standardized = (raw_output - raw_output.mean(dim=-1, keepdim=True)) / (raw_output.std(dim=-1, keepdim=True) + 1e-6)
    return F.log_softmax(standardized, dim=-1)


def rgps_loss(model, states: torch.Tensor) -> torch.Tensor:
    """Soft auxiliary loss toward closing distance on a nearby, weak
    enemy -- see RGPS note above. Returns 0 if no state in the batch meets
    the confidence condition.

    state[:,2]/state[:,3] are now (distance, bearing) rather than (dx, dy)
    -- see SYMMETRY_ANGLE_FIELDS below -- so "cosine similarity between the
    enemy offset and each direction's own unit vector" is now computed
    directly as cos(enemy_bearing - direction_bearing), which is the exact
    same quantity (cos of the angle between the two vectors), just derived
    from the angle instead of reconstructing dx/dy from it.

    CENTER's target weight used to be unconditionally zeroed whenever this
    fires -- an active penalty on CENTER, not a neutral one, every single
    time a weak enemy is nearby, regardless of whether closing further
    was actually useful. Confirmed as a real, distinct asymmetry (not just
    the generic self-reinforcing collapse also affecting CENTER): econ5's
    own ground-truth labels still pick CENTER ~25.7% of the time even in
    enemy-visible states, and Version41's hand-tuned tournament (13-1 vs
    econ5) uses it ~19% of the time -- both far above what a trained
    policy that's collapsed CENTER to ~0% ever showed. Now only zeroed
    when NOT already at bite range (ATTACK_DISTANCE_SQUARED); once
    already in range, CENTER gets the same max-alignment target weight a
    perfectly-aligned real direction would, since holding position (not
    overshooting past the enemy) is the actually-correct move there."""
    e1dist = states[:, 2] * 64.0
    e1bearing = states[:, 3] * math.pi  # normalized angle * pi = radians
    e1h = states[:, 4] * 100.0
    confident = (e1dist ** 2 <= RGPS_CONF_DIST_SQ) & (e1dist > 0) & (e1h > 0) & (e1h <= RGPS_CONF_HEALTH)
    if confident.sum().item() == 0:
        return torch.tensor(0.0)

    sub_states = states[confident]
    sub_bearing = e1bearing[confident]
    sub_dist = e1dist[confident]
    already_in_range = sub_dist ** 2 <= ATTACK_DISTANCE_SQUARED

    alignment = torch.cos(sub_bearing.unsqueeze(1) - DIRECTION_BEARINGS.unsqueeze(0))  # (N, 9)
    alignment[:, 0] = torch.where(already_in_range, torch.ones_like(sub_dist), torch.zeros_like(sub_dist))
    target = F.softmax(alignment * 4.0, dim=-1)  # sharpened soft target, not a hard one-hot

    log_probs = policy_log_probs(model(sub_states))
    return -(target * log_probs).sum(dim=-1).mean()


def argmax_share_loss(model, states: torch.Tensor) -> torch.Tensor:
    """KL(batch-mean prediction || uniform) -- see ARGMAX_SHARE_COEF above
    for why this is a distinct signal from per-sample entropy. Averaging
    the (already-standardized) softmax probabilities *across the batch*
    gives, for each of the 9 actions, roughly "how often does this action
    tend to win/contend for argmax" -- pushing that population-level
    distribution toward uniform directly discourages any one or two
    actions from dominating in aggregate, independent of how confident
    any single state's prediction is. Zero when the batch-mean prediction
    is exactly uniform; grows as it concentrates on fewer actions."""
    probs = policy_log_probs(model(states)).exp()
    mean_probs = probs.mean(dim=0)
    return (mean_probs * (mean_probs.clamp_min(1e-8).log() + math.log(NUM_TILES))).sum()


class ValueNet(nn.Module):
    """Training-time-only state-value baseline for advantage estimation --
    never exported to Java, since the deployed bot only ever needs the
    policy's 9 action logits, not a value estimate."""

    def __init__(self, state_dim, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

# Fixed diverse opponents, always in the roster. Deliberately not just
# self-play: the evo loop already showed pure self-play against near-
# copies drifts toward losing to genuinely different playstyles.
#
# Plain econ5 (not econ5_no_cheese) as of this change: econ5_no_cheese
# was originally substituted in because econ5's cheese-economy edge was
# assumed to inject reward noise unrelated to what this policy (movement/
# combat only) can influence. That assumption doesn't hold on TRAIN_MAPS
# specifically -- confirmed the cheese-economy differentiation isn't
# meaningfully in play on this particular map set, unlike whatever maps
# the original 0/5-vs-econ5 eval used. This also closes a real training/
# eval mismatch found the hard way: a full zero-epsilon sweep of every
# checkpoint from a 240-iteration run showed a flat ~37% win rate against
# real econ5 with no improving trend across the whole run -- unsurprising
# in hindsight, since plain econ5 was never actually in the training
# population being optimized against, only its no-cheese stand-in was.
STATIC_OPPONENTS = ["econ5", "weighted_micro", "micro_move_imitator"]

# Prioritized opponent sampling: which static opponent gets drawn is
# weighted by a rolling estimate of how often we've been *losing* to it
# lately, instead of uniform random.choice(). Motivated by hundreds of
# iterations across several unrelated fixes (mine removal, the king-
# exclusion fix, the entropy fix, doubling the network, three separate
# state-feature additions) all leaving win rate against econ5_no_cheese
# and micro_move_imitator specifically stuck in the high-20s/low-30s%
# while weighted_micro and the self-checkpoint roster stayed solid --
# consistent evidence that uniform sampling was diluting training
# pressure on exactly the two matchups that needed more of it, since each
# static opponent was drawn with the same flat 1/3 chance regardless of
# how much room there was left to improve against it.
OPPONENT_WIN_EMA_DECAY = 0.9  # ~10 recent draws' worth of memory per opponent
MIN_OPPONENT_SAMPLING_WEIGHT = 0.1  # keeps every opponent at least somewhat reachable

# The self-checkpoint pool is now also restricted this way -- but by
# *difficulty* (win rate against us) rather than by recency. A recency
# window (MAX_SELF_CHECKPOINTS, tried earlier) was reverted because it
# let a specific old checkpoint (learner_rl_ckpt10) age out and become
# permanently unreachable, and the fully-trained policy had quietly
# regressed to losing 8/10 games against it by the time that was noticed.
# Restricting instead to the TOP_N_SELF_CHECKPOINTS hardest-for-us
# checkpoints keeps the pool small (so training pressure concentrates on
# genuine weak spots instead of being spread thin across a roster that
# now numbers in the dozens) without hard-coding "recent" as a proxy for
# "relevant" -- an old checkpoint we're currently losing to stays
# eligible regardless of its age, and one we're beating soundly drops out
# regardless of how recent it is. The same residual risk from before
# still applies in a different guise, though: a checkpoint that drops out
# because we're currently beating it isn't tested again until it
# re-enters the top N, so a later regression against it could still go
# unnoticed for a while -- this trades one blind spot for a smaller,
# difficulty-shaped one rather than eliminating the risk entirely.
#
# Cut 30->12: with the roster past 50 self-checkpoints, a genuinely hard
# one (e.g. ckpt95 at 21% win rate for us) was only 1-of-30 weighted
# candidates -- diluted enough that 150+ iterations of nominal "hardest-
# first" sampling still hadn't fixed a persistent loss to a whole cluster
# of iteration-90-to-145 checkpoints (see the best-checkpoint tournament,
# stuck unable to beat iteration 50 through iteration 225). Concentrating
# the same sampling budget onto fewer candidates doesn't reopen the pure-
# self-play overfitting risk STATIC_OPPONENT_PROB was added for: that
# 50% non-self floor is untouched, so this only concentrates pressure
# within the self-play half of training, not the whole thing.
TOP_N_SELF_CHECKPOINTS = 12


def opponent_sampling_weights(opponents, win_rate_ema):
    """Returns a weight per opponent in `opponents`, higher for ones
    we've been losing to more (win_rate_ema defaults to 0.5 -- a neutral
    prior -- for any opponent with no recorded history yet). Floored at
    MIN_OPPONENT_SAMPLING_WEIGHT so an opponent we're currently beating
    soundly doesn't stop getting sampled entirely, which would make a
    future regression against it invisible until it resurfaces on its
    own (the same class of problem the self-checkpoint recency window
    caused before it was removed)."""
    return [max(1.0 - win_rate_ema.get(opp, 0.5), MIN_OPPONENT_SAMPLING_WEIGHT) for opp in opponents]


def hardest_self_checkpoints(self_checkpoints, win_rate_ema, n=TOP_N_SELF_CHECKPOINTS):
    """Returns up to n checkpoint names from self_checkpoints, ranked by
    lowest estimated win rate against us first (i.e. the ones currently
    hardest for the policy to beat). A checkpoint with no recorded
    history yet (e.g. just created by snapshot_self) defaults to a
    neutral 0.5, the same prior opponent_sampling_weights uses, so a
    brand-new checkpoint is treated as "worth checking" rather than
    presumed easy or hard until there's actual data on it."""
    ranked = sorted(self_checkpoints, key=lambda name: win_rate_ema.get(name, 0.5))
    return ranked[:n]

# The self-checkpoint roster used to grow unbounded (every --checkpoint-
# every iterations, forever), and opponent selection was a flat
# random.choice() over STATIC_OPPONENTS + self_checkpoints. By iteration
# 240 that meant 47 self-checkpoints vs. 3 static opponents -- over 90% of
# late-training opponents were some version of itself. A 200-iteration run
# on that roster made held-out trustworthy win rate *worse* (1/12) than a
# 40-iteration run on the same fixed pipeline (3/9), exactly the pure-
# self-play overfitting already diagnosed for the evo loop, just re-
# emerging here through unbounded roster growth instead of pure self-play
# from the start. Fix: a fixed probability of drawing a static opponent
# regardless of roster size.
STATIC_OPPONENT_PROB = 0.5

# Goal-hit exploiters (see train_exploiter's EXPLOITER_WIN_RATE_THRESHOLD
# early-stop) get drawn from a separate, rarely-sampled group instead of
# the normal self_checkpoints pool -- a fresh, narrow specialist that just
# found a real exploit would otherwise get ranked as one of the "hardest"
# opponents by hardest_self_checkpoints and get drawn constantly, pulling
# training toward countering that one specific trick instead of general
# play. 0.5% still guarantees it gets exercised occasionally.
GOAL_EXPLOITER_PROB = 0.005

# NOTE: this used to also cap self_checkpoints to a rolling window of the
# most recent MAX_SELF_CHECKPOINTS, on top of STATIC_OPPONENT_PROB above --
# meant as a second guard against the same crowding-out failure mode. In
# practice it went further than that and made every self-checkpoint older
# than the window *permanently unreachable* as an opponent (a fixed
# window, not a shrinking sampling weight), regardless of STATIC_OPPONENT_
# PROB already bounding how often self-play happens at all. Concretely:
# by iteration 360 the roster had rolled past learner_rl_ckpt10 (from
# essentially the IL warm-start, before any real RL training), and a
# direct match-up test found the fully-trained iter-360 policy losing to
# that iter-10 snapshot 8/10 games (both orders, 5 maps) -- a real
# regression on the actual win condition that the per-turn combat-
# efficiency metric never surfaced, plausibly because training had spent a
# long time only ever facing recent versions of itself and lost the more
# general competence the early snapshot still had. A later attempt to cap
# it again at a larger size (20), re-selected each time to stay evenly
# spaced across the full history, turned out to be mathematically unsafe:
# because rounding can make a later recomputation re-select an iteration
# that an earlier recomputation had already dropped (and whose directory
# was therefore already deleted), the roster could end up referencing a
# checkpoint package that no longer exists on disk. A follow-up doubling-
# resolution scheme fixed that safety hole but degenerated into keeping
# only a dense cluster of the most recent checkpoints, losing the
# whole-history spread it was meant to provide. Removed again: self_
# checkpoints keeps every snapshot ever taken, uniformly sampled whenever
# the self-play branch fires, so old snapshots stay reachable (rarely, as
# the roster grows) instead of aging out entirely. The unbounded-growth
# cost this trades away (compile time and disk usage climbing with total
# iterations run) is real but was judged the lesser problem.


# How much one unit of banked cheese counts for when breaking a 1-1 split
# by final cheese+health -- see iteration_result() below.
CHEESE_HEALTH_RATIO = 2.0


def iteration_result(outcomes, round_nums, team_scores) -> str:
    """Human-readable summary of which side actually came out ahead this
    iteration's collection batch (both orders on each of TRAIN_MAPS -- see
    its comment above). Doesn't touch the RL reward itself: each
    trajectory still gets its own match's real, unambiguous outcome in
    build_batch, which is the correct per-trajectory credit assignment
    regardless of how the other games went. This is purely a clearer
    per-iteration progress signal than a raw W/L count: when the two sides
    split evenly across the batch -- an exact tie is the most position-
    bias-confounded case, exactly what rl_evaluate.py's order-independence
    check exists to catch -- the tie is broken by whichever side finished
    with more (cheese * CHEESE_HEALTH_RATIO + total remaining health)
    summed across every game with a computable score (see rl_collect.py's
    final_team_stats -- this needs the opponent's own package to print
    "[stats]", true for learner_rl/self-checkpoints/learner/
    micro_move_imitator but not e.g. econ5_no_cheese/weighted_micro).
    Falls back to the round-count tiebreak when no game has a usable
    score."""
    if not outcomes:
        return "no decided matches"
    wins = sum(1 for o in outcomes if o > 0)
    losses = len(outcomes) - wins
    if wins != losses:
        return f"{wins}-{losses} -> {'learner_rl' if wins > losses else 'opponent'}"

    margin = 0.0
    any_score = False
    for score in team_scores:
        if score is None:
            continue
        (our_cheese, our_health), (opp_cheese, opp_health) = score
        our_total = our_cheese * CHEESE_HEALTH_RATIO + our_health
        opp_total = opp_cheese * CHEESE_HEALTH_RATIO + opp_health
        margin += our_total - opp_total
        any_score = True
    if any_score and margin != 0:
        side = "learner_rl" if margin > 0 else "opponent"
        return f"{wins}-{losses} split, tiebreak by cheese+health -> {side} (margin {margin:+.0f})"

    fastest = min(range(len(outcomes)), key=lambda i: round_nums[i])
    side = "learner_rl" if outcomes[fastest] > 0 else "opponent"
    return f"{wins}-{losses} split, tiebreak by rounds (no score available) -> {side} (won in round {round_nums[fastest]})"


def behavior_prob(action: int, logits_row: torch.Tensor) -> float:
    """P(action | behavior policy), where the behavior policy is epsilon-
    greedy over the *logged* (pre-update) network's argmax -- see
    RobotPlayer.java's action selection. Note this depends only on whether
    `action` happened to equal that turn's argmax, not on the logged
    exploratory flag: even a greedily-chosen action could have also arisen
    from the random branch, and a randomly-drawn action could coincide
    with the argmax by chance."""
    argmax_action = int(torch.argmax(logits_row).item())
    base = EPSILON / NUM_TILES
    return base + (1 - EPSILON) if action == argmax_action else base


# Symmetry augmentation: the right micro response to "an enemy 2 tiles
# east, an ally 1 tile north" is the same *kind* of response as to its
# mirror image reflected across either axis -- the specific compass
# directions differ, but the underlying tactical picture doesn't. Turning
# every collected step into itself plus its 3 non-identity reflections
# (flip x, flip y, flip both == 180-degree rotation) is 4x the training
# examples from the same collected matches, no extra games needed.
#
# Only *relative* geometry is transformed: enemy/ally offsets, every
# orientation field, the action taken, and the logged logits. Absolute
# self-position (state[0:2], myX/myY) is left as-is -- correctly
# mirroring it would need the map's own width/height, which isn't
# available here -- and the outcome-derived reward/advantage/value-target
# don't depend on orientation at all, so each augmented copy reuses them
# unchanged from the original step.
#
# Direction encoding matches allDirections/dirToInt in RobotPlayer.java:
# 0=CENTER,1=N,2=NE,3=E,4=SE,5=S,6=SW,7=W,8=NW. Each permutation maps an
# original direction index to its transformed one; CENTER is fixed by
# all three, so the all-zero "no enemy/ally" sentinel padding (which
# always has direction 0 alongside zeroed position/health) stays
# correctly zeroed after transform instead of picking up a spurious
# direction.
SYMMETRY_PERMS = {
    "flip_x":    [0, 1, 8, 7, 6, 5, 4, 3, 2],
    "flip_y":    [0, 5, 4, 3, 2, 1, 8, 7, 6],
    "flip_both": [0, 5, 6, 7, 8, 1, 2, 3, 4],
}

# Distance fields (e1x/e2x/e3x/allyX/ally2X/ally3X, indices 2/6/10/19/24/28)
# are invariant under any mirror/rotation -- not listed, need no transform.
#
# Bearing fields (the "*y" name is legacy from when these were dy -- see
# buildState()'s comment) transform as an angle, not a sign flip:
#   flip_x (east-west mirror):  bearing' = -bearing
#   flip_y (north-south mirror): bearing' = 180 - bearing
#   flip_both (180 rotation):    bearing' = bearing + 180
# each wrapped back into (-180, 180]. Verified against the discrete
# direction perms above (e.g. flip_x: E(90) -> -90=W, matching perm's
# E->W swap; flip_y: N(0) -> 180=S, matching perm's N->S swap).
#
# ally2/ally3 (indices 24/25/28/29 dist/bearing, 27/31 facing dir) and the
# canMove flags (32-39) were never included in the old SYMMETRY_NEGATE_X/Y/
# SYMMETRY_DIR_FIELDS lists at all -- a real, previously-unnoticed bug:
# every symmetry-augmented training sample (3 of every 4 copies fed to PPO)
# carried ally2/ally3 fields that were either left un-mirrored (wrong sign/
# angle vs. the rest of that same augmented state) or, for canMove, not
# permuted to match the mirrored action space (e.g. a flip_x sample's
# canMoveE flag still described the ORIGINAL canMoveE, not the mirrored
# scenario's true "can I move where flip_x's action space now calls east,"
# which is the original canMoveW). Fixed here alongside the angle rewrite
# since both needed the same kind of per-field transform table anyway.
# e4/e5/e6/ally4/ally5/ally6 (indices 40-63) appended when K widened 3->6
# (see learner_rl/RobotPlayer.java's buildState() comment) -- same bearing/
# dir transform rules as slots 1-3, just at their new appended indices.
SYMMETRY_ANGLE_FIELDS = [3, 7, 11, 20, 25, 29, 41, 45, 49, 53, 57, 61]        # e1..e6/ally1..ally6 bearing
SYMMETRY_DIR_FIELDS = [5, 9, 13, 14, 22, 27, 31, 43, 47, 51, 55, 59, 63]      # e1d..e6d/ownDir/allyD1..allyD6
CANMOVE_START = 32   # canMoveN..canMoveNW, indices 32-39, order matches allDirections[1:]

# For each transform, canmove_perm[k] gives the OLD canMove index (0-7, N=0)
# that should populate NEW position k -- derived from SYMMETRY_PERMS the
# same way the discrete direction fields are, just applied to array
# position instead of a stored direction-index value (see apply_symmetry).
_CANMOVE_PERM = {t: [SYMMETRY_PERMS[t][d] - 1 for d in range(1, 9)] for t in SYMMETRY_PERMS}


def _wrap180(deg: float) -> float:
    """Wraps a degree value into (-180, 180], matching Java's
    Math.atan2-derived bearing convention (south is always +180, never
    -180)."""
    wrapped = deg % 360.0
    if wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def apply_symmetry(state, action: int, logits, transform: str):
    """Returns (state, action, logits) transformed under "flip_x"/
    "flip_y"/"flip_both" -- see SYMMETRY_PERMS above. state/logits may be
    plain lists or 1-D tensors; the return type matches state's."""
    perm = SYMMETRY_PERMS[transform]
    new_state = list(state)

    for i in SYMMETRY_ANGLE_FIELDS:
        deg = new_state[i] * 180.0
        if transform == "flip_x":
            deg = -deg
        elif transform == "flip_y":
            deg = 180.0 - deg
        elif transform == "flip_both":
            deg = deg + 180.0
        new_state[i] = _wrap180(deg) / 180.0

    for i in SYMMETRY_DIR_FIELDS:
        new_state[i] = perm[int(round(new_state[i] * 8.0))] / 8.0

    canmove_perm = _CANMOVE_PERM[transform]
    old_canmove = new_state[CANMOVE_START:CANMOVE_START + 8]
    new_state[CANMOVE_START:CANMOVE_START + 8] = [old_canmove[canmove_perm[k]] for k in range(8)]

    new_action = perm[action]
    new_logits = [logits[perm[j]] for j in range(NUM_TILES)]
    return new_state, new_action, new_logits


def build_batch(trajectories, value_net):
    """Flattens trajectories into (states, actions, advantages,
    value_targets, behavior_probs) tensors, using Generalized Advantage
    Estimation (GAE, Schulman et al. 2015) instead of a plain Monte-Carlo
    return minus a single scalar baseline. Per-step immediate reward
    combines several shaped signals, all attributed to step t-1 (the step
    whose consequences they actually reflect -- mirrors econ5's
    collect_dataset.py ",C" mechanic) except the match's terminal win/loss
    outcome, added to the last step.

    Trajectories are first regrouped by match_id (multiple of our own
    robots are usually alive at once, all logged as separate trajectory
    dicts in the same flat list) so TEAM_SPIRIT blending and the
    OPPONENT_ZERO_SUM_SCALE adjustment -- both match-level, not single-
    robot, quantities -- can be computed once per match and applied to
    every one of our robots in it, before GAE runs per-robot as before.

    GAE needs a per-step value estimate V(s_t) *within* each trajectory,
    in order -- not just one scalar baseline for the whole batch -- so
    value_net is evaluated here on each trajectory's own state sequence,
    using its weights as of the start of this batch (before this batch's
    training epochs update them). value_targets (= advantage + V(s_t), the
    standard convention) become the value net's own regression target,
    replacing the raw Monte-Carlo return used previously."""
    states, value_states, actions, advantages, value_targets, beh_probs = [], [], [], [], [], []

    by_match = defaultdict(list)
    for traj in trajectories:
        if traj["outcome"] is None:
            continue  # match had no clear winner (timeout/tie) -- no reward signal
        if len(traj["steps"]) == 0:
            continue
        by_match[traj.get("match_id", id(traj))].append(traj)

    for match_id, match_trajs in by_match.items():
        # Pass 1: each robot's own raw reward, exactly as before.
        raw_rewards = []
        for traj in match_trajs:
            steps = traj["steps"]
            T = len(steps)
            rewards = [0.0] * T
            for t, (turn, state, action, exploratory, just_captured, action_value, hp_delta, logits) in enumerate(steps):
                if t > 0 and just_captured:
                    rewards[t - 1] -= CAPTURE_DAMAGE_EQUIVALENT * HP_DELTA_SCALE
                if t > 0:
                    rewards[t - 1] += action_value * ACTION_VALUE_SCALE
                if t > 0:
                    rewards[t - 1] += hp_delta * HP_DELTA_SCALE
                if t > 0:
                    # Potential-difference proximity shaping -- see
                    # PROXIMITY_SHAPING_SCALE above. `state` here is s'
                    # (the state reached by the action taken at t-1);
                    # steps[t-1][1] is s -- matches how action_value/
                    # hp_delta above are attributed to the step whose
                    # consequences they reflect.
                    rewards[t - 1] += PROXIMITY_SHAPING_SCALE * (
                        GAMMA * enemy_potential(state) - enemy_potential(steps[t - 1][1])
                    )
            rewards[T - 1] += traj["outcome"]
            raw_rewards.append(rewards)

        # Pass 2: team-average reward per round (see TEAM_SPIRIT above).
        # Each reward slot i is keyed by the round it's actually
        # attributed to (steps[i][0]), not the robot-local index i,
        # since different robots' local step arrays don't line up
        # turn-for-turn once any of them have died or skipped a round.
        round_sum, round_count = defaultdict(float), defaultdict(int)
        for traj, rewards in zip(match_trajs, raw_rewards):
            steps = traj["steps"]
            for i, r in enumerate(rewards):
                round_num = steps[i][0]
                round_sum[round_num] += r
                round_count[round_num] += 1
        team_avg_by_round = {rnd: round_sum[rnd] / round_count[rnd] for rnd in round_sum}

        # Match-level opponent per-robot hp delta by round (see
        # OPPONENT_ZERO_SUM_SCALE above and rl_collect.py's
        # opponent_hp_delta_by_round) -- identical for every one of our
        # robots in this match, so read once off the first trajectory.
        opp_hp_delta_by_round = match_trajs[0].get("opponent_hp_delta_by_round", {})

        # Both kings' health and both sides' cheese by round (see
        # VALUE_STATE_DIM/KING_HEALTH_SCALE/CHEESE_SCALE above and
        # rl_collect.py's king_and_cheese_by_round) -- also identical for
        # every one of our robots in this match.
        our_king_cheese_by_round = match_trajs[0].get("our_king_cheese_by_round", {})
        opp_king_cheese_by_round = match_trajs[0].get("opp_king_cheese_by_round", {})

        for traj, rewards in zip(match_trajs, raw_rewards):
            steps = traj["steps"]
            T = len(steps)
            for i in range(T):
                round_num = steps[i][0]
                team_avg = team_avg_by_round.get(round_num, rewards[i])
                rewards[i] = TEAM_SPIRIT * team_avg + (1 - TEAM_SPIRIT) * rewards[i]
                if round_num in opp_hp_delta_by_round:
                    # Negative when the opponent's own units lost health
                    # that round (i.e. their prospects got worse) -- the
                    # same sign convention as our own hp_delta -- so
                    # subtracting it here adds reward when they suffer,
                    # not when we do. Skipped (falls back to 0) for
                    # rounds/opponents with no usable data (never prints
                    # "[stats]" at all, or no robot seen in back-to-back
                    # rounds that round).
                    rewards[i] += -opp_hp_delta_by_round[round_num] * OPPONENT_ZERO_SUM_SCALE

            # ValueNet's augmented input (see VALUE_STATE_DIM above):
            # the policy's own state plus this round's team-average raw
            # reward, opponent hp_delta, both kings' health, and both
            # sides' cheese -- all orientation-independent scalars, so
            # unlike the first STATE_DIM entries they don't need a
            # symmetry-specific value below, just appending as-is.
            # opp_hp_delta is raw, unnormalized HP (checked directly:
            # ranged up to 152 in magnitude, std ~25.6, vs. ~0.26 mean
            # abs value for the rest of this state, which is normalized
            # to roughly [-1,1] throughout, e.g. health fields as /100)
            # -- /100 here for the same reason, so it doesn't dominate a
            # small network's early gradients purely from being ~100x
            # every other input's scale. team_avg is already reward-
            # scale (mean/std close to 0), no fix needed there. King
            # health/cheese fall back to 0.0 for a round with no reading
            # (opponent package doesn't print "[stats]" at all, or
            # doesn't have the ",K" marker yet -- see
            # king_and_cheese_by_round) -- same "missing data reads as
            # neutral" convention opp_hp_delta already uses.
            traj_value_states = [
                s[1] + [
                    team_avg_by_round.get(s[0], 0.0),
                    opp_hp_delta_by_round.get(s[0], 0.0) / 100.0,
                    our_king_cheese_by_round.get(s[0], (0.0, 0.0))[0] / KING_HEALTH_SCALE,
                    opp_king_cheese_by_round.get(s[0], (0.0, 0.0))[0] / KING_HEALTH_SCALE,
                    our_king_cheese_by_round.get(s[0], (0.0, 0.0))[1] / CHEESE_SCALE,
                    opp_king_cheese_by_round.get(s[0], (0.0, 0.0))[1] / CHEESE_SCALE,
                ]
                for s in steps
            ]
            with torch.no_grad():
                values = value_net(torch.tensor(traj_value_states, dtype=torch.float32)).squeeze(-1).tolist()

            gae = 0.0
            traj_advantages = [0.0] * T
            for t in range(T - 1, -1, -1):
                next_value = values[t + 1] if t + 1 < T else 0.0  # no bootstrap past the last recorded step; its reward already includes the terminal outcome
                delta = rewards[t] + GAMMA * next_value - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * gae
                traj_advantages[t] = gae

            for t, (turn, state, action, exploratory, just_captured, action_value, hp_delta, logits) in enumerate(steps):
                # Original step plus its 3 symmetry-reflected copies (see
                # apply_symmetry above) -- same advantage/value_target for
                # all 4, since none of them depend on orientation, only
                # beh_prob is recomputed per copy (from that copy's own
                # action/logits).
                extra_features = traj_value_states[t][STATE_DIM:]  # [team_avg, opp_hp_delta, our_king_hp, opp_king_hp, our_cheese, opp_cheese] for this round
                for variant_state, variant_action, variant_logits in (
                    (state, action, logits),
                    apply_symmetry(state, action, logits, "flip_x"),
                    apply_symmetry(state, action, logits, "flip_y"),
                    apply_symmetry(state, action, logits, "flip_both"),
                ):
                    states.append(variant_state)
                    value_states.append(variant_state + extra_features)
                    actions.append(variant_action)
                    advantages.append(traj_advantages[t])
                    value_targets.append(traj_advantages[t] + values[t])
                    beh_probs.append(behavior_prob(variant_action, torch.tensor(variant_logits)))

    if not states:
        return None
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(value_states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(advantages, dtype=torch.float32),
        torch.tensor(value_targets, dtype=torch.float32),
        torch.tensor(beh_probs, dtype=torch.float32),
    )


def compute_dapo_kl(model, reference_model, states: torch.Tensor) -> torch.Tensor:
    """KL(pi_current || pi_reference) -- penalizes the current policy for
    drifting from a frozen snapshot of itself, mirror-descent style."""
    log_probs = policy_log_probs(model(states))
    probs = log_probs.exp()
    with torch.no_grad():
        ref_log_probs = policy_log_probs(reference_model(states))
    return (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()


def update_dapo_reference(reference_model, model, decay: float = DAPO_EMA_DECAY):
    """In-place EMA update of reference_model's weights toward model's
    current weights -- see DAPO_EMA_DECAY's comment above for why this
    replaced a periodic hard refresh. Called once per iteration, after
    that iteration's ppo_update, so the KL term used *during* an
    iteration's epochs still compares against the reference as it stood
    before this iteration's update, consistent with how the old periodic
    refresh only ever changed between iterations too."""
    with torch.no_grad():
        for ref_param, param in zip(reference_model.parameters(), model.parameters()):
            ref_param.mul_(decay).add_(param, alpha=1.0 - decay)


def ppo_update(model, value_net, reference_model, policy_optimizer, value_optimizer, states, value_states, actions, advantages, value_targets, beh_probs,
                epochs=4, clip_eps=PPO_CLIP_EPS, clip_eps_high=PPO_CLIP_EPS_HIGH,
                dapo_coef=DAPO_KL_COEF, rgps_coef=RGPS_COEF,
                il_anchor_model=None, il_anchor_coef=IL_ANCHOR_COEF,
                argmax_share_coef=ARGMAX_SHARE_COEF):
    """Clipped PPO surrogate over `epochs` passes on this one collected
    batch, plus the DAPO KL-stability term, the RGPS auxiliary term, the
    IL-teacher anchor term, and an entropy bonus (see their definitions
    above) added to the same policy loss each epoch. The ratio's
    denominator is the *behavior* policy's probability (the actual
    epsilon-greedy mixture the data was sampled from), not a frozen
    snapshot of pi_theta -- folding the off-policy correction directly
    into the clip mechanism instead of tracking two separate ratios.
    `advantages`/`value_targets` come from build_batch's GAE computation
    (using value_net's weights *before* this call) rather than being
    computed here -- the value net is then fit by plain MSE regression to
    value_targets, the standard "advantage + V(s)" target.

    dapo_coef/rgps_coef/il_anchor_coef/argmax_share_coef default to the
    main-training constants but are overridable to 0 for train_exploiter:
    DAPO's whole point is resisting drift from a stable reference across a
    long, diverse training run, RGPS injects a hand-coded main-training
    heuristic, the IL anchor pulls back toward the ORIGINAL IL baseline,
    and argmax-share pushes toward using all 9 actions roughly equally --
    all four actively fight an exploiter's actual job, which is
    specializing away from the base policy (often toward a narrow,
    repeated counter-tactic) as fast as possible in a short, fixed window
    against one specific target. il_anchor_model is None by default (no
    anchor term at all, e.g. for train_exploiter's calls, which don't pass
    one) -- main() passes the frozen IL teacher loaded once at startup."""
    advantage = advantages
    adv_std = advantage.std()
    if adv_std > 1e-6:
        advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)

    policy_loss_val, ratio_mean, kl_val, rgps_val, entropy_val, il_anchor_val, argmax_share_val = None, None, None, None, None, None, None
    for _ in range(epochs):
        log_probs = policy_log_probs(model(states))
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        pi_a = action_log_probs.exp()
        # Clamped defensively: beh_probs can be as small as EPSILON/9 for a
        # non-argmax action, so an unclamped ratio can spike large even for
        # an action pi_theta doesn't strongly favor. PPO's min(surr1,surr2)
        # only bounds the *optimistic* side (positive advantage); with
        # negative advantage a large ratio is *unbounded* in the other
        # direction by design (this is intentional in vanilla PPO, whose
        # pi_old is always in the same parametric family as pi_theta and so
        # can't produce arbitrarily small denominators the way this
        # epsilon-greedy behavior policy can) -- observed causing the
        # policy's weights to blow up (abs_max > 100) within 3 iterations
        # before this clamp was added.
        ratio = (pi_a / beh_probs).clamp(max=20.0)

        surr1 = ratio * advantage
        # Clip-Higher (see PPO_CLIP_EPS_HIGH above): asymmetric bounds, not
        # the standard symmetric [1-eps, 1+eps] -- gives a low-probability
        # action's ratio more room to rise when it's actually earning it.
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps_high) * advantage
        clipped_surr = torch.min(surr1, surr2)
        # Dual clip (see DUAL_CLIP_C above): on top of the standard clip,
        # floor the negative-advantage case at DUAL_CLIP_C*advantage so a
        # large ratio can't inflict unbounded punishment the way it
        # currently can when a rarely-taken action draws one bad sample.
        dual_clipped_surr = torch.where(
            advantage < 0,
            torch.max(clipped_surr, DUAL_CLIP_C * advantage),
            clipped_surr,
        )
        kl = compute_dapo_kl(model, reference_model, states)
        rgps = rgps_loss(model, states)
        # Mean entropy of pi_theta(.|s) over the batch -- see ENTROPY_COEF
        # above for why this is here at all. Subtracted (not added) since
        # policy_loss is being minimized but entropy is being maximized.
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        if il_anchor_model is not None:
            il_anchor_kl = compute_dapo_kl(model, il_anchor_model, states)
        else:
            il_anchor_kl = torch.tensor(0.0)
        argmax_share = argmax_share_loss(model, states)
        policy_loss = (-dual_clipped_surr.mean() + dapo_coef * kl + rgps_coef * rgps
                       + il_anchor_coef * il_anchor_kl + argmax_share_coef * argmax_share
                       - ENTROPY_COEF * entropy)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        policy_loss_val = policy_loss.item()
        ratio_mean = ratio.mean().item()
        kl_val = kl.item()
        rgps_val = rgps.item()
        entropy_val = entropy.item()
        il_anchor_val = il_anchor_kl.item()
        argmax_share_val = argmax_share.item()

    value_loss_val = None
    for _ in range(epochs):
        value_pred = value_net(value_states).squeeze(-1)
        value_loss = F.mse_loss(value_pred, value_targets)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        value_loss_val = value_loss.item()

    return policy_loss_val, ratio_mean, value_loss_val, kl_val, rgps_val, entropy_val, il_anchor_val, argmax_share_val


# train.py's export_neuralnet_java only unrolls a single hidden layer
# (StudentTileNet's shape) -- RL now trains TeacherTileNet directly (see
# TEACHER_IL_MODEL_PATH/build_warm_start_model below), so self-play
# opponents/checkpoints during training need its full 3-hidden-layer
# (256/128/64) shape unrolled into Java instead. This is ONLY ever used
# during training (self-play runs under bc-overrides' unlimited bytecode,
# see HIDDEN_DIM's comment) -- the actual deployed learner_rl package gets
# overwritten with the small, single-hidden-layer distilled student (via
# the ORIGINAL export_neuralnet_java) once at the very end of main(), after
# distill_to_student_kl. A generated file this size (~52k weighted-sum
# lines) is slower for javac to compile than the old single-layer export;
# that's an expected, inherent cost of training the bigger network, not a
# bug -- see this session's "teacher does the RL, distill at the end" plan.
# Constants (weight/bias literals) budgeted per partition class -- see
# export_teacher_neuralnet_java below. Comfortably under the JVM's
# 65535-entries-per-class-file constant pool cap, leaving margin for
# whatever else (class/method metadata) shares that pool.
NEURAL_NET_PARTITION_BUDGET = 45000


def export_teacher_neuralnet_java(model: "TeacherTileNet", path, package="learner_rl", state_dim=STATE_DIM):
    """Unlike train.py's export_neuralnet_java (one big forward() method --
    fine for a single ~1500-multiply-add hidden layer), this generates ONE
    STATIC METHOD PER NEURON instead of inlining everything into forward().
    Required, not just tidier: a fully-unrolled single method for a
    network this size generates bytecode well past the JVM's 64KB-per-
    method cap -- confirmed directly, javac rejected the naive single-
    method version with "error: code too large". Each per-neuron method's
    own bytecode is bounded by its fan-in, comfortably under that cap.

    A SEPARATE, unrelated JVM limit bites once the teacher gets large
    enough (see TeacherTileNet's 900/450/225 sizing comment in train.py,
    ~567k total weights): the constant pool is capped at 65535 entries
    PER CLASS FILE, and every literal float weight/bias here consumes one
    entry regardless of which method holds it -- confirmed directly,
    javac rejected a ~567k-weight single-class version with "too many
    constants". Neurons are therefore greedily packed into however many
    NeuralNetPart<N>.java sibling classes (same package, package-private
    static methods) are needed to keep each one under
    NEURAL_NET_PARTITION_BUDGET; NeuralNet.forward() just calls across to
    whichever partition holds each neuron. Cross-class calls within the
    player's own package are ordinary instructions to the engine's
    bytecode counter (no different from a same-class call) -- moot
    anyway since this only ever runs under training's unlimited-bytecode
    override.

    (A HadamardTanh variant of this was tried and reverted -- see
    TeacherTileNet's comment in train.py -- back to plain ReLU.)"""
    named = {n: p.data for n, p in model.named_parameters()}
    # TeacherTileNet.shared is Sequential(Linear,ReLU,Linear,ReLU,Linear,ReLU)
    # -- Linear layers sit at indices 0/2/4, matching train.py's class def.
    layers = [
        (named["shared.0.weight"], named["shared.0.bias"], True),
        (named["shared.2.weight"], named["shared.2.bias"], True),
        (named["shared.4.weight"], named["shared.4.bias"], True),
        (named["q_tilescore.weight"], named["q_tilescore.bias"], False),  # no ReLU on the output layer
    ]
    fmt = lambda v: f"{v:.12e}f"

    # Flatten every neuron across every layer into one list, in forward-
    # pass order, each carrying everything needed to write its own
    # per-neuron method body.
    neurons = []  # (layer_idx, j, prev_width, weight_row, bias_value, relu)
    prev_width = state_dim
    for layer_idx, (w, b, relu) in enumerate(layers):
        out_width = w.shape[0]
        for j in range(out_width):
            neurons.append((layer_idx, j, prev_width, w[j], b[j].item(), relu))
        prev_width = out_width

    partitions, current, current_cost = [], [], 0
    for spec in neurons:
        cost = spec[2] + 1  # fan_in weights + 1 bias
        if current and current_cost + cost > NEURAL_NET_PARTITION_BUDGET:
            partitions.append(current)
            current, current_cost = [], 0
        current.append(spec)
        current_cost += cost
    if current:
        partitions.append(current)

    part_class = lambda k: f"NeuralNetPart{k}"
    neuron_partition = {}

    # Stale partitions from a PAST export at a different (larger) size
    # need clearing first -- confirmed directly, resizing the teacher
    # back down from 13 partitions to 2 left the old NeuralNetPart2..12
    # sitting in src/learner_rl/ as dead code (still gets compiled, just
    # never called), since writing only the new partitions' files doesn't
    # touch old ones the new count no longer needs.
    for stale in path.parent.glob("NeuralNetPart*.java"):
        stale.unlink()

    for k, part in enumerate(partitions):
        with open(path.parent / f"{part_class(k)}.java", "w") as f:
            f.write(f"package {package};\n\n")
            f.write(f"class {part_class(k)} {{\n")
            for (layer_idx, j, prev_w, w_row, b_val, relu) in part:
                neuron_partition[(layer_idx, j)] = k
                f.write(f"    static float n{layer_idx}_{j}(float[] x) {{\n")
                f.write(f"        float v = {fmt(b_val)};\n")
                for i in range(prev_w):
                    f.write(f"        v += {fmt(w_row[i].item())} * x[{i}];\n")
                if relu:
                    f.write("        if (v < 0) v = 0;\n")
                f.write("        return v;\n")
                f.write("    }\n\n")
            f.write("}\n")

    with open(path, "w") as f:
        f.write(f"package {package};\n\n")
        f.write("public class NeuralNet {\n")
        f.write(f"    private static final int STATE_DIM = {state_dim};\n\n")

        widths = [state_dim] + [w.shape[0] for w, _, _ in layers]
        f.write("    public float[] forward(float[] in) {\n")
        for layer_idx in range(len(layers)):
            src = "in" if layer_idx == 0 else f"h{layer_idx - 1}"
            out_width = widths[layer_idx + 1]
            f.write(f"        float[] h{layer_idx} = new float[{out_width}];\n")
            for j in range(out_width):
                part_idx = neuron_partition[(layer_idx, j)]
                f.write(f"        h{layer_idx}[{j}] = {part_class(part_idx)}.n{layer_idx}_{j}({src});\n")
        f.write(f"        return h{len(layers) - 1};\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"NeuralNet (teacher, {len(layers)}-layer, {len(partitions)} partition classes) written to {path}", flush=True)


def snapshot_self(model, iteration: int, name: str = None, zero_eps: bool = False, epsilon: float = None) -> str:
    """Copies learner_rl's non-generated Java files into a new frozen
    package named learner_rl_ckpt<iteration> (or `name`, if given -- see
    train_exploiter below, which reuses this for its own naming scheme),
    with this iteration's weights exported into its NeuralNet.java, and
    returns the new package name so it can be added to the opponent
    roster. zero_eps=True forces EPSILON=0 in the copy's RobotPlayer.java
    -- for a package meant to be *evaluated* (a fair, noise-free head-to-
    head) rather than trained against, since EPSILON=0.33 is a training-
    exploration setting that would otherwise make a third of its moves
    random. See evaluate_and_update_best below, the first real caller.
    `epsilon` (mutually exclusive with zero_eps) instead overrides
    EPSILON to any explicit value -- see train_exploiter, which needs a
    small but nonzero rate: hard-zeroing it removed not just noise from
    the win-rate signal but the only source of stochasticity in this
    bot's decision rule (argmax + epsilon-greedy, not softmax sampling),
    and a real run showed that's enough to let on-policy PPO lock into a
    losing action with no random escape and collapse within a single
    10-iteration phase (win rate 42.9% -> 0.0% by the end) -- the same
    self-reinforcing collapse mechanism documented elsewhere in this file
    for main training, just triggered here by removing EPSILON's
    incidental role as an exploration floor against it."""
    if name is None:
        name = f"learner_rl_ckpt{iteration}"
    src_dir = PROJECT_ROOT / "src" / LEARNER
    dest_dir = PROJECT_ROOT / "src" / name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    for java_file in src_dir.glob("*.java"):
        if java_file.name == "NeuralNet.java" or java_file.name.startswith("NeuralNetPart"):
            continue  # regenerated below with this checkpoint's own frozen weights (see export_teacher_neuralnet_java's partitioning)
        text = java_file.read_text()
        # Whole-identifier replace, not just the package line: some files
        # also have an explicit same-package import (e.g. "import
        # learner_rl.Globals;") left over from learner_rl's own creation,
        # which needs renaming too or it'll silently point back at the
        # live learner_rl package instead of this frozen snapshot.
        text = re.sub(r"\blearner_rl\b", name, text)
        if (zero_eps or epsilon is not None) and java_file.name == "RobotPlayer.java":
            before = text
            new_value = 0.0 if zero_eps else epsilon
            # Regex on the declaration shape, not a literal "= 0.33f;"
            # match -- a hardcoded literal silently stopped matching (no
            # exception, just a no-op) the moment learner_rl's own
            # EPSILON was ever changed to something other than 0.33, which
            # is exactly what happened when main training's EPSILON was
            # lowered to 0.2: every exploiter export since then silently
            # inherited whatever learner_rl's current EPSILON was instead
            # of EXPLOITER_EPSILON, caught only by the warning below.
            text = re.sub(r"static final float EPSILON = [^;]+;",
                           f"static final float EPSILON = {new_value}f;", text)
            if text == before:
                print(f"WARNING: zero_eps/epsilon set but EPSILON replace had no effect in {name}/RobotPlayer.java", flush=True)
        (dest_dir / java_file.name).write_text(text)

    export_teacher_neuralnet_java(model, dest_dir / "NeuralNet.java", package=name, state_dim=STATE_DIM)
    # Also save raw weights alongside the Java export: NeuralNet.java is a
    # one-way export (Java-side deployment only), so without this an
    # intermediate checkpoint's argmax/entropy/etc. can't be inspected in
    # Python after the fact -- only the live, constantly-overwritten
    # learner_rl.pth was ever loadable, meaning "how did the policy's
    # behavior change over the course of training" could only be answered
    # for whatever the final checkpoint happened to be.
    torch.save(model.state_dict(), dest_dir / "weights.pth")
    return name


# How often (in main training iterations) to run a cheap head-to-head
# tournament check against the best-so-far checkpoint (see
# evaluate_and_update_best below). Added alongside IL_ANCHOR_COEF above,
# for the same reason: a full round-robin across 29 checkpoints from a
# 599-iteration run found the FINAL checkpoint was one of the WORST in
# the whole population (iteration/win-rate correlation -0.90) -- "last"
# is not a safe proxy for "best" in this setup, so this tracks and
# persists an actual best-so-far via periodic evaluation instead of
# trusting wherever training happens to end. 25 is a compromise: cheap
# enough not to dominate wall-clock (one TRAIN_MAPS-both-orders batch,
# ~12-14 matches, similar cost to one training iteration's own
# collection) while frequent enough to catch a decline reasonably soon
# after it starts, not hundreds of iterations later.
BEST_CHECKPOINT_EVAL_EVERY = 25
BEST_MODEL_PATH = PROJECT_ROOT / "learner_rl_best.pth"
BEST_PACKAGE_NAME = "learner_rl_best"


def evaluate_and_update_best(model, total_iterations: int, best_name, best_iteration):
    """Head-to-head tournament (TRAIN_MAPS, both orders, zero-epsilon on
    both sides for a fair, noise-free comparison) between the current
    policy and the persisted best-so-far. Promotes (overwrites
    BEST_PACKAGE_NAME/BEST_MODEL_PATH with the current policy) only if it
    wins a strict majority; otherwise the existing best is left alone.
    Returns the (possibly updated) (best_name, best_iteration) to persist
    in rl_state.json. best_name is None only on the very first call ever
    (no prior best exists), in which case the current policy is promoted
    unconditionally -- there's nothing yet to compare it against."""
    wip_name = "learner_rl_eval_wip"
    snapshot_self(model, total_iterations, name=wip_name, zero_eps=True)

    if best_name is None:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        snapshot_self(model, total_iterations, name=BEST_PACKAGE_NAME, zero_eps=True)
        recompile()
        print(f"  [best-checkpoint] no prior best -- iteration {total_iterations} promoted unconditionally", flush=True)
        shutil.rmtree(PROJECT_ROOT / "src" / wip_name)
        return BEST_PACKAGE_NAME, total_iterations

    recompile()  # wip package must be compiled before run_batch can play it
    trajectories, outcomes, round_nums, team_scores = run_batch(best_name, maps=TRAIN_MAPS, games=1, subject=wip_name)
    wins = sum(1 for o in outcomes if o > 0)
    losses = len(outcomes) - wins
    promoted = wins > losses
    print(f"  [best-checkpoint] iteration {total_iterations} vs best (iteration {best_iteration}): "
          f"{wins}W/{losses}L -> {'PROMOTED' if promoted else 'kept existing best'}", flush=True)

    if promoted:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        snapshot_self(model, total_iterations, name=BEST_PACKAGE_NAME, zero_eps=True)
        recompile()

    shutil.rmtree(PROJECT_ROOT / "src" / wip_name)
    return (BEST_PACKAGE_NAME, total_iterations) if promoted else (best_name, best_iteration)


# How often (in main training iterations) to spawn a new exploiter, and
# how many PPO iterations to specialize each one for before folding it
# back into the opponent pool. History: 15->30 (the first real exploiter,
# spawned at iteration 50, showed no trend across its full 15 iterations --
# 8W/8L -> noisy dips as low as 6W/10L -> back to 8W/8L -- doubling the
# budget was meant to give a randomly-discovered divergence more chances
# to get reinforced) ->5 (cheaper per spawn, but too short to reliably
# catch a real divergence) ->15 (splitting the difference: back to the
# original budget shown above to actually trend across a run, not the
# short 30-run one that never got a fair test at 15).
EXPLOITER_EVERY = 50
EXPLOITER_TRAIN_ITERS = 10
EXPLOITER_GAMES = 2  # games per map per iteration (was 1) -- halves the
# win-rate metric's binomial noise (28 decided matches/iteration instead
# of 14), at the cost of proportionally more match time per iteration;
# iteration count dropped 15->10 to keep total match volume similar.
EXPLOITER_REPLAY_CAPACITY = 150000  # steps; ~2 iterations' worth of fresh
# exploiter data (see fresh_steps in main()'s loop for the comparable
# per-iteration scale), so each PPO update draws on more than just the
# single most recent noisy on-policy batch.
EXPLOITER_REPLAY_TRAIN_STEPS = 60000
EXPLOITER_VALUE_BURNIN_EPOCHS = 8
EXPLOITER_EPSILON = 0.1  # was zero_eps=True (hard 0), then a real run
# showed that removed the only exploration this bot's decision rule has
# (argmax + epsilon-greedy) and let it collapse to a 0% win rate within
# one 10-iteration phase -- small but nonzero keeps most of the win-rate
# signal's fairness (vs. the original 0.33) while restoring an escape
# hatch against on-policy lock-in.

# Stop early and fold in as soon as a single iteration's win rate against
# the target crosses this, rather than grinding the full iteration
# budget regardless of outcome -- matches AlphaStar's own exploiters,
# which reset/retire on hitting a high (70%) win rate against their
# target instead of running a fixed schedule. Directly motivated by a
# real run: reset-to-baseline (see train_exploiter) peaked at 10W/6L
# (62.5%) by iteration 2-3, then drifted back down to 6W/10L by
# iteration 10 with no DAPO stability term holding it near that early
# find -- stopping right at (or near) the peak keeps the good result
# instead of training past it into a worse one.
EXPLOITER_WIN_RATE_THRESHOLD = 0.7


def train_exploiter(base_value_sd, target_opponent: str, exploiter_id: int, iterations: int = EXPLOITER_TRAIN_ITERS) -> tuple[str, bool]:
    """AlphaStar-style dedicated exploiter: a short-lived fork, reset to
    the de-biased IL baseline (not wherever main training currently is),
    trained with the sole objective of beating `target_opponent`
    specifically (normally the main policy's own just-exported package,
    i.e. "learner_rl" itself) -- not the diverse opponent pool main
    training draws from.

    This targets a specific gap found this session: self-checkpoint
    matchups consistently showed win rates well above 50% (as high as
    78%) while win rate against genuinely different opponents (econ5,
    micro_move_imitator) stayed flat regardless of how many fixes were
    layered onto the *exploration* side of training (full warm-start
    de-bias, dual-clip PPO, Clip-Higher, higher EPSILON). None of those
    touch the actual cause: the self-checkpoint pool is just older/
    younger versions of the same lineage, sharing whatever blind spot the
    current policy has, so beating them doesn't require fixing it.
    AlphaStar's league training hits the identical problem with "main
    exploiter"/"league exploiter" agents whose objective is explicitly
    finding the main agent's flaws, not maximizing their own win rate
    against everyone -- this is a lightweight version of that idea:
    concentrate a short training run's entire advantage signal on
    exactly one fixed, current-generation target instead of diluting it
    across a random opponent mix, then fold the result back in as a
    permanent, targeted hard opponent.

    Reset to the IL baseline instead of warm-started from the current
    main policy, matching AlphaStar's own exploiters (which reset to the
    supervised agent on a schedule, or on hitting a high win rate against
    their target, rather than always inheriting the main agent's current
    weights). This isn't just fidelity to the paper: the first real
    exploiter run (warm-started from the current policy, back when this
    function still took base_model_sd/base_value_sd) showed literally no
    trend across 15 iterations (8W/8L -> noisy dips as low as 6W/10L ->
    back to 8W/8L) -- because it started as a *literal copy* of its own
    target, there was no asymmetry to exploit in the first place, only
    whatever EPSILON's exploration noise happened to stumble into.
    Starting from the (much weaker, but structurally different) IL
    baseline against a fully-trained current target gives it a genuine
    capability gap to search across instead of two identical policies
    trying to diverge from a standing start.

    Deliberately skips the replay buffer (each iteration trains on just
    its own fresh batch) -- this is a short, single-purpose run, not
    something that needs main training's long-memory replay mechanism.

    Unlike the policy, the value net is warm-started from the *current*
    main value net (base_value_sd), not reset. The reset argument above
    is specifically about wanting a different, structurally distinct
    policy to search from -- it doesn't apply to value estimation, and a
    cold, randomly-initialized value net actively hurts here: GAE's
    advantage estimates depend on it, so until it calibrates, every
    epoch's policy gradient is computed from noise, silently burning
    through a chunk of an already-small 30-iteration budget (no replay
    buffer means no accumulated data to calibrate faster from either)
    before any real specialization signal can get through.

    Each iteration re-exports the exploiter's *current* in-memory weights
    to a scratch package (learner_rl_exploiter_wip, reusing snapshot_self
    for the Java-file-copying/export/recompile it already does) and plays
    THAT against target_opponent via run_batch's subject= override --
    without this, run_batch defaults to LEARNER on both the "us" and
    "subject" side whenever target_opponent also happens to be LEARNER
    (the normal case, main policy vs itself), silently collecting
    trajectories that have nothing to do with this exploiter's own
    evolving policy at all."""
    model = build_warm_start_model(STATE_DIM)
    value_net = ValueNet(VALUE_STATE_DIM, hidden=HIDDEN_DIM)
    value_net.load_state_dict(base_value_sd)
    reference_model = TeacherTileNet(STATE_DIM)
    reference_model.load_state_dict(model.state_dict())
    for p in reference_model.parameters():
        p.requires_grad_(False)

    policy_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=LR)

    wip_name = "learner_rl_exploiter_wip"
    replay = None
    hit_threshold = False
    for i in range(iterations):
        # epsilon=EXPLOITER_EPSILON, not zero_eps: the default EPSILON=0.33
        # dilutes both the win-rate this loop measures against
        # EXPLOITER_WIN_RATE_THRESHOLD and the trajectories the PPO update
        # below trains on with a third uniform-random moves, but hard-
        # zeroing it (tried first) removed this bot's only source of
        # exploration and let it collapse to a 0% win rate within one
        # phase -- see EXPLOITER_EPSILON's comment above.
        snapshot_self(model, exploiter_id, name=wip_name, epsilon=EXPLOITER_EPSILON)
        recompile()
        trajectories, outcomes, round_nums, team_scores = run_batch(target_opponent, maps=TRAIN_MAPS, games=EXPLOITER_GAMES, subject=wip_name)
        wins = sum(1 for o in outcomes if o > 0)
        win_rate = wins / len(outcomes) if outcomes else 0.0
        print(f"  [exploiter {exploiter_id}] iter {i + 1}/{iterations} vs {target_opponent}: "
              f"{wins}W/{len(outcomes) - wins}L", flush=True)
        if win_rate >= EXPLOITER_WIN_RATE_THRESHOLD:
            print(f"  [exploiter {exploiter_id}] hit {win_rate:.2f} win rate >= {EXPLOITER_WIN_RATE_THRESHOLD} threshold, "
                  f"stopping early instead of training past this result", flush=True)
            hit_threshold = True
            break
        batch = build_batch(trajectories, value_net)
        if batch is None:
            continue
        if replay is None:
            # value_net was warm-started from main training's own value
            # net (base_value_sd), which is calibrated for main's state
            # distribution, not this much-weaker, structurally distinct
            # IL-baseline exploiter's. Burn in a few value-only regression
            # steps on this first batch (no policy gradient) before it's
            # ever used for a PPO update, so the *next* iteration's
            # build_batch call -- which computes GAE with value_net's
            # weights as of that call -- starts from a value function
            # that's at least seen this policy's own states once, instead
            # of compounding a stale main-training value estimate with a
            # brand-new on-policy update on top of it.
            _, value_states0, _, _, value_targets0, _ = batch
            for _ in range(EXPLOITER_VALUE_BURNIN_EPOCHS):
                value_optimizer.zero_grad()
                F.mse_loss(value_net(value_states0).squeeze(-1), value_targets0).backward()
                value_optimizer.step()
        # Accumulate into a short rolling replay buffer instead of training
        # only on this single iteration's fresh batch -- smooths the PPO
        # update over ~2 iterations' worth of data so one noisy batch
        # doesn't dominate the gradient (see EXPLOITER_REPLAY_CAPACITY).
        replay = batch if replay is None else append_to_replay(replay, batch, EXPLOITER_REPLAY_CAPACITY)
        states, value_states, actions, advantages, value_targets, beh_probs = sample_from_replay(replay, EXPLOITER_REPLAY_TRAIN_STEPS)
        # dapo_coef/rgps_coef=0: DAPO's stability pull and RGPS's hand-
        # coded heuristic are both main-training-specific and actively
        # work against an exploiter's actual job (see this function's
        # docstring). reference_model/update_dapo_reference are still
        # threaded through since ppo_update expects them, but with
        # dapo_coef=0 they no longer influence the loss at all.
        ppo_update(model, value_net, reference_model, policy_optimizer, value_optimizer,
                   states, value_states, actions, advantages, value_targets, beh_probs, epochs=4,
                   dapo_coef=0.0, rgps_coef=0.0, il_anchor_coef=0.0, argmax_share_coef=0.0)
        update_dapo_reference(reference_model, model)

    name = f"learner_rl_exploiter{exploiter_id}"
    # zero_eps=True (not EXPLOITER_EPSILON): this is a single, one-time
    # final export, not part of the iterative training loop above, so
    # the collapse risk that motivates EXPLOITER_EPSILON there doesn't
    # apply here. This package is folded permanently into the main
    # opponent roster, so it should play its sharpest, fully specialized
    # strategy every time it's drawn, not keep taking random moves.
    snapshot_self(model, exploiter_id, name=name, zero_eps=True)
    wip_dir = PROJECT_ROOT / "src" / wip_name
    if wip_dir.exists():
        shutil.rmtree(wip_dir)  # scratch package, not needed once the permanent name exists
    print(f"  [exploiter {exploiter_id}] specialized against {target_opponent}, folded in as {name}"
          f"{' (hit goal threshold)' if hit_threshold else ''}", flush=True)
    return name, hit_threshold


def recompile():
    import subprocess
    result = subprocess.run(["./gradlew", "compileJava", "--console=plain"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        sys.exit("Recompile failed after weight export -- see output above.")


def load_state():
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        state.setdefault("opponent_win_rate", {})
        state.setdefault("best_checkpoint_name", None)
        state.setdefault("best_checkpoint_iteration", None)
        state.setdefault("action_freq_ema", [1.0 / NUM_TILES] * NUM_TILES)
        state.setdefault("goal_exploiters", [])
        return state
    return {
        "total_iterations": 0, "self_checkpoints": [], "opponent_win_rate": {},
        "best_checkpoint_name": None, "best_checkpoint_iteration": None,
        "action_freq_ema": [1.0 / NUM_TILES] * NUM_TILES,
        "goal_exploiters": [],
    }


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


# qnet.pth (the IL model distilled from econ5's own tile-scoring
# heuristic) doesn't just favor CENTER -- it carries an uneven Q-value
# spread across *all* 9 actions, inherited from econ5's own scoring, not
# from anything RL introduced. Fixing only CENTER's ~75-unit excess (the
# first version of this correction) stopped the CENTER collapse (its
# argmax share: 63.2% at warm-start -> 100% after 530 RL iterations ->
# ~2% right after the fix), but within the next 240 iterations a *new*
# collapse emerged onto EAST/WEST alone (100% combined, 0% for every
# other direction including CENTER again) -- the same self-reinforcing
# on-policy dynamic simply found the next-largest inherited edge and ran
# away with it. A dual-clip PPO fix (see DUAL_CLIP_C above) addresses the
# mechanism that lets an edge snowball once training starts, but applied
# from iteration 0 it still weakened, not prevented, the same E/W
# collapse (E and W's raw mean Q were already ~8 units above the 9-action
# average at warm-start, versus -23 to -37 for the diagonals) -- a real
# inherited edge that a fairer training dynamic alone can't undo if it's
# already there before training begins. This generalizes the CENTER fix
# to all 9 actions.
#
# Same rationale as before (see this file's git history / session notes:
# econ5's own TileScore.score() has a genuine, non-uniform mean score per
# action -- CENTER highest, then W, then the rest -- and any net trained
# via MSE against it inherits some version of that, which self-reinforcing
# on-policy RL can then run away with if left uncorrected at warm-start).
#
# Superseded the old mean-recentering version of this constant (subtract
# the 9-action mean Q from each) after a 500-iteration run built on top of
# it (with IL_ANCHOR_COEF continuously pulling the live policy back toward
# build_warm_start_model()'s output every iteration) collapsed onto the
# diagonals instead of CENTER: argmax(diagonal) went 77.4% at the anchor
# itself -> 98.5% by iteration 75 -> 98.9% by iteration 140, with SW alone
# reaching 56%. Root cause: the old constant only zeroed the *mean* Q
# across actions, which the file's own prior comment already flagged as
# insufficient ("doesn't perfectly recover econ5's own per-action argmax
# distribution... CENTER undershoots to 2.7%, NW/SE overshoot to 15-20%")
# -- a flat mean doesn't imply a flat *argmax* share when per-action
# variance also differs. This version instead directly calibrates argmax
# share: starting from raw teacher.pth's output over dataset_enemy.pt,
# iteratively nudges each action's bias by its (count - target) error
# until every one of the 9 actions is argmax for an equal share of states
# (see calibrate_bias_correction.py in git history) -- actually equalized,
# not just mean-shifted. This is also, unlike the old constant, the anchor
# IL_ANCHOR_COEF's KL term pulls the live policy toward every iteration
# (build_warm_start_model() serves both roles) -- so it now anchors
# toward "no inherent directional preference", not toward whatever
# leftover skew a pure mean-shift happened to leave in place.
#
# Recalibrated for the STATE_DIM 40->64 change (nearest-3 -> nearest-6
# enemies/allies): a from-scratch measurement was required since a new IL
# teacher was trained on the wider state -- the old constant's values are
# specific to the old 40-dim teacher's output distribution, not portable.
#
# Recalibrated AGAIN after a Hadamard-tanh architecture experiment (see
# TeacherTileNet's comment in train.py) was tried and reverted back to
# plain ReLU -- the Hadamard-tanh teacher's raw outputs had collapsed to
# near-total cross-action correlation (~1.000, effectively one shared
# signal plus tiny offsets), which made this calibration itself fail to
# converge (oscillated, never settled) rather than just producing a bad
# result -- a useful tell in hindsight that the underlying network had a
# real problem, not just this constant.
#
# Recalibrated a third time for the 900/450/225 (~9.7x param count)
# teacher -- a from-scratch measurement was required again since this is
# a differently-shaped network with its own raw output distribution, not
# a portable correction. Measured directly against teacher.pth/
# dataset_enemy.pt (13157 samples), converged cleanly to within
# ~11.10-11.12% per action (same clean convergence as the pre-Hadamard
# ReLU network, unlike the collapsed Hadamard-tanh attempt above --
# confirms the bigger ReLU network didn't reintroduce that failure mode).
#
# Recalibrated a fourth time after reverting 900/450/225 back down to
# 256/128/64 -- the bigger teacher made self-play matches themselves
# ~4.2x slower (it's what every robot calls every turn DURING training,
# not just an offline learning cost), for no benefit this run actually
# needed once ARGMAX_SHARE_COEF was handling the collapse directly. Back
# to the original-sized teacher's own output distribution, so back to a
# from-scratch measurement again.
ACTION_BIAS_CORRECTION = [30.34, -11.09, -2.57, -2.76, -7.63, 8.68, 26.40, -16.51, -24.85]


def build_warm_start_model(state_dim: int = STATE_DIM) -> TeacherTileNet:
    """Loads teacher.pth (IL-trained via train.py's train_teacher(), same
    TeacherTileNet architecture RL now trains directly -- see
    TEACHER_IL_MODEL_PATH) and applies ACTION_BIAS_CORRECTION. Unlike the
    old StudentTileNet version, no partial-transfer path is needed: the
    teacher's architecture is fixed (256/128/64 hidden, see train.py), so
    IL and RL always agree on shape as long as STATE_DIM does."""
    model = TeacherTileNet(state_dim)
    il_sd = torch.load(TEACHER_IL_MODEL_PATH, weights_only=True, map_location="cpu")
    with torch.no_grad():
        model.load_state_dict(il_sd)
        model.q_tilescore.bias -= torch.tensor(ACTION_BIAS_CORRECTION, dtype=model.q_tilescore.bias.dtype)
    return model


def load_replay_buffer():
    """Returns (states, value_states, actions, advantages, value_targets,
    beh_probs) tensors. Discarded (empty) instead of loaded if the shape
    doesn't match STATE_DIM/VALUE_STATE_DIM, or if STATE_FORMAT_VERSION
    doesn't match -- shape alone isn't enough to catch every kind of
    staleness: this session's polar-coordinate state change kept STATE_DIM
    at 40 (same column count, e.g. e1x/e1y just changed MEANING from
    (dx,dy) to (distance,bearing)), which a shape-only check silently
    passed straight through -- confirmed directly, a 40000-step buffer
    from before that change loaded and got mixed into training with no
    warning at all. STATE_FORMAT_VERSION exists specifically to catch this
    class of change (bump it any time a state field's semantics change
    without its width changing); old-format buffers (from before either
    value_states or this version tag existed) fall into the same
    mismatch path and get discarded rather than crashing on a missing
    key."""
    if REPLAY_BUFFER_PATH.exists():
        data = torch.load(REPLAY_BUFFER_PATH, weights_only=True, map_location="cpu")
        if ("value_states" in data and data.get("state_format_version") == STATE_FORMAT_VERSION
                and data["states"].shape[1] == STATE_DIM
                and data["value_states"].shape[1] == VALUE_STATE_DIM):
            return (data["states"], data["value_states"], data["actions"],
                    data["advantages"], data["value_targets"], data["beh_probs"])
        print(f"Discarding replay buffer at {REPLAY_BUFFER_PATH}: shape/format-version mismatch "
              f"with current STATE_DIM ({STATE_DIM})/VALUE_STATE_DIM ({VALUE_STATE_DIM})/"
              f"STATE_FORMAT_VERSION ({STATE_FORMAT_VERSION})", flush=True)
    return (
        torch.empty(0, STATE_DIM), torch.empty(0, VALUE_STATE_DIM), torch.empty(0, dtype=torch.long),
        torch.empty(0), torch.empty(0), torch.empty(0),
    )


def save_replay_buffer(buffer):
    states, value_states, actions, advantages, value_targets, beh_probs = buffer
    torch.save({
        "states": states, "value_states": value_states, "actions": actions, "advantages": advantages,
        "value_targets": value_targets, "beh_probs": beh_probs,
        "state_format_version": STATE_FORMAT_VERSION,
    }, REPLAY_BUFFER_PATH)


def append_to_replay(buffer, new_batch, capacity: int):
    """Concatenates new_batch onto buffer and trims to the most recent
    `capacity` steps (oldest dropped first) -- a rolling window, not an
    ever-growing buffer, so advantage/value-target staleness (see
    build_batch's docstring: these are computed once at collection time,
    not refreshed as value_net keeps training) stays bounded."""
    combined = tuple(torch.cat([old, new]) for old, new in zip(buffer, new_batch))
    if len(combined[0]) > capacity:
        combined = tuple(t[-capacity:] for t in combined)
    return combined


def sample_from_replay(buffer, n: int):
    states, value_states, actions, advantages, value_targets, beh_probs = buffer
    total = len(states)
    if total <= n:
        return states, value_states, actions, advantages, value_targets, beh_probs
    idx = torch.randperm(total)[:n]
    return states[idx], value_states[idx], actions[idx], advantages[idx], value_targets[idx], beh_probs[idx]


DISTILL_STUDENT_EPOCHS = 200
DISTILL_BATCH_SIZE = 2048
DISTILL_LR = 1e-3


def distill_to_student_kl(teacher: "TeacherTileNet", states: torch.Tensor,
                           hidden_dim: int = HIDDEN_DIM, epochs: int = DISTILL_STUDENT_EPOCHS,
                           batch_size: int = DISTILL_BATCH_SIZE, lr: float = DISTILL_LR) -> StudentTileNet:
    """Final teacher->student distillation, run once after RL training
    completes (see main()'s end). Targets the STOCHASTIC POLICY --
    KL divergence between policy_log_probs' standardized softmax outputs --
    not raw MSE-on-logits like the original IL distillation (train.py's
    distill()).

    Why this has to differ from IL's distillation: the teacher's raw
    9 outputs were trained via MSE against econ5's own tile scores
    initially, but RL never uses that raw scale -- policy_log_probs
    z-score-standardizes the raw output before every part of PPO's loss
    (ratios, entropy, DAPO KL) touches it, and standardization is
    invariant to any per-sample affine rescaling of the raw output. So
    after RL, the teacher's raw output values carry no guaranteed
    consistent scale -- only the distribution they induce does. An MSE
    target against that raw scale would be fitting whatever arbitrary
    magnitude the teacher's output drifted to during RL, not the thing
    that actually determines deployed behavior (argmax under
    EPSILON-greedy, i.e. the induced softmax shape). KL/cross-entropy on
    the standardized softmax sidesteps this: it only cares about relative
    structure (rank ordering, relative confidence), which is scale-
    invariant and exactly what argmax-based deployment actually uses --
    also the standard choice in the policy-distillation literature (Rusu
    et al. 2015) for the same reason.

    `states` should be a representative sample of what the trained policy
    actually encounters -- main() passes the accumulated replay buffer
    (already includes the 4x symmetry augmentation), not the original IL
    dataset, since that reflects the RL-trained teacher's own state
    distribution rather than econ5's."""
    student = StudentTileNet(STATE_DIM, hidden=hidden_dim)
    teacher.eval()
    with torch.no_grad():
        teacher_log_probs = policy_log_probs(teacher(states))

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    n = len(states)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            student_log_probs = policy_log_probs(student(states[idx]))
            loss = F.kl_div(student_log_probs, teacher_log_probs[idx], log_target=True, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"  distill epoch {epoch + 1}/{epochs} kl={total_loss / n:.4f}", flush=True)
    return student


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default=None, help="Force a single fixed opponent every iteration, instead of sampling the diverse roster (mainly for debugging)")
    parser.add_argument("--games", type=int, default=1, help="Games per pairing per map per iteration (default 1)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of collect+update iterations (default 1)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Snapshot self into the opponent roster every N iterations (default 5)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Clipped-surrogate passes per collected batch (default 4)")
    args = parser.parse_args()

    # MODEL_PATH now holds TeacherTileNet weights (RL trains the teacher
    # directly -- see build_warm_start_model); an old StudentTileNet-shaped
    # checkpoint from before this change fails the shape check below and
    # falls back to a fresh teacher warm-start, same as any other mismatch.
    if MODEL_PATH.exists():
        saved = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
        saved_w = saved.get("shared.0.weight")
        if saved_w is not None and saved_w.shape == (256, STATE_DIM):
            model = TeacherTileNet(STATE_DIM)
            model.load_state_dict(saved)
            print(f"Loaded existing RL checkpoint from {MODEL_PATH}")
        else:
            print(f"Existing checkpoint's dims don't match current teacher shape "
                  f"(state={STATE_DIM}, hidden0=256) -- warm-starting fresh instead.")
            model = build_warm_start_model(STATE_DIM)
    else:
        model = build_warm_start_model(STATE_DIM)
        print(f"Warm-started from imitation-learned teacher weights at {TEACHER_IL_MODEL_PATH}")

    # Frozen IL-teacher anchor for ppo_update's il_anchor_coef term (see
    # IL_ANCHOR_COEF above) -- a SECOND, independent copy of the same
    # bias-corrected warm-start point `model` began from, never updated
    # after this. Deliberately the bias-corrected version (not raw
    # teacher.pth) so the anchor's implied "ideal" distribution matches
    # what the policy was actually steered toward at initialization,
    # rather than pulling it back toward a differently-biased target.
    il_teacher_anchor = build_warm_start_model(STATE_DIM)
    for p in il_teacher_anchor.parameters():
        p.requires_grad_(False)
    il_teacher_anchor.eval()

    value_net = ValueNet(VALUE_STATE_DIM, hidden=HIDDEN_DIM)
    value_saved = torch.load(VALUE_MODEL_PATH, weights_only=True, map_location="cpu") if VALUE_MODEL_PATH.exists() else None
    if value_saved is not None and value_saved["net.0.weight"].shape[1] == VALUE_STATE_DIM and value_saved["net.0.weight"].shape[0] == HIDDEN_DIM:
        value_net.load_state_dict(value_saved)
        print(f"Loaded existing value net from {VALUE_MODEL_PATH}")
    else:
        if value_saved is not None:
            print(f"Existing value net's dims (state={value_saved['net.0.weight'].shape[1]}, hidden={value_saved['net.0.weight'].shape[0]}) "
                  f"don't match current (state={VALUE_STATE_DIM}, hidden={HIDDEN_DIM}) -- initializing a fresh one instead (no IL prior exists for it anyway).")
        else:
            print("Initializing a fresh value net (no imitation-learning equivalent exists for it)")

    # reference_model always takes its shape from `model` (freshly built
    # above if there was a dim mismatch), so no separate dimension check
    # is needed here -- it's just "does a compatible snapshot exist" vs.
    # "derive one from the current policy."
    reference_model = TeacherTileNet(STATE_DIM)
    reference_saved = torch.load(REFERENCE_MODEL_PATH, weights_only=True, map_location="cpu") if REFERENCE_MODEL_PATH.exists() else None
    reference_saved_w = reference_saved.get("shared.0.weight") if reference_saved is not None else None
    if reference_saved_w is not None and reference_saved_w.shape == (256, STATE_DIM):
        reference_model.load_state_dict(reference_saved)
        print(f"Loaded existing DAPO reference snapshot from {REFERENCE_MODEL_PATH}")
    else:
        if reference_saved is not None:
            print("Existing DAPO reference's dims don't match current teacher shape "
                  "-- reinitializing it from the current policy instead.")
        reference_model.load_state_dict(model.state_dict())
        torch.save(reference_model.state_dict(), REFERENCE_MODEL_PATH)
        print("Initialized DAPO reference snapshot from the starting policy")
    for p in reference_model.parameters():
        p.requires_grad_(False)

    policy_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=LR)

    state = load_state()
    total_iterations = state["total_iterations"]
    self_checkpoints = state["self_checkpoints"]
    opponent_win_rate = state["opponent_win_rate"]
    best_checkpoint_name = state["best_checkpoint_name"]
    best_checkpoint_iteration = state["best_checkpoint_iteration"]
    action_freq_ema = state["action_freq_ema"]
    goal_exploiters = state["goal_exploiters"]
    replay_buffer = load_replay_buffer()
    print(f"Replay buffer: {len(replay_buffer[0])} steps carried over", flush=True)

    for it in range(args.iterations):
        total_iterations += 1
        if args.opponent:
            opponent = args.opponent
        elif goal_exploiters and random.random() < GOAL_EXPLOITER_PROB:
            opponent = random.choice(goal_exploiters)
        elif not self_checkpoints or random.random() < STATIC_OPPONENT_PROB:
            weights = opponent_sampling_weights(STATIC_OPPONENTS, opponent_win_rate)
            opponent = random.choices(STATIC_OPPONENTS, weights=weights)[0]
        else:
            pool = hardest_self_checkpoints(self_checkpoints, opponent_win_rate)
            # Always-guaranteed, not just periodic: TOP_N_SELF_CHECKPOINTS
            # ranks purely by EMA win rate, so the actual best-tracked
            # checkpoint (the one the final distillation depends on) could
            # silently drop out of the pool if its measured win rate isn't
            # currently among the worst -- exactly the risk that let the
            # policy go 175+ iterations without ever facing iteration 50
            # again. Appended after ranking, not folded into it, so it
            # doesn't distort hardest_self_checkpoints' own selection.
            if best_checkpoint_name is not None and best_checkpoint_name not in pool:
                pool = pool + [best_checkpoint_name]
            weights = opponent_sampling_weights(pool, opponent_win_rate)
            opponent = random.choices(pool, weights=weights)[0]
        t_match_start = time.perf_counter()
        trajectories, outcomes, round_nums, team_scores = run_batch(opponent, maps=TRAIN_MAPS, games=args.games)
        match_time = time.perf_counter() - t_match_start
        wins = sum(1 for o in outcomes if o > 0)

        # Rolling win-rate estimate per opponent, used to weight static-
        # opponent sampling above (see opponent_sampling_weights) -- kept
        # for every opponent, not just static ones, since it's cheap and
        # gives a persisted, at-a-glance history for free.
        if outcomes:
            this_rate = wins / len(outcomes)
            prev = opponent_win_rate.get(opponent, this_rate)
            opponent_win_rate[opponent] = OPPONENT_WIN_EMA_DECAY * prev + (1 - OPPONENT_WIN_EMA_DECAY) * this_rate

        print(f"[iter {total_iterations}] vs {opponent}: collected {len(trajectories)} trajectories, "
              f"{len(outcomes)} decided matches ({wins}W/{len(outcomes) - wins}L), "
              f"result: {iteration_result(outcomes, round_nums, team_scores)}", flush=True)

        # GAE needs value_net's weights as of *before* this batch's
        # training epochs -- computed here, then this fresh batch is
        # folded into the replay buffer and training draws from the
        # buffer as a whole (see REPLAY_CAPACITY/REPLAY_TRAIN_STEPS
        # above), not just from what was collected this iteration.
        t_nn_start = time.perf_counter()
        fresh_batch = build_batch(trajectories, value_net)
        if fresh_batch is None:
            print("  no usable steps this iteration, skipping update", flush=True)
            nn_time = time.perf_counter() - t_nn_start
        else:
            replay_buffer = append_to_replay(replay_buffer, fresh_batch, REPLAY_CAPACITY)
            save_replay_buffer(replay_buffer)

            # EMA'd on THIS iteration's fresh, genuinely-on-policy actions
            # (not the replay-sampled training batch below, which mixes
            # in older data) -- the most current read available on how
            # the live policy is actually behaving right now.
            fresh_actions = fresh_batch[2]
            if len(fresh_actions) > 0:
                fresh_counts = torch.bincount(fresh_actions, minlength=NUM_TILES).float()
                fresh_freq = (fresh_counts / fresh_counts.sum()).tolist()
                action_freq_ema = [ACTION_FREQ_EMA_DECAY * old + (1 - ACTION_FREQ_EMA_DECAY) * new
                                    for old, new in zip(action_freq_ema, fresh_freq)]

            states, value_states, actions, advantages, value_targets, beh_probs = sample_from_replay(replay_buffer, REPLAY_TRAIN_STEPS)
            # Count-based exploration bonus (see ACTION_FREQ_EMA_DECAY/
            # COUNT_BONUS_COEF above): added directly to the advantage of
            # whichever action was actually taken at each step, so a
            # currently-rare action looks more worth reinforcing to PPO's
            # surrogate objective -- attacks the data-starvation side of
            # the collapse dynamic, not just the loss side.
            freq_tensor = torch.tensor(action_freq_ema)
            action_bonus = (COUNT_BONUS_COEF * (1.0 / NUM_TILES - freq_tensor)).clamp(min=0.0)
            advantages = advantages + action_bonus[actions]
            policy_loss, ratio_mean, value_loss, kl, rgps, entropy, il_anchor_kl, argmax_share = ppo_update(
                model, value_net, reference_model, policy_optimizer, value_optimizer,
                states, value_states, actions, advantages, value_targets, beh_probs, epochs=args.ppo_epochs,
                il_anchor_model=il_teacher_anchor,
            )
            nn_time = time.perf_counter() - t_nn_start
            print(f"  fresh_steps={len(fresh_batch[0])} replay_buffer={len(replay_buffer[0])} train_steps={len(states)} "
                  f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                  f"mean_advantage={advantages.mean().item():.3f} mean_ratio={ratio_mean:.3f} "
                  f"dapo_kl={kl:.4f} rgps_loss={rgps:.4f} il_anchor_kl={il_anchor_kl:.4f} "
                  f"argmax_share_kl={argmax_share:.4f} entropy={entropy:.4f} "
                  f"action_freq_ema={['%.3f' % f for f in action_freq_ema]}", flush=True)

            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(value_net.state_dict(), VALUE_MODEL_PATH)
            # learner_rl's live NeuralNet.java holds the TEACHER during
            # training (self-play runs under bc-overrides' unlimited
            # bytecode -- see HIDDEN_DIM's comment) -- it's overwritten one
            # final time with the small distilled student at the very end
            # of this function, which is what actually gets deployed.
            export_teacher_neuralnet_java(model, PROJECT_ROOT / "src" / LEARNER / "NeuralNet.java", package=LEARNER, state_dim=STATE_DIM)

            # EMA toward the just-updated policy (see DAPO_EMA_DECAY above)
            # -- replaces the old periodic hard refresh, so this now runs
            # every iteration a policy update actually happened instead of
            # every --dapo-refresh-every iterations.
            update_dapo_reference(reference_model, model)
            torch.save(reference_model.state_dict(), REFERENCE_MODEL_PATH)

        if total_iterations % args.checkpoint_every == 0:
            name = snapshot_self(model, total_iterations)
            self_checkpoints.append(name)
            print(f"  snapshotted self as {name}, added to opponent roster ({len(self_checkpoints)} self-checkpoints eligible)", flush=True)

        # One recompile covers both the updated learner_rl weights and any
        # checkpoint package just created -- both must be buildable before
        # either can be used in the next iteration's collection batch. The
        # exploiter spawn below needs its target (LEARNER's own live
        # package) freshly compiled too, so this has to happen first.
        t_compile_start = time.perf_counter()
        recompile()
        compile_time = time.perf_counter() - t_compile_start
        # match_time = actual Battlecode matches (run_batch, real game
        # simulation across TRAIN_MAPS); nn_time = build_batch (GAE) +
        # ppo_update (the actual forward/backward passes and gradient
        # steps) + saving/exporting weights; compile_time = javac
        # recompiling learner_rl (and any new checkpoint) via gradlew.
        # Added directly to answer "how much of an iteration is the
        # network itself vs. playing out the games" -- not obvious from
        # the existing per-iteration prints, which had no timing at all.
        print(f"  timing: match={match_time:.1f}s nn={nn_time:.1f}s compile={compile_time:.1f}s "
              f"total={match_time + nn_time + compile_time:.1f}s", flush=True)

        if total_iterations % EXPLOITER_EVERY == 0:
            exploiter_name, exploiter_hit_goal = train_exploiter(value_net.state_dict(), LEARNER, total_iterations)
            if exploiter_hit_goal:
                # A goal-hit exploiter found a genuine exploit against a
                # fixed target, not just "trained the full budget without
                # finding much" -- but it's a narrow specialist, and
                # hardest_self_checkpoints ranks purely by how badly we're
                # currently losing to something, so mixing it into
                # self_checkpoints would make it dominate the ranking (a
                # freshly-hard opponent) and get drawn constantly, pulling
                # training toward countering one specific trick instead of
                # general play. GOAL_EXPLOITER_PROB gives it a rare,
                # deliberate check instead.
                goal_exploiters.append(exploiter_name)
            else:
                self_checkpoints.append(exploiter_name)
            recompile()  # covers the new exploiterN package train_exploiter just wrote

        if total_iterations % BEST_CHECKPOINT_EVAL_EVERY == 0:
            best_checkpoint_name, best_checkpoint_iteration = evaluate_and_update_best(
                model, total_iterations, best_checkpoint_name, best_checkpoint_iteration)

        save_state({
            "total_iterations": total_iterations,
            "self_checkpoints": self_checkpoints,
            "opponent_win_rate": opponent_win_rate,
            "best_checkpoint_name": best_checkpoint_name,
            "best_checkpoint_iteration": best_checkpoint_iteration,
            "action_freq_ema": action_freq_ema,
            "goal_exploiters": goal_exploiters,
        })

    # Final step: distill down to the small, single-hidden-layer student
    # that can actually run within the real game's bytecode budget (see
    # HIDDEN_DIM's comment -- the teacher only ever ran under training's
    # unlimited-bytecode override). Distills from the BEST-TRACKED
    # checkpoint (see evaluate_and_update_best/BEST_MODEL_PATH above), not
    # blindly from wherever the final iteration happens to land -- the
    # whole reason that tracking exists is the round-robin finding that
    # the final iteration of the previous run was one of the WORST
    # checkpoints in its own population (iteration/win-rate correlation
    # -0.90). Falls back to the final `model` only if no best was ever
    # recorded (e.g. a run shorter than BEST_CHECKPOINT_EVAL_EVERY).
    # Uses the accumulated replay buffer as the state sample -- states the
    # trained policy actually visited, not the original econ5 IL dataset.
    distill_source = model
    if best_checkpoint_name is not None and BEST_MODEL_PATH.exists():
        distill_source = TeacherTileNet(STATE_DIM)
        distill_source.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True, map_location="cpu"))
        print(f"\nDistilling from the best-tracked checkpoint (iteration {best_checkpoint_iteration}), "
              f"not the final iteration ({total_iterations}) -- see BEST_CHECKPOINT_EVAL_EVERY above.", flush=True)
    else:
        print(f"\nNo best-tracked checkpoint recorded -- distilling from the final iteration ({total_iterations}) instead.", flush=True)

    if len(replay_buffer[0]) > 0:
        print(f"Distilling to small student (KL, {DISTILL_STUDENT_EPOCHS} epochs, "
              f"{len(replay_buffer[0])} states)...", flush=True)
        student = distill_to_student_kl(distill_source, replay_buffer[0])
        torch.save(student.state_dict(), DISTILLED_STUDENT_PATH)
        export_neuralnet_java(student, PROJECT_ROOT / "src" / LEARNER / "NeuralNet.java", package=LEARNER, state_dim=STATE_DIM, hidden=HIDDEN_DIM)
        recompile()
        print(f"Final distilled student saved to {DISTILLED_STUDENT_PATH} and exported into {LEARNER} for deployment.", flush=True)
    else:
        print("\nNo replay buffer data to distill from -- skipping final KL distillation "
              f"({LEARNER}'s NeuralNet.java still holds the full teacher).", flush=True)


if __name__ == "__main__":
    main()
