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
import random
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import StudentTileNet, export_neuralnet_java  # reuse existing arch + Java exporter
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
    "trainingmap5", "trainingmap6", "trainingmap7",
]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "learner_rl.pth"
VALUE_MODEL_PATH = PROJECT_ROOT / "learner_rl_value.pth"  # training-time-only baseline net, never exported to Java
REFERENCE_MODEL_PATH = PROJECT_ROOT / "learner_rl_reference.pth"  # frozen DAPO snapshot, refreshed periodically
IL_MODEL_PATH = PROJECT_ROOT / "qnet.pth"  # original imitation-learned student weights, used to warm-start
STATE_PATH = PROJECT_ROOT / "rl_state.json"  # persists the global iteration count + self-checkpoint roster across separate invocations
REPLAY_BUFFER_PATH = PROJECT_ROOT / "rl_replay_buffer.pt"  # persists collected steps across iterations/invocations

# 18 original + ownHealth + nearest-ally(dx,dy,health,direction) -- see
# RobotPlayer.java's buildState(). Own and enemy orientations were already
# present in the original 18 (dirToInt(rc.getDirection()) and e1d/e2d/e3d);
# the ally's facing direction was the one orientation missing. qnet.pth
# was trained with the original 18, so warm-starting now needs a partial
# weight transfer (see build_warm_start_model) rather than a plain
# load_state_dict.
STATE_DIM = 23
NUM_TILES = 9
EPSILON = 0.1  # must match learner_rl/RobotPlayer.java's EPSILON
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
PPO_CLIP_EPS = 0.2

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

# Denser per-turn shaping, all on one consistent "HP of damage" scale:
# hpDelta is already raw HP (0-100 range), and actionValue (bites, thrown-
# rat collisions, capturing an enemy -- see RobotPlayer.java's
# autoActions()) is now logged in the same raw-HP units instead of a
# separate ad hoc bonus constant per action type. HP_DELTA_SCALE maps that
# onto a comparable footing with the +-1 terminal win/loss reward (a
# full-health swing becomes +-1), and is applied uniformly to both sides
# instead of needing a different constant for each event type.
HP_DELTA_SCALE = 1.0 / 100.0

# Getting captured takes this robot fully out of the fight -- symmetric
# with actionValue's CAPTURE_DAMAGE_EQUIVALENT for capturing an enemy
# (RobotPlayer.java), just on the losing end instead of the winning one.
CAPTURE_DAMAGE_EQUIVALENT = 25.0

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
RGPS_COEF = 0.05
RGPS_CONF_DIST_SQ = 2.0   # enemy within ~1.4 tiles
RGPS_CONF_HEALTH = 30.0   # out of 100
DIRECTION_DELTAS = torch.tensor([
    [0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1],
], dtype=torch.float32)


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
    the confidence condition."""
    e1dx = states[:, 2] * 64.0
    e1dy = states[:, 3] * 64.0
    e1h = states[:, 4] * 100.0
    dist_sq = e1dx ** 2 + e1dy ** 2
    confident = (dist_sq <= RGPS_CONF_DIST_SQ) & (dist_sq > 0) & (e1h > 0) & (e1h <= RGPS_CONF_HEALTH)
    if confident.sum().item() == 0:
        return torch.tensor(0.0)

    sub_states = states[confident]
    enemy_vec = torch.stack([e1dx[confident], e1dy[confident]], dim=1)
    enemy_vec = enemy_vec / (enemy_vec.norm(dim=1, keepdim=True) + 1e-6)

    dirs_norm = DIRECTION_DELTAS.clone()
    dirs_norm[1:] = dirs_norm[1:] / dirs_norm[1:].norm(dim=1, keepdim=True)  # leave CENTER as the zero vector

    alignment = enemy_vec @ dirs_norm.T  # (N, 9) cosine similarity of each direction with "toward the enemy"
    target = F.softmax(alignment * 4.0, dim=-1)  # sharpened soft target, not a hard one-hot

    log_probs = policy_log_probs(model(sub_states))
    return -(target * log_probs).sum(dim=-1).mean()


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
# econ5_no_cheese (not plain econ5) -- eval showed learner_rl going 0/5
# trustworthy vs econ5 but 3/5 vs econ5_no_cheese, and econ5_no_cheese
# itself going 0/4 vs plain econ5 -- econ5's edge is mostly its cheese
# economy, an axis this policy (movement/combat only) can't influence at
# all, so training against full econ5 mostly injects reward noise
# unrelated to what's being learned rather than a useful hard opponent.
STATIC_OPPONENTS = ["econ5_no_cheese", "weighted_micro", "micro_move_imitator"]

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
# general competence the early snapshot still had. Removed: self_
# checkpoints now keeps every snapshot ever taken, uniformly sampled
# whenever the self-play branch fires, so old snapshots stay reachable
# (rarely, as the roster grows) instead of aging out entirely.


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


def build_batch(trajectories, value_net):
    """Flattens trajectories into (states, actions, advantages,
    value_targets, behavior_probs) tensors, using Generalized Advantage
    Estimation (GAE, Schulman et al. 2015) instead of a plain Monte-Carlo
    return minus a single scalar baseline. Per-step immediate reward
    combines several shaped signals, all attributed to step t-1 (the step
    whose consequences they actually reflect -- mirrors econ5's
    collect_dataset.py ",C" mechanic) except the match's terminal win/loss
    outcome, added to the last step.

    GAE needs a per-step value estimate V(s_t) *within* each trajectory,
    in order -- not just one scalar baseline for the whole batch -- so
    value_net is evaluated here on each trajectory's own state sequence,
    using its weights as of the start of this batch (before this batch's
    training epochs update them). value_targets (= advantage + V(s_t), the
    standard convention) become the value net's own regression target,
    replacing the raw Monte-Carlo return used previously."""
    states, actions, advantages, value_targets, beh_probs = [], [], [], [], []
    for traj in trajectories:
        if traj["outcome"] is None:
            continue  # match had no clear winner (timeout/tie) -- no reward signal
        steps = traj["steps"]
        T = len(steps)
        if T == 0:
            continue

        rewards = [0.0] * T
        for t, (turn, state, action, exploratory, just_captured, action_value, hp_delta, logits) in enumerate(steps):
            if t > 0 and just_captured:
                rewards[t - 1] -= CAPTURE_DAMAGE_EQUIVALENT * HP_DELTA_SCALE
            if t > 0:
                rewards[t - 1] += action_value * HP_DELTA_SCALE
            if t > 0:
                rewards[t - 1] += hp_delta * HP_DELTA_SCALE
        rewards[T - 1] += traj["outcome"]

        traj_states = [s[1] for s in steps]
        with torch.no_grad():
            values = value_net(torch.tensor(traj_states, dtype=torch.float32)).squeeze(-1).tolist()

        gae = 0.0
        traj_advantages = [0.0] * T
        for t in range(T - 1, -1, -1):
            next_value = values[t + 1] if t + 1 < T else 0.0  # no bootstrap past the last recorded step; its reward already includes the terminal outcome
            delta = rewards[t] + GAMMA * next_value - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            traj_advantages[t] = gae

        for t, (turn, state, action, exploratory, just_captured, action_value, hp_delta, logits) in enumerate(steps):
            logits_t = torch.tensor(logits)
            states.append(state)
            actions.append(action)
            advantages.append(traj_advantages[t])
            value_targets.append(traj_advantages[t] + values[t])
            beh_probs.append(behavior_prob(action, logits_t))

    if not states:
        return None
    return (
        torch.tensor(states, dtype=torch.float32),
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


def ppo_update(model, value_net, reference_model, policy_optimizer, value_optimizer, states, actions, advantages, value_targets, beh_probs,
                epochs=4, clip_eps=PPO_CLIP_EPS):
    """Clipped PPO surrogate over `epochs` passes on this one collected
    batch, plus the DAPO KL-stability term and the RGPS auxiliary term
    (see their definitions above) added to the same policy loss each
    epoch. The ratio's denominator is the *behavior* policy's probability
    (the actual epsilon-greedy mixture the data was sampled from), not a
    frozen snapshot of pi_theta -- folding the off-policy correction
    directly into the clip mechanism instead of tracking two separate
    ratios. `advantages`/`value_targets` come from build_batch's GAE
    computation (using value_net's weights *before* this call) rather than
    being computed here -- the value net is then fit by plain MSE
    regression to value_targets, the standard "advantage + V(s)" target."""
    advantage = advantages
    adv_std = advantage.std()
    if adv_std > 1e-6:
        advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)

    policy_loss_val, ratio_mean, kl_val, rgps_val = None, None, None, None
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
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
        kl = compute_dapo_kl(model, reference_model, states)
        rgps = rgps_loss(model, states)
        policy_loss = -torch.min(surr1, surr2).mean() + DAPO_KL_COEF * kl + RGPS_COEF * rgps

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        policy_loss_val = policy_loss.item()
        ratio_mean = ratio.mean().item()
        kl_val = kl.item()
        rgps_val = rgps.item()

    value_loss_val = None
    for _ in range(epochs):
        value_pred = value_net(states).squeeze(-1)
        value_loss = F.mse_loss(value_pred, value_targets)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        value_loss_val = value_loss.item()

    return policy_loss_val, ratio_mean, value_loss_val, kl_val, rgps_val


def snapshot_self(model, iteration: int) -> str:
    """Copies learner_rl's non-generated Java files into a new frozen
    package named learner_rl_ckpt<iteration>, with this iteration's weights
    exported into its NeuralNet.java, and returns the new package name so
    it can be added to the opponent roster."""
    name = f"learner_rl_ckpt{iteration}"
    src_dir = PROJECT_ROOT / "src" / LEARNER
    dest_dir = PROJECT_ROOT / "src" / name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    for java_file in src_dir.glob("*.java"):
        if java_file.name == "NeuralNet.java":
            continue  # regenerated below with this checkpoint's own frozen weights
        text = java_file.read_text()
        # Whole-identifier replace, not just the package line: some files
        # also have an explicit same-package import (e.g. "import
        # learner_rl.Globals;") left over from learner_rl's own creation,
        # which needs renaming too or it'll silently point back at the
        # live learner_rl package instead of this frozen snapshot.
        text = re.sub(r"\blearner_rl\b", name, text)
        (dest_dir / java_file.name).write_text(text)

    export_neuralnet_java(model, dest_dir / "NeuralNet.java", package=name, state_dim=STATE_DIM)
    return name


def recompile():
    import subprocess
    result = subprocess.run(["./gradlew", "compileJava", "--console=plain"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        sys.exit("Recompile failed after weight export -- see output above.")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"total_iterations": 0, "self_checkpoints": []}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def build_warm_start_model(state_dim: int) -> StudentTileNet:
    """Loads qnet.pth (trained with whatever state dim it was trained with
    -- originally 18) and copies its weights into a state_dim-sized net.
    If state_dim matches, this is just a plain load. If it doesn't (e.g.
    after adding new state features), the original columns of the input
    layer get their IL-trained weights and any new columns start at their
    default random init: there's no imitation-learning prior for a state
    feature that didn't exist when qnet.pth was trained, so RL has to
    learn those from scratch."""
    model = StudentTileNet(state_dim, hidden=24)
    il_sd = torch.load(IL_MODEL_PATH, weights_only=True, map_location="cpu")
    il_dim = il_sd["shared.0.weight"].shape[1]
    with torch.no_grad():
        if il_dim == state_dim:
            model.load_state_dict(il_sd)
        else:
            n = min(il_dim, state_dim)
            model.shared[0].weight[:, :n] = il_sd["shared.0.weight"][:, :n]
            model.shared[0].bias[:] = il_sd["shared.0.bias"]
            model.q_tilescore.weight[:] = il_sd["q_tilescore.weight"]
            model.q_tilescore.bias[:] = il_sd["q_tilescore.bias"]
    return model


def load_replay_buffer():
    """Returns (states, actions, advantages, value_targets, beh_probs)
    tensors. Discarded (empty) instead of loaded if its state dim doesn't
    match the current STATE_DIM, e.g. after a state-feature change --
    stale-shaped data can't be concatenated with fresh batches anyway."""
    if REPLAY_BUFFER_PATH.exists():
        data = torch.load(REPLAY_BUFFER_PATH, weights_only=True, map_location="cpu")
        if data["states"].shape[1] == STATE_DIM:
            return data["states"], data["actions"], data["advantages"], data["value_targets"], data["beh_probs"]
        print(f"Discarding replay buffer at {REPLAY_BUFFER_PATH}: its state dim ({data['states'].shape[1]}) "
              f"doesn't match current STATE_DIM ({STATE_DIM})", flush=True)
    return (
        torch.empty(0, STATE_DIM), torch.empty(0, dtype=torch.long),
        torch.empty(0), torch.empty(0), torch.empty(0),
    )


def save_replay_buffer(buffer):
    states, actions, advantages, value_targets, beh_probs = buffer
    torch.save({
        "states": states, "actions": actions, "advantages": advantages,
        "value_targets": value_targets, "beh_probs": beh_probs,
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
    states, actions, advantages, value_targets, beh_probs = buffer
    total = len(states)
    if total <= n:
        return states, actions, advantages, value_targets, beh_probs
    idx = torch.randperm(total)[:n]
    return states[idx], actions[idx], advantages[idx], value_targets[idx], beh_probs[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default=None, help="Force a single fixed opponent every iteration, instead of sampling the diverse roster (mainly for debugging)")
    parser.add_argument("--games", type=int, default=1, help="Games per pairing per map per iteration (default 1)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of collect+update iterations (default 1)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Snapshot self into the opponent roster every N iterations (default 5)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Clipped-surrogate passes per collected batch (default 4)")
    args = parser.parse_args()

    if MODEL_PATH.exists():
        saved = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
        if saved["shared.0.weight"].shape[1] == STATE_DIM:
            model = StudentTileNet(STATE_DIM, hidden=24)
            model.load_state_dict(saved)
            print(f"Loaded existing RL checkpoint from {MODEL_PATH}")
        else:
            print(f"Existing checkpoint's state dim ({saved['shared.0.weight'].shape[1]}) doesn't match "
                  f"current STATE_DIM ({STATE_DIM}) -- warm-starting fresh instead (partial IL weight transfer).")
            model = build_warm_start_model(STATE_DIM)
    else:
        model = build_warm_start_model(STATE_DIM)
        print(f"Warm-started from imitation-learned weights at {IL_MODEL_PATH} "
              f"(partial transfer since STATE_DIM={STATE_DIM} differs from qnet.pth's original dim)")

    value_net = ValueNet(STATE_DIM, hidden=24)
    value_saved = torch.load(VALUE_MODEL_PATH, weights_only=True, map_location="cpu") if VALUE_MODEL_PATH.exists() else None
    if value_saved is not None and value_saved["net.0.weight"].shape[1] == STATE_DIM:
        value_net.load_state_dict(value_saved)
        print(f"Loaded existing value net from {VALUE_MODEL_PATH}")
    else:
        if value_saved is not None:
            print(f"Existing value net's state dim ({value_saved['net.0.weight'].shape[1]}) doesn't match "
                  f"current STATE_DIM ({STATE_DIM}) -- initializing a fresh one instead (no IL prior exists for it anyway).")
        else:
            print("Initializing a fresh value net (no imitation-learning equivalent exists for it)")

    # reference_model always takes its shape from `model` (freshly built
    # above if there was a dim mismatch), so no separate dimension check
    # is needed here -- it's just "does a compatible snapshot exist" vs.
    # "derive one from the current policy."
    reference_model = StudentTileNet(STATE_DIM, hidden=24)
    reference_saved = torch.load(REFERENCE_MODEL_PATH, weights_only=True, map_location="cpu") if REFERENCE_MODEL_PATH.exists() else None
    if reference_saved is not None and reference_saved["shared.0.weight"].shape[1] == STATE_DIM:
        reference_model.load_state_dict(reference_saved)
        print(f"Loaded existing DAPO reference snapshot from {REFERENCE_MODEL_PATH}")
    else:
        if reference_saved is not None:
            print(f"Existing DAPO reference's state dim ({reference_saved['shared.0.weight'].shape[1]}) doesn't match "
                  f"current STATE_DIM ({STATE_DIM}) -- reinitializing it from the current policy instead.")
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
    replay_buffer = load_replay_buffer()
    print(f"Replay buffer: {len(replay_buffer[0])} steps carried over", flush=True)

    for it in range(args.iterations):
        total_iterations += 1
        if args.opponent:
            opponent = args.opponent
        elif not self_checkpoints or random.random() < STATIC_OPPONENT_PROB:
            opponent = random.choice(STATIC_OPPONENTS)
        else:
            opponent = random.choice(self_checkpoints)
        trajectories, outcomes, round_nums, team_scores = run_batch(opponent, maps=TRAIN_MAPS, games=args.games)
        wins = sum(1 for o in outcomes if o > 0)
        print(f"[iter {total_iterations}] vs {opponent}: collected {len(trajectories)} trajectories, "
              f"{len(outcomes)} decided matches ({wins}W/{len(outcomes) - wins}L), "
              f"result: {iteration_result(outcomes, round_nums, team_scores)}", flush=True)

        # GAE needs value_net's weights as of *before* this batch's
        # training epochs -- computed here, then this fresh batch is
        # folded into the replay buffer and training draws from the
        # buffer as a whole (see REPLAY_CAPACITY/REPLAY_TRAIN_STEPS
        # above), not just from what was collected this iteration.
        fresh_batch = build_batch(trajectories, value_net)
        if fresh_batch is None:
            print("  no usable steps this iteration, skipping update", flush=True)
        else:
            replay_buffer = append_to_replay(replay_buffer, fresh_batch, REPLAY_CAPACITY)
            save_replay_buffer(replay_buffer)

            states, actions, advantages, value_targets, beh_probs = sample_from_replay(replay_buffer, REPLAY_TRAIN_STEPS)
            policy_loss, ratio_mean, value_loss, kl, rgps = ppo_update(
                model, value_net, reference_model, policy_optimizer, value_optimizer,
                states, actions, advantages, value_targets, beh_probs, epochs=args.ppo_epochs,
            )
            print(f"  fresh_steps={len(fresh_batch[0])} replay_buffer={len(replay_buffer[0])} train_steps={len(states)} "
                  f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
                  f"mean_advantage={advantages.mean().item():.3f} mean_ratio={ratio_mean:.3f} "
                  f"dapo_kl={kl:.4f} rgps_loss={rgps:.4f}", flush=True)

            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(value_net.state_dict(), VALUE_MODEL_PATH)
            export_neuralnet_java(model, PROJECT_ROOT / "src" / LEARNER / "NeuralNet.java", package=LEARNER, state_dim=STATE_DIM)

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
        # either can be used in the next iteration's collection batch.
        recompile()
        save_state({"total_iterations": total_iterations, "self_checkpoints": self_checkpoints})


if __name__ == "__main__":
    main()
