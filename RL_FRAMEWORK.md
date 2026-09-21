# RL training framework — orientation for future sessions

This file exists so a Claude Code session with no memory of prior conversations can
pick up this RL pipeline, understand what it does, run it safely, and extend it to
new tasks. If you're reading this cold: skim "Architecture" and "How to run it"
first, then come back to the rest as needed.

## What this is

`learner_rl` is a Battlecode 2026 bot (`BABY_RAT` movement policy, specifically:
which of 9 tiles — CENTER + 8 directions — to move to each turn) trained by
self-play PPO on top of an imitation-learned starting point, with a distillation
step at the end to compress the trained network down to something that fits the
game's real per-turn bytecode budget. Everything lives in `rl_train.py` (the
whole pipeline) plus `train.py` (network architectures + IL pretraining +
Java-export code, imported by `rl_train.py`).

## Architecture

**Two networks, not one.**
- `TeacherTileNet` (`train.py`): 3-layer (256/128/64 hidden), the network PPO
  actually trains. Too expensive to run inside the game's real bytecode limit,
  so it only ever runs during training's self-play matches, which use an
  unlimited-bytecode engine override (see "bc-overrides" below).
- `StudentTileNet` (`train.py`): 1 hidden layer (`HIDDEN_DIM=32`), small enough
  to deploy for real. Produced by KL-distilling the teacher's policy into it
  at the end of a run (`distill_to_student_kl` in `rl_train.py`), not trained
  directly.

**Self-play with a growing, weighted opponent pool**, not pure self-play against
only the latest version of itself (pure self-play was tried and produced worse
held-out performance — see the comment above `STATIC_OPPONENT_PROB` in
`rl_train.py`). Each iteration's opponent is drawn from:
- `STATIC_OPPONENTS` (`econ5`, `weighted_micro`, `micro_move_imitator`) with
  probability `STATIC_OPPONENT_PROB` (0.5),
- otherwise the `TOP_N_SELF_CHECKPOINTS` (12) *hardest* (lowest measured win
  rate) entries from `self_checkpoints`, a growing list of past snapshots
  (`snapshot_self()`, one every `--checkpoint-every` iterations) plus folded-in
  exploiters (see below). All weighted toward whatever the model currently
  struggles against (`opponent_sampling_weights`), not drawn uniformly.
- A rare 0.5% (`GOAL_EXPLOITER_PROB`) chance of drawing from `goal_exploiters`
  instead — see "Exploiter mechanism."

**Best-checkpoint tracking, not "use whatever the last iteration produced."**
Every `BEST_CHECKPOINT_EVAL_EVERY` (25) iterations, `evaluate_and_update_best()`
plays the current policy against the current best in a fair, deterministic
(`zero_eps=True`) head-to-head and only promotes on a real win. This exists
because later iterations are *not* reliably better than earlier ones in this
kind of long self-play run — a prior run measured a **-0.90 correlation**
between iteration number and win rate (see the comment above
`distill_to_student_kl`'s caller in `main()`). The final distillation pulls
from `learner_rl_best.pth` / `BEST_MODEL_PATH`, not the literal last iteration.
**If you want "the best model right now," read `rl_state.json`'s
`best_checkpoint_name`/`best_checkpoint_iteration` — never assume the live
`learner_rl` package or the highest-numbered checkpoint is the strongest one.**

**Exploiter mechanism** (`train_exploiter()`, AlphaStar-style): every
`EXPLOITER_EVERY` (50) iterations, a short-lived fork resets to the IL baseline
(not wherever main training currently is — a literal copy of the target has no
asymmetry to exploit) and trains for up to `EXPLOITER_TRAIN_ITERS` (10)
iterations with the sole objective of beating the current `learner_rl` package,
stopping early if it hits `EXPLOITER_WIN_RATE_THRESHOLD` (0.7). Every
regularizer that stabilizes main training (DAPO, RGPS, IL-anchor, argmax-share)
is deliberately zeroed for the exploiter — those actively fight "specialize
fast against one target," which is the whole point here. If it *never* finds
anything, it still gets folded into `self_checkpoints` as a normal (if weak)
opponent; if it *does* hit the threshold, it goes into `goal_exploiters`
instead (drawn rarely, at `GOAL_EXPLOITER_PROB`) so a narrow specialist that
found a real hole doesn't dominate normal opponent sampling and pull training
toward countering one trick instead of general play.

**Collapse prevention.** On-policy PPO with an argmax-based deployed policy has
a documented failure mode in this codebase: whichever action has even a slight
edge gets sampled more, its advantage estimate stabilizes, the gradient
reinforces it further, and it snowballs toward 100% usage while other actions
starve toward 0%. This happened twice before the current defenses existed
(CENTER, then EAST/WEST). Current defenses, all active simultaneously:
- `ENTROPY_COEF` (0.05) — per-sample softmax sharpness penalty.
- `ARGMAX_SHARE_COEF` (0.1) — population-level KL-to-uniform penalty on the
  *batch-mean* prediction (`argmax_share_loss`). Different from entropy: a
  policy can have healthy per-sample entropy while still being argmax-collapsed
  in aggregate; this targets that directly.
- Count-based exploration bonus (`COUNT_BONUS_COEF=2.0`) — adds advantage
  bonus to actions under-sampled relative to a decayed EMA
  (`ACTION_FREQ_EMA_DECAY`) of how often they're actually taken.
- `EPSILON` (in `src/learner_rl/RobotPlayer.java`, currently **0.2**) —
  epsilon-greedy exploration is the *only* source of stochasticity in the
  deployed decision rule (argmax + epsilon-greedy, not softmax sampling). This
  has a documented history: 0.1 caused a ~73x sampling-rate gap between the
  current-argmax action and trailing ones and was measured to let collapse
  survive other fixes; raised to 0.33 cut that to ~20x. Currently at 0.2
  (~37x gap) as a deliberate, only-partially-validated test of whether the
  newer defenses above (which didn't exist when 0.1 failed) are now enough to
  hold a smaller floor — **watch `argmax_share_kl` and `action_freq_ema` in
  the training log after touching this; revert toward 0.33 if either starts
  climbing without bound the way it did before.**

None of these fully prevent drift, they slow/bound it. Check `action_freq_ema`
in `rl_state.json` or the log periodically — uniform is `1/9 ≈ 0.111` per
action; sustained drift of any single action toward 0 or a large multiple of
uniform is the thing to watch for.

## Key files

| file | role |
|---|---|
| `rl_train.py` | The whole RL loop: PPO, opponent sampling, exploiter, best-tracking, checkpointing, final distillation. Read this first. |
| `train.py` | `TeacherTileNet`/`StudentTileNet` definitions, IL pretraining (`train_teacher`), `export_neuralnet_java` (compact single-class Java export for deployment), `distill()` (original IL distillation, separate from RL's own KL distillation). |
| `rl_collect.py` | `run_batch()` — plays `subject` vs `opponent` across given maps (both team orders), parses `[traj]`/`[stats]` lines out of match logs into trajectories. Used by both training's own data collection and any ad hoc evaluation script. |
| `parallel_run.py` | Runs many matches in parallel without paying Gradle/JVM overhead per match. **Always removes the per-turn bytecode limit** via `bc-overrides` (see below) — the tool of choice for playing the (bytecode-hungry) teacher network directly, or for large evaluation sweeps. |
| `collect_dataset.py` | Builds the IL dataset from econ5-vs-econ5 logs (the original imitation-learning data source; state layout must mirror `rl_train.py`'s `buildState()`/`learner_rl/RobotPlayer.java`). |
| `gradle.properties` | Build config. `teamA`/`teamB`/`maps` here are what `./gradlew run` uses when you don't override on the command line. |
| `src/learner_rl/` | The live package `recompile()` overwrites every iteration (`NeuralNet.java`/`NeuralNetPart*.java` only — `RobotPlayer.java` and friends are hand-edited, not regenerated). |
| `src/learner_rl_ckptN/`, `src/learner_rl_exploiterN/` | Frozen snapshots (`snapshot_self()`), each with its own `weights.pth` alongside the Java export. These accumulate over a run into hundreds of packages — see "Known issues" below. |
| `src/learner_rl_best/` | The current best-tracked checkpoint, always exported `zero_eps=True` (deterministic). |
| `rl_state.json` | Persisted iteration count, `self_checkpoints`, `goal_exploiters`, `opponent_win_rate` (EMA per opponent, ~10-draw memory), `best_checkpoint_name`/`iteration`, `action_freq_ema`. |
| `rl_replay_buffer.pt` | Persisted rolling replay buffer (`REPLAY_CAPACITY=40000` steps ≈ 4-5 iterations), carried across restarts. |
| `learner_rl.pth` / `learner_rl_value.pth` / `learner_rl_reference.pth` | Live policy / value-net / DAPO-reference weights, loaded on resume. |
| `bc-overrides/` | A recompiled `battlecode.common.UnitType` with every unit's bytecode limit raised to ~100,000,000, placed *before* the real engine jar on the classpath (`parallel_run.py`'s `build_classpath()`). This is how the teacher network gets to run in a real match at all — there's no config flag for this, it's a shadowed class. |

## How to run it

```bash
# Resume (default — loads learner_rl.pth/rl_state.json/replay buffer if present)
/Applications/anaconda3/bin/python3 rl_train.py --iterations 500 --games 1 --checkpoint-every 5
```

Use the full anaconda path, not plain `python3` — the system Python
intermittently lacks `torch` depending on shell state.

**Fresh restart** (new architecture/state change, or deliberately discarding a
run): move aside (don't delete — these are cheap to keep and expensive to
regenerate) `rl_state.json`, `learner_rl.pth`, `learner_rl_value.pth`,
`learner_rl_reference.pth`, `rl_replay_buffer.pt`, `learner_rl_best.pth`, then
launch the same command. Leave `src/learner_rl_ckpt*`/`exploiter*` packages
alone — each one's `weights.pth` is a real, non-regeneratable snapshot someone
may still want (e.g. for distillation experiments), only the *live* state
needs resetting.

**Stopping**: always `kill -TERM <pid>`, not `-9` — the loop saves state at
iteration boundaries and SIGTERM lets it finish the current step cleanly.
Verify with `ps -p <pid>` before assuming it's down.

**Never run two things that write `rl_collect_matchups.txt` or
`matches/parallel_logs/*.log` at the same time as live training** — both
`rl_collect.py`'s ad hoc scripts and `parallel_run.py` write to shared,
hardcoded paths with no override. Stop training first, run the standalone
thing, then resume. Pure-Python work with no match-playing (distillation,
static argmax checks on already-saved `.pth` files) is safe to run alongside
live training.

**Distilling a checkpoint outside the automatic end-of-run step**: see
`distill_to_student_kl(teacher, states, epochs=...)` in `rl_train.py` — needs
a `TeacherTileNet` loaded from some checkpoint's weights and a representative
`states` tensor (a snapshot of `rl_replay_buffer.pt`, copied first if training
is live, since `torch.save` isn't atomic and a concurrent read can catch a
partial write). ~800 epochs is a good default (diminishing returns confirmed
empirically past that; 200 is the old baked-in default and measurably worse;
6400 barely improves on 800). Export via `train.py`'s `export_neuralnet_java`.

## Key constants (current values, `rl_train.py` unless noted)

| constant | value | what it controls |
|---|---|---|
| `STATE_DIM` | 64 | Input width — K=6 nearest enemies + K=6 nearest allies (polar: distance/bearing/health/facing) + self state + 8 canMove flags. |
| `NUM_TILES` | 9 | Action space: CENTER + 8 compass directions. |
| `HIDDEN_DIM` | 32 | Student (deployed) network's single hidden layer width. |
| `LR` | 3e-4 | Adam LR, both teacher and value net. |
| `PPO_CLIP_EPS` / `_HIGH` | 0.2 / 0.28 | Dual-clip PPO range (`DUAL_CLIP_C=3.0`). |
| `REPLAY_CAPACITY` / `_TRAIN_STEPS` | 40000 / 20000 | Main training's rolling replay buffer and per-update sample size. |
| `EPSILON` (`src/learner_rl/RobotPlayer.java`) | 0.2 | Main policy's exploration floor — see "Collapse prevention" above. |
| `STATIC_OPPONENT_PROB` | 0.5 | Fraction of iterations drawing a static opponent vs. a self-checkpoint. |
| `TOP_N_SELF_CHECKPOINTS` | 12 | How many hardest self-checkpoints are eligible per iteration. |
| `EXPLOITER_EVERY` / `_TRAIN_ITERS` / `_GAMES` | 50 / 10 / 2 | Exploiter phase cadence, budget, matches-per-iteration. |
| `EXPLOITER_EPSILON` | 0.1 | Exploiter's own exploration floor — zeroing this entirely (tried first) let it collapse to a 0% win rate within one phase; see the comment on `snapshot_self`'s `epsilon` param. |
| `EXPLOITER_WIN_RATE_THRESHOLD` | 0.7 | Early-stop / "found something real" bar. |
| `GOAL_EXPLOITER_PROB` | 0.005 | How often a goal-hit exploiter gets drawn as an opponent. |
| `BEST_CHECKPOINT_EVAL_EVERY` | 25 | How often the best-checkpoint head-to-head runs. |
| `RGPS_COEF` | 0.1 | Weight of the hand-coded "approach a nearby weak enemy" auxiliary loss. |
| `ARGMAX_SHARE_COEF` / `ENTROPY_COEF` / `COUNT_BONUS_COEF` | 0.1 / 0.05 / 2.0 | Collapse-prevention terms, see above. |
| `IL_ANCHOR_COEF` | 0.02 | Pull back toward the original IL teacher's policy (prevents unbounded drift). |
| `DAPO_KL_COEF` | 0.1 | Stability term vs. an EMA'd reference snapshot of the policy. |

## Known issues / operational notes

- **`src/` accumulates hundreds of packages over a run** (each checkpoint and
  exploiter gets its own directory with a full Java export, ~50k generated
  lines for the multi-partition teacher format). This has twice caused real
  build breakage: first `OutOfMemoryError`/GC thrashing in `javac` itself
  (fixed by raising `org.gradle.jvmargs` to `-Xmx6g`), then a second
  `OutOfMemoryError` inside Gradle's *native file watcher* specifically at
  ~417 packages / ~4,439 files (fixed by `org.gradle.vfs.watch=false` — nothing
  here needs live file-watching). If compiles start taking unreasonably long
  or crash again, check `gradle.properties` first, and consider whether it's
  time to prune old `src/learner_rl_ckpt*`/`exploiter*` directories that
  aren't referenced by the current `rl_state.json`'s `self_checkpoints` list.
- **`recompile()` builds the entire `src/` tree**, not just `learner_rl` — if
  *any* other package in the repo has a compile error (including someone
  else's unrelated, actively-being-edited bot), training's `recompile()` call
  crashes the whole loop with `sys.exit`. This isn't a bug in `rl_train.py`;
  it's inherent to one shared Gradle sourceSet. If training dies on a
  compile error in a file you don't recognize, check who's editing it before
  assuming it's related to RL training at all.
- **Gradle daemons can go stale and stack up**, especially after a
  `gradle.properties` JVM-args change (old daemons with the old heap setting
  become "incompatible" and a new one spins up alongside them instead of
  replacing them). If things feel sluggish, `ps aux | grep GradleDaemon` and
  clean up old ones (`./gradlew --stop`, or `kill -TERM`/`-9` if unresponsive
  — they're fully disposable).
- **`snapshot_self()`'s `zero_eps`/`epsilon` override is regex-based on the
  literal `EPSILON` declaration shape**, not hardcoded to `0.33f` — it used to
  be hardcoded and silently stopped working (no exception, just a no-op) the
  moment `learner_rl`'s own `EPSILON` was ever changed to something else. If
  you change `EPSILON` again, this should keep working, but the lesson stands:
  a string-literal match against a value that's meant to be tunable is a
  trap — prefer matching the declaration's *shape*.
- **Match durations matter for time budgeting.** A regular training iteration
  is ~55-105s depending on how matches play out; an exploiter sub-iteration
  (2x the match volume) is ~150-165s. A full iteration cycle including the
  periodic exploiter phase and best-checkpoint eval averages ~105-110s/iteration
  empirically. Use this to estimate how many iterations fit in a given wall-clock
  budget rather than guessing.

## Adapting this for a different task

The recipe here — warm-start from IL, self-play PPO against a diverse/weighted
opponent pool, periodic best-checkpoint tracking with fair deterministic
eval, an AlphaStar-style exploiter for finding blind spots, explicit
population-level collapse defenses, and a final distillation step to fit a
real resource budget — isn't specific to "which of 9 tiles to move to." To
reuse it for a different micro decision (e.g. attack-vs-retreat, a discrete
throw-target choice, trap placement) or something structurally different
entirely:

1. **State/action definition is the one truly task-specific piece.** Redefine
   `buildState()`'s feature vector and `NUM_TILES`/`STATE_DIM` for the new
   decision; keep the K-nearest-with-insertion-sort pattern if the new task
   also cares about "nearest K things of a type" — it's a cheap, bytecode-light
   way to get a fixed-width, information-dense encoding of a variable number
   of nearby units.
2. **Keep the two-network split** (a bigger "teacher" trained under the
   unlimited-bytecode override, distilled down to whatever fits the real
   budget) *if* the natural teacher architecture would exceed the deployed
   bytecode limit. If the new decision is small enough that a compact network
   trains fine directly, skip the distillation step and its added complexity
   entirely.
3. **Keep best-checkpoint tracking and the exploiter mechanism regardless of
   task** — the "later iterations aren't reliably better" finding and the
   "self-checkpoint pool shares blind spots with itself" finding are both
   properties of self-play PPO in general, not specific to movement.
4. **Re-derive the collapse-prevention coefficients empirically for the new
   action space** rather than copying `ARGMAX_SHARE_COEF=0.1`/`ENTROPY_COEF=0.05`
   wholesale — they were tuned against this specific 9-action, CENTER-biased
   problem. A binary decision (attack/retreat) or a much larger discrete space
   would likely need different values, and might not exhibit the same collapse
   dynamic at all (it's most visible with several roughly-substitutable options
   competing for "argmax," like 9 spatially adjacent tiles).
5. **`EPSILON`'s dual role is easy to miss**: it's simultaneously "how the
   deployed decision rule explores during data collection" and (via
   `snapshot_self`'s `zero_eps`) "the only way to get a fair, deterministic
   comparison between two checkpoints." Any new task inherits this coupling
   unless the decision rule is changed to sample from a distribution instead
   of argmax+epsilon-greedy — worth considering directly if you're starting
   fresh, since it's the root cause of both the exploiter-epsilon bug and the
   fixed-per-robot-ID exploration-determinism quirk documented in this
   session's history (map's engine-level random seed is fixed per map file,
   so `new Random(rc.getID())`-seeded exploration reproduces identically
   across replays of the same map/spawn-slot — a real if likely minor
   reduction in exploration diversity, never fully investigated/fixed).
