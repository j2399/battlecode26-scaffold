package learner_rl;

import battlecode.common.*;
import java.util.Arrays;
import java.util.Random;

/**
 * RL-training variant of learner: adds epsilon-greedy exploration on top of
 * the distilled net's argmax, and logs (state, action taken, raw logits) per
 * turn so an offline Python trainer can reconstruct trajectories and run a
 * policy-gradient update. The deployed movement decision otherwise matches
 * learner exactly; only the action-selection and logging are new.
 */
public strictfp class RobotPlayer {
    static int turnCount = 0;
    static NeuralNet nn = new NeuralNet();

    // Exploration rate for epsilon-greedy: with this probability, move to a
    // uniformly random direction instead of the net's argmax. On-policy RL
    // needs *some* chance of trying a non-greedy action to get a gradient
    // signal on anything other than what the current policy already prefers.
    // Raised 0.1->0.33: at 0.1, once any action becomes the current argmax
    // it captured ~91% of all collected samples (90% "follow the policy"
    // plus its 1/9 share of the 10% uniform slice) while each of the other
    // 8 actions got only ~1.25% -- a ~73x sampling-rate gap that let a
    // self-reinforcing collapse onto a small subset of actions (CENTER,
    // then EAST/WEST) survive both a warm-start bias fix and dual-clip PPO,
    // since neither of those touches how much data a trailing action gets
    // in the first place, only how it's weighted once collected. 0.33 cuts
    // that gap to ~20x. Lowered 0.33->0.2 to test whether the RGPS fix +
    // count-based exploration bonus + argmax-share regularizer (all added
    // after 0.33 was chosen, none of which existed when 0.1 was measured
    // above) are now enough to hold the line at a smaller floor -- gap at
    // 0.2 is ~37x, between the previously-safe 20x and the previously-
    // unsafe 73x, genuinely untested territory rather than a known-safe
    // value. Watch argmax_share_kl and action_freq_ema closely after this
    // change; revert to 0.33 if either starts moving the way it did before.
    static final float EPSILON = 0.2f;
    static Random rng;

    // Tracked turn-to-turn to detect getting captured (by the enemy) for
    // shaped-reward logging -- mirrors econ5's collect_dataset.py ",C"
    // flag mechanic. (Capturing an ENEMY is tracked differently below,
    // via autoActions()'s own return value, since that's an action this
    // robot took rather than something that happened to it.)
    static boolean prevBeingCarried = false;

    // Accumulates the damage-equivalent value of whatever autoActions()
    // did during the previous full turn (across both calls within that
    // turn), read-and-reset at the top of this one -- a single unified
    // signal covering bites, captures, and thrown-rat collisions, all
    // expressed on the same "HP of damage" scale as hpDelta below so the
    // Python side can weight them consistently instead of needing a
    // separate ad hoc bonus constant per action type.
    static float actionValueLastTurn = 0f;
    static float prevHealth = -1f;  // -1 sentinel: not yet initialized (first turn)

    // Capturing an enemy takes it fully out of the fight; valued the same
    // as weighted_micro/CombatTileScore.java's ENEMY_CARRY_BASE_VALUE
    // rather than an arbitrary new number.
    static final float CAPTURE_DAMAGE_EQUIVALENT = 25.0f;

    // Ported from weighted_micro/CombatTileScore.java's
    // offensiveThrowValue(): find the nearest *other* enemy beyond the
    // minimum throwable distance and estimate collision damage from how
    // many tiles of flight remain when the throw would reach it, capped
    // by that enemy's own health. THROW_DAMAGE_DEFAULT is the fallback
    // when no such second target is identifiable (nothing to estimate
    // against with any confidence) -- an assumed average hit rather than
    // crediting 0.
    static final int MIN_THROW_DIST_SQ = 3;
    static final float THROW_DAMAGE_DEFAULT = 50.0f;

    static final Direction[] allDirections = {
        Direction.CENTER,
        Direction.NORTH,
        Direction.NORTHEAST,
        Direction.EAST,
        Direction.SOUTHEAST,
        Direction.SOUTH,
        Direction.SOUTHWEST,
        Direction.WEST,
        Direction.NORTHWEST,
    };

    static int dirToInt(Direction d) {
        if (d == Direction.CENTER) return 0;
        if (d == Direction.NORTH) return 1;
        if (d == Direction.NORTHEAST) return 2;
        if (d == Direction.EAST) return 3;
        if (d == Direction.SOUTHEAST) return 4;
        if (d == Direction.SOUTH) return 5;
        if (d == Direction.SOUTHWEST) return 6;
        if (d == Direction.WEST) return 7;
        if (d == Direction.NORTHWEST) return 8;
        return 0;
    }

    public static void run(RobotController rc) throws GameActionException {
        try { Globals.init(rc); } catch (Exception e) {
            System.out.println("Globals.init exception: " + e.getMessage());
        }
        rng = new Random(rc.getID());

        while (true) {
            turnCount++;
            try {
                if (rc.getType() == UnitType.RAT_KING) {
                    RatKing.run(rc);
                    continue;
                }

                // Every rat squeaks once a round -- a cheap, omnidirectional
                // broadcast (SQUEAK_RADIUS_SQUARED=16) heard by any teammate
                // within range, unlike vision, which for a BABY_RAT is a
                // narrower 90-degree cone facing wherever this robot happens
                // to be turned (visionConeRadiusSquared=20, visionConeAngle=
                // 90). The message content carries no information -- only
                // its source location (read in buildState() below) matters,
                // giving allies awareness of this robot's position even when
                // it's outside their vision cone (e.g. behind them).
                rc.squeak(0);

                // See RatKing.java's identical line: a minimal, always-on
                // per-turn stats line, separate from "[traj]" (which is only
                // ever collected for one side -- see the team-mislabeling
                // bug fix in rl_collect.py), so external tooling can read
                // both sides' cheese/health without needing every opponent
                // instrumented like learner_rl's own RL logging.
                System.out.println("[stats] " + rc.getID() + "," + rc.getRoundNum() + "," + rc.getGlobalCheese() + "," + rc.getHealth());

                boolean carriedNow = rc.isBeingCarried();
                boolean justCaptured = carriedNow && !prevBeingCarried;
                prevBeingCarried = carriedNow;

                float actionValue = actionValueLastTurn;
                actionValueLastTurn = 0f;

                float currentHealth = rc.getHealth();
                float hpDelta = (prevHealth < 0) ? 0f : (currentHealth - prevHealth);
                prevHealth = currentHealth;

                Team opponent = rc.getTeam().opponent();
                RobotInfo[] enemies = rc.senseNearbyRobots(-1, opponent);

                // Saved before this turn's move so the pre-move ideal-direction
                // check below and the post-move faceClosest() can both consider
                // every enemy seen this turn, not just whichever one was
                // closest before we moved -- a different pre-move enemy can
                // become the truly closest one once our own position changes.
                RobotInfo[] preMoveEnemies = enemies;

                actionValueLastTurn += autoActions(rc, enemies);

                if (enemies.length == 0) {
                    rc.setIndicatorString("RUSH");
                    if (rc.isMovementReady())
                        Navigator.moveTo(Globals.oppositeLocation);
                    enemies = rc.senseNearbyRobots(-1, opponent);
                    actionValueLastTurn += autoActions(rc, enemies);
                } else {
                    float[] state = buildState(rc, enemies);
                    float[] q = nn.forward(state);
                    int bestTile = argmax(q);

                    boolean exploratory = rng.nextFloat() < EPSILON;
                    int actionTaken = exploratory ? rng.nextInt(9) : bestTile;

                    logTrajectoryStep(rc, state, q, actionTaken, exploratory, justCaptured, actionValue, hpDelta);

                    rc.setIndicatorString("tile" + actionTaken);

                    // If the direction we're about to move in already faces the
                    // closest enemy we saw before moving, turn+move that way in
                    // one step -- we end up facing correctly as a byproduct of
                    // the move, so there's no need to re-face afterward. Only
                    // when that doesn't hold (or we can't move that way) do we
                    // fall back to moving via the rotate cascade and figuring
                    // out facing separately once the move (and this turn's
                    // actions) have actually happened.
                    boolean facedTowardMove = false;

                    if (actionTaken > 0) {
                        Direction dirIdeal = allDirections[actionTaken];
                        MapLocation myLoc = rc.getLocation();
                        MapLocation preMoveClosestLoc = closestEnemyLoc(rc, preMoveEnemies);

                        if (preMoveClosestLoc != null
                                && myLoc.directionTo(preMoveClosestLoc) == dirIdeal
                                && rc.isMovementReady() && rc.canMove(dirIdeal)) {
                            if (rc.getDirection() != dirIdeal && rc.canTurn(dirIdeal))
                                rc.turn(dirIdeal);
                            rc.move(dirIdeal);
                            facedTowardMove = true;
                        } else if (rc.isMovementReady()) {
                            if (rc.canMove(dirIdeal)) {
                                rc.move(dirIdeal);
                            } else if (rc.canMove(dirIdeal.rotateLeft())) {
                                rc.move(dirIdeal.rotateLeft());
                            } else if (rc.canMove(dirIdeal.rotateRight())) {
                                rc.move(dirIdeal.rotateRight());
                            } else if (rc.canMove(dirIdeal.rotateLeft().rotateLeft())) {
                                rc.move(dirIdeal.rotateLeft().rotateLeft());
                            } else if (rc.canMove(dirIdeal.rotateRight().rotateRight())) {
                                rc.move(dirIdeal.rotateRight().rotateRight());
                            }
                        }
                    }

                    enemies = rc.senseNearbyRobots(-1, opponent);
                    actionValueLastTurn += autoActions(rc, enemies);

                    if (!facedTowardMove) {
                        faceClosest(rc, enemies, preMoveEnemies);
                    }
                }

            } catch (GameActionException e) {
                System.out.println("GameActionException");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("Exception");
                e.printStackTrace();
            } finally {
                Clock.yield();
            }
        }
    }

    /**
     * Logs one (state, action, logits) sample for offline RL training.
     * Format: "[traj] id,turn,state0,...,state17|action,exploratory(0/1),justCaptured(0/1),justCapturedEnemy(0/1),didAttack(0/1),hpDelta|q0,...,q8"
     * -- distinct "[traj]" tag so a Python parser can grep just these lines
     * out of a match's full stdout without confusing them with anything
     * else. justCaptured/justCapturedEnemy/didAttack/hpDelta all describe
     * what became true or changed THIS turn as a result of last turn's
     * actions (see run()) -- the Python side retroactively attributes that
     * shaped reward to the *previous* logged step, mirroring econ5's
     * collect_dataset.py ",C" mechanic.
     */
    /**
     * Format: "[traj] id,turn,state...|action,exploratory(0/1),justCaptured(0/1),actionValue,hpDelta|q0,...,q8"
     * actionValue folds bites, thrown-rat collisions, and capturing an
     * enemy into one damage-equivalent number (see autoActions()) instead
     * of a separate boolean+bonus-constant per action type; justCaptured
     * (getting captured BY the enemy) stays a separate signal since it's
     * something that happened to this robot, not an action it took.
     */
    static void logTrajectoryStep(RobotController rc, float[] state, float[] q, int action, boolean exploratory,
                                   boolean justCaptured, float actionValue, float hpDelta) {
        StringBuilder sb = new StringBuilder();
        sb.append("[traj] ").append(rc.getID()).append(',').append(rc.getRoundNum());
        for (float s : state) sb.append(',').append(s);
        sb.append('|').append(action).append(',').append(exploratory ? 1 : 0)
          .append(',').append(justCaptured ? 1 : 0)
          .append(',').append(actionValue).append(',').append(hpDelta);
        sb.append('|');
        for (int i = 0; i < q.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(q[i]);
        }
        System.out.println(sb);
    }

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) best = i;
        }
        return best;
    }

    static MapLocation closestEnemyLoc(RobotController rc, RobotInfo[] enemies) {
        MapLocation myLoc = rc.getLocation();
        MapLocation best = null;
        int bestDist = Integer.MAX_VALUE;
        for (RobotInfo ri : enemies) {
            int d = myLoc.distanceSquaredTo(ri.getLocation());
            if (d < bestDist) {
                bestDist = d;
                best = ri.getLocation();
            }
        }
        return best;
    }

    /** Turns toward whichever enemy is closest right now, considering both
     * the freshly-sensed post-move enemies and every enemy location saved
     * before we moved this turn (a pre-move enemy that wasn't the closest
     * one back then can be the closest one now that our own position has
     * changed, so the whole saved set is checked, not just the single
     * previous closest). */
    static void faceClosest(RobotController rc, RobotInfo[] enemies, RobotInfo[] remembered) throws GameActionException {
        if (!rc.canTurn()) return;

        MapLocation myLoc = rc.getLocation();
        MapLocation closest = null;
        int bestDist = Integer.MAX_VALUE;

        for (RobotInfo ri : enemies) {
            int d = myLoc.distanceSquaredTo(ri.getLocation());
            if (d < bestDist) {
                bestDist = d;
                closest = ri.getLocation();
            }
        }

        if (remembered != null) {
            for (RobotInfo ri : remembered) {
                int d = myLoc.distanceSquaredTo(ri.getLocation());
                if (d < bestDist) {
                    bestDist = d;
                    closest = ri.getLocation();
                }
            }
        }

        if (closest != null) {
            Direction toClosest = myLoc.directionTo(closest);
            if (rc.getDirection() != toClosest && rc.canTurn(toClosest))
                rc.turn(toClosest);
        }
    }

    /** Returns whether an attack (bite) actually connected this call --
     * used to log a per-turn "didAttack" shaped-reward signal. Throwing
     * and carrying don't count, only rc.attack(...). */
    /** Estimated collision damage from throwing toward targetLoc/targetHealth
     * -- shared by the generic-throw and throw-at-king cases below. Ported
     * from weighted_micro/CombatTileScore.java's offensiveThrowValue(). */
    static float estimateThrowDamageTo(MapLocation from, MapLocation targetLoc, int targetHealth) {
        int distSq = from.distanceSquaredTo(targetLoc);
        int maxFlightTiles = GameConstants.TILES_FLOWN_PER_TURN * GameConstants.THROW_DURATION;
        int tilesAway = (int) Math.round(Math.sqrt(distSq));
        if (tilesAway > maxFlightTiles) return THROW_DAMAGE_DEFAULT;
        int remainingTiles = Math.max(0, maxFlightTiles - tilesAway);
        float collisionDamage = GameConstants.THROW_DAMAGE_PER_TILE * remainingTiles;
        return Math.min(collisionDamage, targetHealth);
    }

    /** Finds the nearest enemy (other than the one being thrown) beyond
     * MIN_THROW_DIST_SQ and estimates the throw's collision damage against
     * it; THROW_DAMAGE_DEFAULT if no such second target is identifiable. */
    static float estimateOffensiveThrowDamage(RobotController rc, RobotInfo carried, RobotInfo[] enemies) {
        MapLocation from = rc.getLocation();
        int bestDistSq = Integer.MAX_VALUE;
        RobotInfo secondTarget = null;
        for (RobotInfo enemy : enemies) {
            if (enemy.getID() == carried.getID()) continue;
            int distSq = from.distanceSquaredTo(enemy.getLocation());
            if (distSq < MIN_THROW_DIST_SQ) continue;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                secondTarget = enemy;
            }
        }
        if (secondTarget == null) return THROW_DAMAGE_DEFAULT;
        return estimateThrowDamageTo(from, secondTarget.getLocation(), (int) secondTarget.getHealth());
    }

    /** Returns the damage-equivalent value of whatever action fired this
     * call (bite=GameConstants.RAT_BITE_DAMAGE, thrown-rat collision=an
     * estimate or THROW_DAMAGE_DEFAULT, capturing an enemy=
     * CAPTURE_DAMAGE_EQUIVALENT, nothing=0) -- used to log a per-turn
     * shaped-reward signal that folds all three action types onto the
     * same "HP of damage" scale instead of separate ad hoc bonuses. */
    static float autoActions(RobotController rc, RobotInfo[] enemies) throws GameActionException {
        if (!rc.isActionReady()) return 0f;

        if (rc.getCarrying() != null && rc.canThrowRat() && enemies.length > 0 && rc.canTurn()) {
            RobotInfo carried = rc.getCarrying();
            RobotInfo closest = null;
            int closestDist = Integer.MAX_VALUE;
            MapLocation myLoc = rc.getLocation();
            for (RobotInfo ri : enemies) {
                int d = myLoc.distanceSquaredTo(ri.getLocation());
                if (d < closestDist) {
                    closestDist = d;
                    closest = ri;
                }
            }
            if (closest != null) {
                rc.turn(myLoc.directionTo(closest.getLocation()));
            }
            if (rc.canThrowRat()) {
                float estimate = estimateOffensiveThrowDamage(rc, carried, enemies);
                rc.throwRat();
                return estimate;
            }
        }

        if (!rc.isActionReady()) return 0f;

        for (RobotInfo ri : enemies) {
            if (ri.getType() == UnitType.RAT_KING) {
                MapLocation kingLoc = ri.getLocation();
                Direction toKing = rc.getLocation().directionTo(kingLoc);
                if (rc.getDirection() != toKing && rc.canTurn(toKing))
                    rc.turn(toKing);
                if (rc.canThrowRat() && rc.getDirection() == toKing) {
                    float estimate = estimateThrowDamageTo(rc.getLocation(), kingLoc, (int) ri.getHealth());
                    rc.throwRat();
                    return estimate;
                }
                MapLocation atkLoc = kingLoc.subtract(toKing);
                if (rc.canAttack(atkLoc)) {
                    rc.attack(atkLoc);
                    return GameConstants.RAT_BITE_DAMAGE;
                }
                break;
            }
        }

        if (!rc.isActionReady()) return 0f;
        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            if (ri.getType() == UnitType.BABY_RAT && rc.canCarryRat(el)) {
                rc.carryRat(el);
                return CAPTURE_DAMAGE_EQUIVALENT;
            }
        }
        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            if (rc.canAttack(el)) {
                rc.attack(el);
                return GameConstants.RAT_BITE_DAMAGE;
            }
        }
        return 0f;
    }

    // Inserts a new (distSq, dist, bearing, health, dir) candidate into the
    // K parallel arrays (already sorted ascending by distSq), shifting
    // anything farther-away right and dropping off the end -- a simple
    // insertion sort specialized for "find the K nearest of a stream,"
    // reused for both enemies and allies below instead of hand-writing a
    // separate cascade of if/else-if per K (only tractable for the old
    // K=3; doesn't scale to K=6 legibly).
    static void insertNearest(int[] distSq, float[] dist, float[] bear, float[] health, int[] dir,
                               int newDistSq, float newDist, float newBear, float newHealth, int newDir) {
        int n = distSq.length;
        if (newDistSq >= distSq[n - 1]) return;
        int i = n - 1;
        while (i > 0 && distSq[i - 1] > newDistSq) {
            distSq[i] = distSq[i - 1]; dist[i] = dist[i - 1]; bear[i] = bear[i - 1];
            health[i] = health[i - 1]; dir[i] = dir[i - 1];
            i--;
        }
        distSq[i] = newDistSq; dist[i] = newDist; bear[i] = newBear; health[i] = newHealth; dir[i] = newDir;
    }

    static final int K_NEAREST = 6;

    static float[] buildState(RobotController rc, RobotInfo[] enemies) throws GameActionException {
        MapLocation loc = rc.getLocation();
        int myX = loc.x, myY = loc.y;

        // e_x/e_y now hold (distance, bearing-in-degrees-from-north) instead
        // of raw (dx, dy) offsets -- see the state-layout comment on
        // buildState()'s return statement below for the normalization and
        // rl_train.py's SYMMETRY_ANGLE_FIELDS for how this transforms under
        // the 4x symmetry augmentation. Bearing convention: atan2(dx, dy)
        // (dx=east offset as the "y" arg, dy=north offset as the "x" arg)
        // gives 0=north, 90=east, 180/-180=south, -90=west, increasing
        // clockwise -- matches allDirections/dirToInt's ordering exactly.
        int[] eDistSq = new int[K_NEAREST];
        float[] eDist = new float[K_NEAREST], eBear = new float[K_NEAREST], eHealth = new float[K_NEAREST];
        int[] eDir = new int[K_NEAREST];
        Arrays.fill(eDistSq, Integer.MAX_VALUE);

        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            int d = loc.distanceSquaredTo(el);
            float h = ri.getHealth();
            int dir = dirToInt(ri.getDirection());
            int dx = el.x - myX, dy = el.y - myY;
            float dist = (float) Math.sqrt((double) dx * dx + (double) dy * dy);
            float bearing = (float) Math.toDegrees(Math.atan2(dx, dy));
            insertNearest(eDistSq, eDist, eBear, eHealth, eDir, d, dist, bearing, h, dir);
        }

        // Nearest K allies (excluding self), by distance -- mirrors the
        // enemy-tracking loop above. Own-health awareness and knowing more
        // than just the single nearest ally both matter for judging
        // whether a fight (or getting swarmed) is actually winnable --
        // weighted_micro's hand-written heuristic already uses both.
        RobotInfo[] allies = rc.senseNearbyRobots(-1, rc.getTeam());
        int[] aDistSq = new int[K_NEAREST];
        float[] aDist = new float[K_NEAREST], aBear = new float[K_NEAREST], aHealth = new float[K_NEAREST];
        int[] aDir = new int[K_NEAREST];
        Arrays.fill(aDistSq, Integer.MAX_VALUE);
        for (RobotInfo ri : allies) {
            if (ri.getID() == rc.getID()) continue;
            MapLocation al = ri.getLocation();
            int d = loc.distanceSquaredTo(al);
            float h = ri.getHealth();
            int dir = dirToInt(ri.getDirection());
            int dx = al.x - myX, dy = al.y - myY;
            float dist = (float) Math.sqrt((double) dx * dx + (double) dy * dy);
            float bearing = (float) Math.toDegrees(Math.atan2(dx, dy));
            insertNearest(aDistSq, aDist, aBear, aHealth, aDir, d, dist, bearing, h, dir);
        }

        // Also fold in allies heard (not necessarily seen) via last round's
        // squeak() -- see the squeak() call in run(). A heard-only ally's
        // health/direction aren't knowable from a squeak alone (Message
        // only carries a source location), so those default to 0 for that
        // candidate exactly like "no ally at all" already does; only its
        // position competes with the seen-allies loop above for the K
        // nearest slots.
        Message[] heardSqueaks = rc.readSqueaks(rc.getRoundNum() - 1);
        for (Message msg : heardSqueaks) {
            if (msg.getSenderID() == rc.getID()) continue;
            MapLocation al = msg.getSource();
            int d = loc.distanceSquaredTo(al);
            int dx = al.x - myX, dy = al.y - myY;
            float dist = (float) Math.sqrt((double) dx * dx + (double) dy * dy);
            float bearing = (float) Math.toDegrees(Math.atan2(dx, dy));
            insertNearest(aDistSq, aDist, aBear, aHealth, aDir, d, dist, bearing, 0, 0);
        }

        // Local numbers/health advantage: sum of health across every
        // currently-sensed ally (including self) minus every currently-
        // sensed enemy, both already vision-limited by senseNearbyRobots so
        // "local" falls out of that for free. Previously the state only
        // ever exposed the *nearest* ally/enemies individually -- with no
        // aggregate signal, the policy had no way to tell "we outnumber
        // them here" from "it's just me against three of them" even though
        // both cases can show an identical nearest-enemy reading. Scaled by
        // the same /100 as other health fields, so a value of 1.0 reads as
        // "one baby rat's worth of health ahead."
        //
        // RAT_KING excluded on both sides: at 600 health vs. a baby rat's
        // 100, including it swamped this into a near-guaranteed huge
        // negative reading (measured: -203.9 average vs -22.5 without a
        // king nearby) exactly when a rat is near the enemy's king --
        // i.e. exactly when it should be pressing the attack on the actual
        // win condition, not reading "badly outmatched, retreat." The king
        // isn't a comparable combat threat to a baby rat; it's a
        // stationary win-condition target, so its HP doesn't belong in a
        // "should I fight here" combat-balance signal.
        float localHealthSum = rc.getHealth();
        for (RobotInfo ri : allies) {
            if (ri.getID() == rc.getID()) continue;
            if (ri.getType() != UnitType.BABY_RAT) continue;
            localHealthSum += ri.getHealth();
        }
        for (RobotInfo ri : enemies) {
            if (ri.getType() != UnitType.BABY_RAT) continue;
            localHealthSum -= ri.getHealth();
        }

        // Which of the 8 adjacent tiles are actually movable right now.
        // Reuses rc.canMove() (already called every turn in the movement
        // fallback cascade, so known to be bytecode-affordable) rather
        // than rc.senseMapInfo() (constructs a whole object with wall/
        // dirt/trap/cheese fields -- more expensive per call for a
        // question that doesn't need that much detail). Deliberately
        // includes transient occupancy, not just permanent terrain:
        // "blocked by a wall" and "blocked by an ally standing there"
        // both mean the same thing for a movement decision made this
        // turn. Order matches allDirections/dirToInt (N,NE,E,SE,S,SW,W,NW).
        float canMoveN = rc.canMove(Direction.NORTH) ? 1.0f : 0.0f;
        float canMoveNE = rc.canMove(Direction.NORTHEAST) ? 1.0f : 0.0f;
        float canMoveE = rc.canMove(Direction.EAST) ? 1.0f : 0.0f;
        float canMoveSE = rc.canMove(Direction.SOUTHEAST) ? 1.0f : 0.0f;
        float canMoveS = rc.canMove(Direction.SOUTH) ? 1.0f : 0.0f;
        float canMoveSW = rc.canMove(Direction.SOUTHWEST) ? 1.0f : 0.0f;
        float canMoveW = rc.canMove(Direction.WEST) ? 1.0f : 0.0f;
        float canMoveNW = rc.canMove(Direction.NORTHWEST) ? 1.0f : 0.0f;

        // New fields are appended after the original 40 (canMove, ending at
        // index 39), not inserted -- keeps every existing column index
        // (e1dist at 2, e1health at 4, etc.) stable for anything already
        // indexing into the state by position (e.g. rl_train.py's RGPS
        // proxy). e_x fields are distance (/64); e_y fields are bearing-in-
        // degrees-from-north, normalized by /180 into (-1, 1] (0=north,
        // ~0.5=east, 1/-1=south, -0.5=west) -- see the loops above. Distance
        // is invariant under mirroring; bearing transforms under
        // rl_train.py's apply_symmetry the same way the old (x,y) pair did,
        // just expressed as an angle op instead of a sign flip -- see
        // SYMMETRY_ANGLE_FIELDS there. Enemy/ally slots 4-6 (appended
        // below, same 4-field layout as slots 1-3) widen the state from
        // K=3 to K=6 nearest of each -- 3 was blind to anything past the
        // closest few in a real group fight (allies flanking wide, or a
        // 4th+ enemy already in range), collapsing genuinely different
        // multi-unit situations onto the same 3-nearest reading.
        return new float[]{
            myX / 64.0f,
            myY / 64.0f,
            eDist[0] / 64.0f, eBear[0] / 180.0f, eHealth[0] / 100.0f, eDir[0] / 8.0f,
            eDist[1] / 64.0f, eBear[1] / 180.0f, eHealth[1] / 100.0f, eDir[1] / 8.0f,
            eDist[2] / 64.0f, eBear[2] / 180.0f, eHealth[2] / 100.0f, eDir[2] / 8.0f,
            dirToInt(rc.getDirection()) / 8.0f,
            Math.min(rc.getMovementCooldownTurns(), 20) / 20.0f,
            Math.min(rc.getActionCooldownTurns(), 20) / 20.0f,
            rc.getCarrying() != null ? 1.0f : 0.0f,
            rc.getHealth() / 100.0f,
            aDist[0] / 64.0f, aBear[0] / 180.0f, aHealth[0] / 100.0f, aDir[0] / 8.0f,
            localHealthSum / 100.0f,
            aDist[1] / 64.0f, aBear[1] / 180.0f, aHealth[1] / 100.0f, aDir[1] / 8.0f,
            aDist[2] / 64.0f, aBear[2] / 180.0f, aHealth[2] / 100.0f, aDir[2] / 8.0f,
            canMoveN, canMoveNE, canMoveE, canMoveSE, canMoveS, canMoveSW, canMoveW, canMoveNW,
            eDist[3] / 64.0f, eBear[3] / 180.0f, eHealth[3] / 100.0f, eDir[3] / 8.0f,
            eDist[4] / 64.0f, eBear[4] / 180.0f, eHealth[4] / 100.0f, eDir[4] / 8.0f,
            eDist[5] / 64.0f, eBear[5] / 180.0f, eHealth[5] / 100.0f, eDir[5] / 8.0f,
            aDist[3] / 64.0f, aBear[3] / 180.0f, aHealth[3] / 100.0f, aDir[3] / 8.0f,
            aDist[4] / 64.0f, aBear[4] / 180.0f, aHealth[4] / 100.0f, aDir[4] / 8.0f,
            aDist[5] / 64.0f, aBear[5] / 180.0f, aHealth[5] / 100.0f, aDir[5] / 8.0f,
        };
    }
}
