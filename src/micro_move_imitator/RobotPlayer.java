package micro_move_imitator;

import battlecode.common.*;

public strictfp class RobotPlayer {
    static int turnCount = 0;
    static NeuralNet nn = new NeuralNet();

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

                Team opponent = rc.getTeam().opponent();
                RobotInfo[] enemies = rc.senseNearbyRobots(-1, opponent);

                // Saved before this turn's move so the pre-move ideal-direction
                // check below and the post-move faceClosest() can both consider
                // every enemy seen this turn, not just whichever one was
                // closest before we moved -- a different pre-move enemy can
                // become the truly closest one once our own position changes.
                RobotInfo[] preMoveEnemies = enemies;

                autoActions(rc, enemies);

                if (enemies.length == 0) {
                    rc.setIndicatorString("RUSH");
                    if (rc.isMovementReady())
                        Navigator.moveTo(Globals.oppositeLocation);
                    enemies = rc.senseNearbyRobots(-1, opponent);
                    autoActions(rc, enemies);
                } else {
                    float[] state = buildState(rc, enemies);
                    float[] q = nn.forward(state);
                    int bestTile = argmax(q);

                    rc.setIndicatorString("tile" + bestTile);

                    // If the direction we're about to move in already faces the
                    // closest enemy we saw before moving, turn+move that way in
                    // one step -- we end up facing correctly as a byproduct of
                    // the move, so there's no need to re-face afterward. Only
                    // when that doesn't hold (or we can't move that way) do we
                    // fall back to moving via the rotate cascade and figuring
                    // out facing separately once the move (and this turn's
                    // actions) have actually happened.
                    boolean facedTowardMove = false;

                    if (bestTile > 0) {
                        Direction dirIdeal = allDirections[bestTile];
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
                    autoActions(rc, enemies);

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

    static void autoActions(RobotController rc, RobotInfo[] enemies) throws GameActionException {
        if (!rc.isActionReady()) return;

        if (rc.getCarrying() != null && rc.canThrowRat() && enemies.length > 0 && rc.canTurn()) {
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
            if (rc.canThrowRat())
                rc.throwRat();
        }

        if (!rc.isActionReady()) return;

        for (RobotInfo ri : enemies) {
            if (ri.getType() == UnitType.RAT_KING) {
                MapLocation kingLoc = ri.getLocation();
                Direction toKing = rc.getLocation().directionTo(kingLoc);
                if (rc.getDirection() != toKing && rc.canTurn(toKing))
                    rc.turn(toKing);
                if (rc.canThrowRat() && rc.getDirection() == toKing) {
                    rc.throwRat();
                    return;
                }
                MapLocation atkLoc = kingLoc.subtract(toKing);
                if (rc.canAttack(atkLoc)) {
                    rc.attack(atkLoc);
                    return;
                }
                break;
            }
        }

        if (!rc.isActionReady()) return;
        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            if (ri.getType() == UnitType.BABY_RAT && rc.canCarryRat(el)) {
                rc.carryRat(el);
                return;
            }
        }
        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            if (rc.canAttack(el)) {
                rc.attack(el);
                return;
            }
        }
    }

    static float[] buildState(RobotController rc, RobotInfo[] enemies) throws GameActionException {
        MapLocation loc = rc.getLocation();
        int myX = loc.x, myY = loc.y;

        int b1 = Integer.MAX_VALUE, b2 = Integer.MAX_VALUE, b3 = Integer.MAX_VALUE;
        int e1x = 0, e1y = 0, e2x = 0, e2y = 0, e3x = 0, e3y = 0;
        float e1h = 0, e2h = 0, e3h = 0;
        int e1d = 0, e2d = 0, e3d = 0;

        for (RobotInfo ri : enemies) {
            MapLocation el = ri.getLocation();
            int d = loc.distanceSquaredTo(el);
            float h = ri.getHealth();
            int dir = dirToInt(ri.getDirection());
            if (d < b1) {
                b3 = b2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                b2 = b1; e2x = e1x; e2y = e1y; e2h = e1h; e2d = e1d;
                b1 = d;  e1x = el.x - myX; e1y = el.y - myY; e1h = h; e1d = dir;
            } else if (d < b2) {
                b3 = b2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                b2 = d;  e2x = el.x - myX; e2y = el.y - myY; e2h = h; e2d = dir;
            } else if (d < b3) {
                b3 = d;  e3x = el.x - myX; e3y = el.y - myY; e3h = h; e3d = dir;
            }
        }

        return new float[]{
            myX / 64.0f,
            myY / 64.0f,
            e1x / 64.0f, e1y / 64.0f, e1h / 100.0f, e1d / 8.0f,
            e2x / 64.0f, e2y / 64.0f, e2h / 100.0f, e2d / 8.0f,
            e3x / 64.0f, e3y / 64.0f, e3h / 100.0f, e3d / 8.0f,
            dirToInt(rc.getDirection()) / 8.0f,
            Math.min(rc.getMovementCooldownTurns(), 20) / 20.0f,
            Math.min(rc.getActionCooldownTurns(), 20) / 20.0f,
            rc.getCarrying() != null ? 1.0f : 0.0f,
        };
    }
}
