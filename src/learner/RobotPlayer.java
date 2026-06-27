package learner;

import battlecode.common.*;

public strictfp class RobotPlayer {
    static int turnCount = 0;

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

    static NeuralNet nn = new NeuralNet();

    static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) best = i;
        }
        return best;
    }

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

                Team opponent = rc.getTeam().opponent();
                RobotInfo[] enemies = rc.senseNearbyRobots(-1, opponent);

                if (enemies.length > 0) {
                    float[] state = buildState(rc, enemies);
                    float[][] q = nn.forward(state);

                    int bestAction    = argmax(q[2]);
                    int bestActionDir = argmax(q[3]);
                    if (bestAction > 0)
                        executeAction(rc, enemies, bestAction, bestActionDir);

                    int bestMoved  = argmax(q[0]);
                    int bestFacing = argmax(q[1]);

                    rc.setIndicatorString("m" + bestMoved + " f" + bestFacing + " a" + bestAction + " d" + bestActionDir);

                    if (bestMoved > 0) {
                        Direction d = allDirections[bestMoved];
                        if (rc.getDirection() != d && rc.canTurn(d))
                            rc.turn(d);
                        if (rc.isMovementReady()) {
                            if (rc.canMove(d)) {
                                rc.move(d);
                            } else if (rc.canMove(d.rotateLeft())) {
                                d = d.rotateLeft();
                                rc.move(d);
                            } else if (rc.canMove(d.rotateRight())) {
                                d = d.rotateRight();
                                rc.move(d);
                            }
                        }
                    } else {
                        Direction targetFacing = allDirections[bestFacing];
                        if (rc.getDirection() != targetFacing && rc.canTurn(targetFacing))
                            rc.turn(targetFacing);
                    }

                    enemies = rc.senseNearbyRobots(-1, opponent);
                    if (enemies.length > 0) {
                        state = buildState(rc, enemies);
                        q = nn.forward(state);
                        bestAction    = argmax(q[2]);
                        bestActionDir = argmax(q[3]);
                        if (bestAction > 0)
                            executeAction(rc, enemies, bestAction, bestActionDir);
                    }
                } else {
                    rc.setIndicatorString("RUSH");
                    if (rc.isMovementReady())
                        Navigator.moveTo(Globals.oppositeLocation);
                    enemies = rc.senseNearbyRobots(-1, opponent);
                    if (enemies.length > 0) {
                        float[] state = buildState(rc, enemies);
                        float[][] q = nn.forward(state);
                        int bestAction    = argmax(q[2]);
                        int bestActionDir = argmax(q[3]);
                        if (bestAction > 0)
                            executeAction(rc, enemies, bestAction, bestActionDir);
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

    static void executeAction(RobotController rc, RobotInfo[] enemies, int actionType, int actionDir) throws GameActionException {
        if (!rc.isActionReady()) return;

        Direction turnDir = allDirections[actionDir];
        if (rc.getDirection() != turnDir && rc.canTurn(turnDir))
            rc.turn(turnDir);

        if (actionType == 1) {
            MapLocation myLoc = rc.getLocation();
            RobotInfo closest = null;
            int closestDist = Integer.MAX_VALUE;
            for (RobotInfo ri : enemies) {
                int d = myLoc.distanceSquaredTo(ri.getLocation());
                if (d < closestDist) {
                    closestDist = d;
                    closest = ri;
                }
            }
            if (closest != null && rc.canAttack(closest.getLocation()))
                rc.attack(closest.getLocation());
        } else if (actionType == 2) {
            for (RobotInfo ri : enemies) {
                MapLocation el = ri.getLocation();
                if (ri.getType() == UnitType.BABY_RAT && rc.canCarryRat(el)) {
                    rc.carryRat(el);
                    return;
                }
            }
        } else if (actionType == 3) {
            if (rc.canThrowRat())
                rc.throwRat();
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
