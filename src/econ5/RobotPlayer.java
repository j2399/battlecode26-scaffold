package econ5;

import battlecode.common.*;

import java.util.Random;


/**
 * RobotPlayer is the class that describes your main robot strategy.
 * The run() method inside this class is like your main function: this is what we'll call once your robot
 * is created!
 */
public strictfp class RobotPlayer {
    /**
     * We will use this variable to count the number of turns this robot has been alive.
     * You can use static variables like this to save any information you want. Keep in mind that even though
     * these variables are static, in Battlecode they aren't actually shared between your robots.
     */
    static int turnCount = 0;

    /**
     * A random number generator.
     * We will use this RNG to make some random moves. The Random class is provided by the java.util.Random
     * import at the top of this file. Here, we *seed* the RNG with a constant number (6147); this makes sure
     * we get the same sequence of numbers every time this code is run. This is very useful for debugging!
     */
    static final Random rng = new Random(6147);
    static boolean prevCarried = false;

    /** Array containing all the possible movement directions. */
    static final Direction[] directions = {
            Direction.NORTH,
            Direction.NORTHEAST,
            Direction.EAST,
            Direction.SOUTHEAST,
            Direction.SOUTH,
            Direction.SOUTHWEST,
            Direction.WEST,
            Direction.NORTHWEST,
    };

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
        switch (d) {
            case CENTER:     return 0;
            case NORTH:      return 1;
            case NORTHEAST:  return 2;
            case EAST:       return 3;
            case SOUTHEAST:  return 4;
            case SOUTH:      return 5;
            case SOUTHWEST:  return 6;
            case WEST:       return 7;
            case NORTHWEST:  return 8;
            default:         return 0;
        }
    }

    /**
     * run() is the method that is called when a robot is instantiated in the Battlecode world.
     * It is like the main function for your robot. If this method returns, the robot dies!
     *
     * @param rc  The RobotController object. You use it to perform actions from this robot, and to get
     *            information on its current status. Essentially your portal to interacting with the world.
     **/
    @SuppressWarnings("unused")
    public static void run(RobotController rc) throws GameActionException {
        try {
            Globals.init(rc);
        }  catch (Exception e) {
        e.printStackTrace();

    }
        // Hello world! Standard output is very useful for debugging.
        // Everything you say here will be directly viewable in your terminal when you run a match!
       // System.out.println("I'm alive");

        // You can also use indicators to save debug notes in replays.
       // rc.setIndicatorString("Hello world!");

        while (true) {

            // This code runs during the entire lifespan of the robot, which is why it is in an infinite
            // loop. If we ever leave this loop and return from run(), the robot dies! At the end of the
            // loop, we call Clock.yield(), signifying that we've done everything we want to do.

            turnCount += 1;  // We have now been alive for one more turn!

            // Try/catch blocks stop unhandled exceptions, which cause your robot to explode.

            try {
                Globals.turnValue = 0;
                Globals.attackTargetDir = 0;
                Globals.carryTargetDir = 0;
                Globals.throwTargetDir = 0;
                double prevHealth = rc.getHealth();

                switch (rc.getType()){
                    case RAT_KING -> RatKing.run(rc);
                    case BABY_RAT -> BabyRat.run(rc);
                }

                RobotInfo[] friendlyRats = rc.senseNearbyRobots(-1, rc.getTeam());
                int enemyCount = rc.senseNearbyRobots(-1, rc.getTeam().opponent()).length;
                Globals.turnValue += 0.5 * enemyCount;

                double healthLost = Math.max(0, prevHealth - rc.getHealth());
                Globals.turnValue -= healthLost;

                if (rc.getMovementCooldownTurns() <= 10) {
                    Globals.turnValue += 1;
                }

                MapLocation cur = rc.getLocation();
                int myX = cur.x;
                int myY = cur.y;
                int myDir = dirToInt(rc.getDirection());
                int moveCd = rc.getMovementCooldownTurns();
                int actionCd = rc.getActionCooldownTurns();
                int carrying = rc.getCarrying() != null ? 1 : 0;

                // e_x/e_y now log (distance, bearing-in-degrees-from-north)
                // instead of raw (dx, dy) -- mirrors learner_rl/RobotPlayer.
                // java's buildState() (see that file's comment for the
                // bearing convention/atan2 argument order); both stay raw
                // here (un-normalized), matching this file's existing
                // convention of logging raw ints for train.py's parse_obs()
                // to scale on the Python side.
                RobotInfo[] enemyRats = rc.senseNearbyRobots(-1, rc.getTeam().opponent());
                int e1x = 0, e1y = 0, e1h = 0, e1d = 0;
                int e2x = 0, e2y = 0, e2h = 0, e2d = 0;
                int e3x = 0, e3y = 0, e3h = 0, e3d = 0;
                int best1 = Integer.MAX_VALUE, best2 = Integer.MAX_VALUE, best3 = Integer.MAX_VALUE;
                for (RobotInfo ri : enemyRats) {
                    MapLocation el = ri.getLocation();
                    int d = cur.distanceSquaredTo(el);
                    int h = (int) ri.getHealth();
                    int dir = dirToInt(ri.getDirection());
                    int dx = el.x - myX, dy = el.y - myY;
                    int dist = (int) Math.round(Math.sqrt((double) dx * dx + (double) dy * dy));
                    int bearing = (int) Math.round(Math.toDegrees(Math.atan2(dx, dy)));
                    if (d < best1) {
                        best3 = best2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                        best2 = best1; e2x = e1x; e2y = e1y; e2h = e1h; e2d = e1d;
                        best1 = d; e1x = dist; e1y = bearing; e1h = h; e1d = dir;
                    } else if (d < best2) {
                        best3 = best2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                        best2 = d; e2x = dist; e2y = bearing; e2h = h; e2d = dir;
                    } else if (d < best3) {
                        best3 = d; e3x = dist; e3y = bearing; e3h = h; e3d = dir;
                    }
                }

                // Nearest 3 allies (excluding self), own health, local
                // numbers/health advantage, and canMove flags -- mirrors
                // learner_rl/RobotPlayer.java's buildState() exactly (see
                // that file for the reasoning behind each field), so the
                // dataset this produces can train a model over the SAME
                // 40-dim state learner_rl actually uses, instead of only
                // the original 18 raw fields below. Heard-squeak-based
                // ally awareness is deliberately not replicated here --
                // econ5's own robots never call rc.squeak(), so there'd be
                // nothing to hear; the visible-allies loop below is the
                // real source of ally data either way.
                int ownHealthInt = (int) rc.getHealth();
                int a1 = Integer.MAX_VALUE, a2 = Integer.MAX_VALUE, a3 = Integer.MAX_VALUE;
                int allyX = 0, allyY = 0, ally2X = 0, ally2Y = 0, ally3X = 0, ally3Y = 0;
                int allyH = 0, ally2H = 0, ally3H = 0;
                int allyD = 0, ally2D = 0, ally3D = 0;
                for (RobotInfo ri : friendlyRats) {
                    if (ri.getID() == rc.getID()) continue;
                    MapLocation al = ri.getLocation();
                    int d = cur.distanceSquaredTo(al);
                    int h = (int) ri.getHealth();
                    int dir = dirToInt(ri.getDirection());
                    int dx = al.x - myX, dy = al.y - myY;
                    int dist = (int) Math.round(Math.sqrt((double) dx * dx + (double) dy * dy));
                    int bearing = (int) Math.round(Math.toDegrees(Math.atan2(dx, dy)));
                    if (d < a1) {
                        a3 = a2; ally3X = ally2X; ally3Y = ally2Y; ally3H = ally2H; ally3D = ally2D;
                        a2 = a1; ally2X = allyX; ally2Y = allyY; ally2H = allyH; ally2D = allyD;
                        a1 = d;  allyX = dist; allyY = bearing; allyH = h; allyD = dir;
                    } else if (d < a2) {
                        a3 = a2; ally3X = ally2X; ally3Y = ally2Y; ally3H = ally2H; ally3D = ally2D;
                        a2 = d;  ally2X = dist; ally2Y = bearing; ally2H = h; ally2D = dir;
                    } else if (d < a3) {
                        a3 = d;  ally3X = dist; ally3Y = bearing; ally3H = h; ally3D = dir;
                    }
                }

                int localHealthSum = ownHealthInt;
                for (RobotInfo ri : friendlyRats) {
                    if (ri.getID() == rc.getID()) continue;
                    if (ri.getType() != UnitType.BABY_RAT) continue;
                    localHealthSum += (int) ri.getHealth();
                }
                for (RobotInfo ri : enemyRats) {
                    if (ri.getType() != UnitType.BABY_RAT) continue;
                    localHealthSum -= (int) ri.getHealth();
                }

                int canMoveN = rc.canMove(Direction.NORTH) ? 1 : 0;
                int canMoveNE = rc.canMove(Direction.NORTHEAST) ? 1 : 0;
                int canMoveE = rc.canMove(Direction.EAST) ? 1 : 0;
                int canMoveSE = rc.canMove(Direction.SOUTHEAST) ? 1 : 0;
                int canMoveS = rc.canMove(Direction.SOUTH) ? 1 : 0;
                int canMoveSW = rc.canMove(Direction.SOUTHWEST) ? 1 : 0;
                int canMoveW = rc.canMove(Direction.WEST) ? 1 : 0;
                int canMoveNW = rc.canMove(Direction.NORTHWEST) ? 1 : 0;

                boolean carriedNow = rc.isBeingCarried();
                boolean justCaptured = carriedNow && !prevCarried;
                prevCarried = carriedNow;

                StringBuilder sb = new StringBuilder();
                sb.append(rc.getID()).append(',');
                sb.append(myX).append(',');
                sb.append(myY).append(',');
                sb.append(e1x).append(',').append(e1y).append(',').append(e1h).append(',').append(e1d).append(',');
                sb.append(e2x).append(',').append(e2y).append(',').append(e2h).append(',').append(e2d).append(',');
                sb.append(e3x).append(',').append(e3y).append(',').append(e3h).append(',').append(e3d).append(',');
                sb.append(myDir).append(',');
                sb.append(moveCd).append(',');
                sb.append(actionCd).append(',');
                sb.append(carrying).append(',');
                sb.append(ownHealthInt).append(',');
                sb.append(allyX).append(',').append(allyY).append(',').append(allyH).append(',').append(allyD).append(',');
                sb.append(localHealthSum).append(',');
                sb.append(ally2X).append(',').append(ally2Y).append(',').append(ally2H).append(',').append(ally2D).append(',');
                sb.append(ally3X).append(',').append(ally3Y).append(',').append(ally3H).append(',').append(ally3D).append(',');
                sb.append(canMoveN).append(',').append(canMoveNE).append(',').append(canMoveE).append(',').append(canMoveSE).append(',');
                sb.append(canMoveS).append(',').append(canMoveSW).append(',').append(canMoveW).append(',').append(canMoveNW);
                if (justCaptured) sb.append(",C");
                String vs = String.format("%.1f", Globals.turnValue);
                sb.append('|').append(vs).append('|');
                for (int i = 0; i < 9; i++) {
                    MapLocation tile = cur.add(allDirections[i]);
                    int ts = 0;
                    if (i == 0 || rc.onTheMap(tile)) {
                        try {
                            ts = TileScore.score(rc, enemyRats, friendlyRats, tile, null, null);
                        } catch (Exception e) {
                            ts = 0;
                        }
                    }
                    if (i > 0) sb.append(',');
                    sb.append(ts);
                }
                rc.setIndicatorString(vs);
                System.out.println(sb);
            }
            catch (GameActionException e) {
                e.printStackTrace();

            } catch (Exception e) {
                e.printStackTrace();

            } finally {
                // Signify we've done everything we want to do, thereby ending our turn.
                // This will make our code wait until the next turn, and then perform this loop again.
                Clock.yield();
            }
            // End of loop: go back to the top. Clock.yield() has ended, so it's time for another turn!
        }

        // Your code should never reach here (unless it's intentional)! Self-destruction imminent...
    }
}
