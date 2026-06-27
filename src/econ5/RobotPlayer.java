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
                    if (d < best1) {
                        best3 = best2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                        best2 = best1; e2x = e1x; e2y = e1y; e2h = e1h; e2d = e1d;
                        best1 = d; e1x = el.x - myX; e1y = el.y - myY; e1h = h; e1d = dir;
                    } else if (d < best2) {
                        best3 = best2; e3x = e2x; e3y = e2y; e3h = e2h; e3d = e2d;
                        best2 = d; e2x = el.x - myX; e2y = el.y - myY; e2h = h; e2d = dir;
                    } else if (d < best3) {
                        best3 = d; e3x = el.x - myX; e3y = el.y - myY; e3h = h; e3d = dir;
                    }
                }

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
                sb.append(carrying);
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
