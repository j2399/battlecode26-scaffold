package econ5;

import battlecode.common.*;

import java.util.Arrays;
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

    // Same K-nearest insertion helper as learner_rl/RobotPlayer.java's
    // buildState() -- see that file's comment. Int-typed here to match
    // this file's existing convention of logging raw ints for train.py's
    // parse_obs() to scale on the Python side.
    static void insertNearest(int[] distSq, int[] dist, int[] bear, int[] health, int[] dir,
                               int newDistSq, int newDist, int newBear, int newHealth, int newDir) {
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
                int[] eDistSq = new int[K_NEAREST];
                int[] eDist = new int[K_NEAREST], eBear = new int[K_NEAREST], eHealth = new int[K_NEAREST];
                int[] eDir = new int[K_NEAREST];
                Arrays.fill(eDistSq, Integer.MAX_VALUE);
                for (RobotInfo ri : enemyRats) {
                    MapLocation el = ri.getLocation();
                    int d = cur.distanceSquaredTo(el);
                    int h = (int) ri.getHealth();
                    int dir = dirToInt(ri.getDirection());
                    int dx = el.x - myX, dy = el.y - myY;
                    int dist = (int) Math.round(Math.sqrt((double) dx * dx + (double) dy * dy));
                    int bearing = (int) Math.round(Math.toDegrees(Math.atan2(dx, dy)));
                    insertNearest(eDistSq, eDist, eBear, eHealth, eDir, d, dist, bearing, h, dir);
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
                int[] aDistSq = new int[K_NEAREST];
                int[] aDist = new int[K_NEAREST], aBear = new int[K_NEAREST], aHealth = new int[K_NEAREST];
                int[] aDir = new int[K_NEAREST];
                Arrays.fill(aDistSq, Integer.MAX_VALUE);
                for (RobotInfo ri : friendlyRats) {
                    if (ri.getID() == rc.getID()) continue;
                    MapLocation al = ri.getLocation();
                    int d = cur.distanceSquaredTo(al);
                    int h = (int) ri.getHealth();
                    int dir = dirToInt(ri.getDirection());
                    int dx = al.x - myX, dy = al.y - myY;
                    int dist = (int) Math.round(Math.sqrt((double) dx * dx + (double) dy * dy));
                    int bearing = (int) Math.round(Math.toDegrees(Math.atan2(dx, dy)));
                    insertNearest(aDistSq, aDist, aBear, aHealth, aDir, d, dist, bearing, h, dir);
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
                sb.append(eDist[0]).append(',').append(eBear[0]).append(',').append(eHealth[0]).append(',').append(eDir[0]).append(',');
                sb.append(eDist[1]).append(',').append(eBear[1]).append(',').append(eHealth[1]).append(',').append(eDir[1]).append(',');
                sb.append(eDist[2]).append(',').append(eBear[2]).append(',').append(eHealth[2]).append(',').append(eDir[2]).append(',');
                sb.append(myDir).append(',');
                sb.append(moveCd).append(',');
                sb.append(actionCd).append(',');
                sb.append(carrying).append(',');
                sb.append(ownHealthInt).append(',');
                sb.append(aDist[0]).append(',').append(aBear[0]).append(',').append(aHealth[0]).append(',').append(aDir[0]).append(',');
                sb.append(localHealthSum).append(',');
                sb.append(aDist[1]).append(',').append(aBear[1]).append(',').append(aHealth[1]).append(',').append(aDir[1]).append(',');
                sb.append(aDist[2]).append(',').append(aBear[2]).append(',').append(aHealth[2]).append(',').append(aDir[2]).append(',');
                sb.append(canMoveN).append(',').append(canMoveNE).append(',').append(canMoveE).append(',').append(canMoveSE).append(',');
                sb.append(canMoveS).append(',').append(canMoveSW).append(',').append(canMoveW).append(',').append(canMoveNW).append(',');
                // Appended slots 4-6, same (dist,bearing,health,dir) layout
                // as slots 1-3 above -- see learner_rl/RobotPlayer.java's
                // buildState() comment for why K widened from 3 to 6.
                sb.append(eDist[3]).append(',').append(eBear[3]).append(',').append(eHealth[3]).append(',').append(eDir[3]).append(',');
                sb.append(eDist[4]).append(',').append(eBear[4]).append(',').append(eHealth[4]).append(',').append(eDir[4]).append(',');
                sb.append(eDist[5]).append(',').append(eBear[5]).append(',').append(eHealth[5]).append(',').append(eDir[5]).append(',');
                sb.append(aDist[3]).append(',').append(aBear[3]).append(',').append(aHealth[3]).append(',').append(aDir[3]).append(',');
                sb.append(aDist[4]).append(',').append(aBear[4]).append(',').append(aHealth[4]).append(',').append(aDir[4]).append(',');
                sb.append(aDist[5]).append(',').append(aBear[5]).append(',').append(aHealth[5]).append(',').append(aDir[5]);
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
