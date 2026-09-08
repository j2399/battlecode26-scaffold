package weighted_base;

import battlecode.common.*;

/**
 * Shared base for King and Rat: holds state initialized once at spawn and
 * refreshed every turn, plus constants both unit types use.
 */
public class Unit {
    public static RobotController rc;

    public static Team myTeam;
    public static Team opponentTeam;

    public static int mapWidth;
    public static int mapHeight;

    public static int turnCount;
    public static int myID;

    public static MapLocation myLoc;
    public static RobotInfo[] allyRats;
    public static RobotInfo[] enemyRats;
    public static Message[] messages;

    // TODO add more global variables

    public static MapLocation target;
    public static final Direction[] ALL_DIRECTIONS = {
        Direction.NORTH,
        Direction.NORTHEAST,
        Direction.EAST,
        Direction.SOUTHEAST,
        Direction.SOUTH,
        Direction.SOUTHWEST,
        Direction.WEST,
        Direction.NORTHWEST,
    };

    /** Called once, right after the robot spawns. */
    public static void init(RobotController robotController) throws GameActionException {
        rc = robotController;
        myTeam = rc.getTeam();
        opponentTeam = myTeam.opponent();
        mapWidth = rc.getMapWidth();
        mapHeight = rc.getMapHeight();
        myID = rc.getID();
        myLoc=rc.getLocation();
        turnCount=rc.getRoundNum();
    }
    public static void update() throws GameActionException {
        allyRats= rc.senseNearbyRobots(-1, myTeam);
        enemyRats= rc.senseNearbyRobots(-1, opponentTeam);
        turnCount=rc.getRoundNum();

        // TODO add more things to maintain at the start of each round in update


    }
}
