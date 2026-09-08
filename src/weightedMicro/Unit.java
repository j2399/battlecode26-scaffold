package weightedMicro;

import battlecode.common.*;

/**
 * Shared base for King and Rat: holds state initialized once at spawn and
 * refreshed every turn, plus constants both unit types use.
 */
public class Unit {
    public static RobotController rc;

    public static Team teamMy;
    public static Team teamOpponent;

    public static int w;
    public static int h;

    public static int round;
    public static int idMy;

    public static MapLocation locMy;
    public static RobotInfo[] ratsAlly;
    public static RobotInfo[] ratsEnemy;
    public static Message[] messages;
    public static MapInfo[] nearby;
    public static Direction dirMy;

    public static boolean actCan;
    public static boolean moveCan;

    // TODO add more global variables

    public static MapLocation target;
    public static final Direction[] directionsAdjacent = {
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
        teamMy = rc.getTeam();
        teamOpponent = teamMy.opponent();
        w = rc.getMapWidth();
        h = rc.getMapHeight();
        idMy = rc.getID();
        locMy=rc.getLocation();
        round=rc.getRoundNum();
        actCan=true;
        moveCan=true;
        nearby = rc.senseNearbyMapInfos();
        dirMy=rc.getDirection();
    }

    /** Called at the start of every turn to update values */
    public static void update() throws GameActionException {
        ratsAlly= rc.senseNearbyRobots(-1, teamMy);
        ratsEnemy= rc.senseNearbyRobots(-1, teamOpponent);
        round=rc.getRoundNum();
        actCan=rc.isActionReady();
        moveCan=rc.isMovementReady();
        locMy=rc.getLocation();
        nearby = rc.senseNearbyMapInfos();
        dirMy=rc.getDirection();

        // TODO add more things to maintain at each update (turn start and after move/turn)

    }

    /** Called when moving in a specified direction, turns before moving */
    public static void turnMove(Direction dir) throws GameActionException{
        if (!moveCan) return;
        boolean moved=false;
         if (rc.canMove(dir)){
            if (rc.canTurn(dir)){
                rc.turn(dir);
            }
            rc.move(dir);
            moved=true;
        }
        else { 
            dir=dir.rotateLeft();
            if (rc.canMove(dir)){
                if (rc.canTurn(dir)){
                    rc.turn(dir);
                }
                rc.move(dir);
                moved=true;
            }
            else { 
                dir=dir.rotateRight().rotateRight();
                    if (rc.canMove(dir)){
                        if (rc.canTurn(dir)){
                            rc.turn(dir);
                        }
                    rc.move(dir);
                    moved=true;
                }
            }
        }
        if (moved) update();
    }
}
