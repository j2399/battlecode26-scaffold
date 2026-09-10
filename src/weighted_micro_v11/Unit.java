package weighted_micro_v11;

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
    public static MapInfo[] nearby;
    public static Direction selfDir;

    public static boolean canAct;
    public static boolean canMove;

    // TODO add more global variables

    public static MapLocation target;
    public static final Direction[] adjacentDirections = {
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
        canAct=true;
        canMove=true;
        nearby = rc.senseNearbyMapInfos();
        selfDir=rc.getDirection();
    }

    /** Called at the start of every turn to update values */
    public static void update() throws GameActionException {
        allyRats= rc.senseNearbyRobots(-1, myTeam);
        enemyRats= rc.senseNearbyRobots(-1, opponentTeam);
        turnCount=rc.getRoundNum();
        canAct=rc.isActionReady();
        canMove=rc.isMovementReady();
        myLoc=rc.getLocation();
        nearby = rc.senseNearbyMapInfos();
        selfDir=rc.getDirection();

        // TODO add more things to maintain at each update (turn start and after move/turn)

    }

    /** Called when moving in a specified direction, turns before moving */
    public static void turn_then_move(Direction dir) throws GameActionException{
        if (!canMove) return;
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
