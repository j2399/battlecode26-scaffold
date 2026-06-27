package learner;

import battlecode.common.*;

public class BabyRat extends Globals {

    static final int MAP_SIZE = 64 * 64;
    static final int OWN_STATS = 4;
    static final int MAX_ENEMIES = 30;
    static final int ENEMY_SLOTS = MAX_ENEMIES * 3;
    static final int MAX_ALLIES = 10;
    static final int ALLY_SLOTS = MAX_ALLIES * 3;
    static final int ARRAY_SIZE = MAP_SIZE + OWN_STATS + ENEMY_SLOTS + ALLY_SLOTS;

    public static int[] obs = new int[ARRAY_SIZE];

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

    public static void run(RobotController rc) throws GameActionException {
        MapLocation myLoc = rc.getLocation();

        for (MapInfo mi : rc.senseNearbyMapInfos()) {
            MapLocation loc = mi.getMapLocation();
            int idx = loc.x * 64 + loc.y;
            if (mi.isWall()) {
                obs[idx] = 3;
            } else if (mi.isDirt()) {
                obs[idx] = 2;
            } else {
                obs[idx] = 1;
            }
        }

        obs[MAP_SIZE] = (int) rc.getHealth();
        obs[MAP_SIZE + 1] = dirToInt(rc.getDirection());
        obs[MAP_SIZE + 2] = rc.getMovementCooldownTurns();
        obs[MAP_SIZE + 3] = rc.getActionCooldownTurns();

        int enemySlot = MAP_SIZE + OWN_STATS;
        for (int i = 0; i < MAX_ENEMIES; i++) {
            obs[enemySlot + i * 3] = 0;
            obs[enemySlot + i * 3 + 1] = 0;
            obs[enemySlot + i * 3 + 2] = 0;
        }

        int e = 0;
        for (RobotInfo ri : rc.senseNearbyRobots(-1, opponentTeam)) {
            if (e >= MAX_ENEMIES) break;
            MapLocation loc = ri.getLocation();
            obs[enemySlot + e * 3] = loc.x * 64 + loc.y;
            obs[enemySlot + e * 3 + 1] = (int) ri.getHealth();
            obs[enemySlot + e * 3 + 2] = dirToInt(ri.getDirection());
            e++;
        }

        int allySlot = enemySlot + ENEMY_SLOTS;
        for (int i = 0; i < MAX_ALLIES; i++) {
            obs[allySlot + i * 3] = 0;
            obs[allySlot + i * 3 + 1] = 0;
            obs[allySlot + i * 3 + 2] = 0;
        }

        int a = 0;
        for (RobotInfo ri : rc.senseNearbyRobots(-1, myTeam)) {
            if (ri.getType() != UnitType.BABY_RAT) continue;
            if (a >= MAX_ALLIES) break;
            MapLocation loc = ri.getLocation();
            obs[allySlot + a * 3] = loc.x * 64 + loc.y;
            obs[allySlot + a * 3 + 1] = (int) ri.getHealth();
            obs[allySlot + a * 3 + 2] = dirToInt(ri.getDirection());
            a++;
        }
    }
}
