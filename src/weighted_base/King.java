package weighted_base;

import battlecode.common.*;

public class King extends Unit {

    // TODO inplement king logic

    public static void run(RobotController rc) throws GameActionException {
        

        trySpawnRat();
    }

    static void trySpawnRat() throws GameActionException {
        for (Direction dir : ALL_DIRECTIONS) {
             for (Direction dir2 : ALL_DIRECTIONS) {
                MapLocation spawnLoc = myLoc.add(dir).add(dir2);
                if (rc.canBuildRat(spawnLoc)) {
                    rc.buildRat(spawnLoc);
                    return;
                }
            }
        }
    }
}
