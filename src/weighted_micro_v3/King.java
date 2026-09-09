package weighted_micro_v3;

import battlecode.common.*;

public class King extends Unit {

    // TODO inplement king logic

    public static void run() throws GameActionException {
        
        if (rc.getAllCheese()>300){
        trySpawnRat();
        }
    }

    static void trySpawnRat() throws GameActionException {
        for (Direction dir : adjacentDirections) {
             for (Direction dir2 : adjacentDirections) {
                MapLocation spawnLoc = myLoc.add(dir).add(dir2);
                if (rc.canBuildRat(spawnLoc)) {
                    rc.buildRat(spawnLoc);
                    return;
                }
            }
        }
    }
}
