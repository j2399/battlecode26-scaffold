package weightedMicro;

import battlecode.common.*;

public class King extends Unit {

    // TODO inplement king logic

    public static void run() throws GameActionException {
        
        if (rc.getAllCheese()>300){
        spawnRatTry();
        }
    }

    static void spawnRatTry() throws GameActionException {
        for (Direction dir : directionsAdjacent) {
             for (Direction dir2 : directionsAdjacent) {
                MapLocation locSpawn = locMy.add(dir).add(dir2);
                if (rc.canBuildRat(locSpawn)) {
                    rc.buildRat(locSpawn);
                    return;
                }
            }
        }
    }
}
