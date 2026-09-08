package weighted_base;

import battlecode.common.*;

import java.util.Random;

public class ExploreState extends Unit {
    static final Random rng = new Random(6147);

    // TODO implement explore state 

    public static void run() throws GameActionException {
        Direction dir = ALL_DIRECTIONS[rng.nextInt(ALL_DIRECTIONS.length)];
        if (rc.canMove(dir)) {
            rc.move(dir);
        }
    }
}
