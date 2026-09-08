package prev_weighted_micro;

import battlecode.common.*;

public class CombatTileScore extends Unit {
    public static int tile_score(MapLocation tile_loc, boolean can_act) throws GameActionException{
        //todo implement and optimize getting tile scores
        return 1;
    }

    // returns score of attacking, laying a mine, and throwing
    public static int[] action_score(MapLocation bot_loc) throws GameActionException{
        //todo implement and optimize getting score of acting now
        return new int[]{2, 1, 1};
    }
}
