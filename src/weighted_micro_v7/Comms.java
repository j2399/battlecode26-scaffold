package weighted_micro_v7;

import battlecode.common.*;

/**
 * Wraps the 64-slot shared array (values 0-1023, king-write / anyone-read)
 * and squeak() broadcasts so King and Rat don't touch the raw RobotController
 * calls directly.
 */
public class Comms extends Unit {
    public static final int NUM_SLOTS = GameConstants.SHARED_ARRAY_SIZE;
    public static final int MAX_VALUE = GameConstants.COMM_ARRAY_MAX_VALUE;

    // TODO: implement comms
    public static final int SLOT_RAT_TARGET_X = 0;
    public static final int SLOT_RAT_TARGET_Y = 1;

    /** Only rat kings can write; safe to call from Rat.java, becomes a no-op there. */
    public static void write(int index, int value) throws GameActionException {
        if (rc.getType() != UnitType.RAT_KING) return;
        rc.writeSharedArray(index, value);
    }

    public static int read(int index) throws GameActionException {
        return rc.readSharedArray(index);
    }

    /** Broadcast a short message to nearby allies/cats within radius sqrt(16). Returns false if the per-turn squeak limit was hit. */
    public static boolean squeak(int messageContent) throws GameActionException {
        return rc.squeak(messageContent);
    }

    public static Message[] ratRead(){
        return rc.readSqueaks(rc.getRoundNum());
    }
    
}
