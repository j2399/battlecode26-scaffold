package testbot;

import battlecode.common.*;

public class RobotPlayer {
    // Shared array indices for global state (0-63 available)
    static final int WEIGHT_START = 0;      // Start of weight array
    static final int NUM_WEIGHTS = 8;       // Number of weights (8 * 4 = 32 total)
    static final int BEST_FITNESS_IDX = 8;   // Best fitness value
    static final int UPDATE_COUNT_IDX = 9;   // Number of updates

    public static void run(RobotController rc) throws GameActionException {
        while (true) {
            int round = rc.getRoundNum();

            if (round == 1) {
                // Initialize weights to 0 in first round
                for (int i = 0; i < NUM_WEIGHTS; i++) {
                    rc.writeSharedArray(WEIGHT_START + i, 0);
                }
                rc.writeSharedArray(BEST_FITNESS_IDX, 0);
                rc.writeSharedArray(UPDATE_COUNT_IDX, 0);
            }

            // Simple random movement
            Direction[] dirs = Direction.values();
            int dirIdx = (rc.getID() + round) % dirs.length;
            if (rc.canMove(dirs[dirIdx])) {
                rc.move(dirs[dirIdx]);
            }

            // Update weights - compute new value first, then write (max value is 1023)
            int currentWeight = rc.readSharedArray(WEIGHT_START);
            int updates = rc.readSharedArray(UPDATE_COUNT_IDX);
            
            // Compute new value in valid range before writing
            int newWeight = (currentWeight + 1);
            if (newWeight > 1023) newWeight = 0;
            
            int newUpdates = updates + 1;
            if (newUpdates > 1023) newUpdates = 0;
            
            rc.writeSharedArray(WEIGHT_START, newWeight);
            rc.writeSharedArray(UPDATE_COUNT_IDX, newUpdates);

            if (round % 100 == 0) {
                int w0 = rc.readSharedArray(WEIGHT_START);
                int up = rc.readSharedArray(UPDATE_COUNT_IDX);
                rc.setIndicatorString("w0=" + w0 + " up=" + up);
            }

            Clock.yield();
        }
    }
}