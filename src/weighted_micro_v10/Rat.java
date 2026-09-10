package weighted_micro_v10;

import battlecode.common.*;

/**
 * Baby rat control: a small state machine over ExploreState / CombatState /
 * PathState. Rat.run() only decides *which* state should act this turn;
 * each state class holds its own behavior.
 */
public class Rat extends Unit {

    public enum Mode {
        EXPLORE,
        COMBAT,
        PATH,
    }

    public static Mode mode = Mode.EXPLORE;

    public static void run() throws GameActionException {


        // TODO use messages to get target and or other information
        messages=Comms.ratRead();

        updateMode();

        switch (mode) {
            case COMBAT -> CombatState.run();
            case PATH -> PathState.run();
            case EXPLORE -> ExploreState.run();
        }
    }

    /** Decide which state should own this turn, based on what we currently sense. */
    static void updateMode() throws GameActionException {
        if (enemyRats.length!=0) {
            mode = Mode.COMBAT;
        } else if (target != null) {
            mode = Mode.PATH;
        } else {
            mode = Mode.EXPLORE;
        }
    }

}
