package weighted_micro_v18;

import battlecode.common.*;

/** Active when the rat king has broadcast a destination via the shared array. */
public class PathState extends Unit {
    static MapLocation destination;

    // TODO implement path state

    public static void run() throws GameActionException {
        if (destination == null) return;

        if (myLoc.equals(destination)) {
            destination = null;
            return;
        }
        Nav.moveTo(destination);
    }
}
