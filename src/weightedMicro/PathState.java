package weightedMicro;

import battlecode.common.*;

/** Active when the rat king has broadcast a destination via the shared array. */
public class PathState extends Unit {
    static MapLocation destination;

    // TODO implement path state

    public static void run() throws GameActionException {
        if (destination == null) return;

        if (locMy.equals(destination)) {
            destination = null;
            return;
        }
        Nav.toMove(destination);
    }
}
