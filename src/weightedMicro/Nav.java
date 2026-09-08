package weightedMicro;

import battlecode.common.*;

public class Nav extends Unit {
    static MapLocation targetCurrent;

    // TODO implement real navigation (this is just a greedy stub)
    public static boolean toMove(MapLocation target) throws GameActionException {
        if (target == null || locMy.equals(target)) return false;
        targetCurrent = target;

        Direction targetTo = locMy.directionTo(target);
        if (rc.canMove(targetTo)) {
            rc.move(targetTo);
            return true;
        }

        Direction left = targetTo.rotateLeft();
        Direction right = targetTo.rotateRight();
        if (rc.canMove(left)) {
            rc.move(left);
            return true;
        }
        if (rc.canMove(right)) {
            rc.move(right);
            return true;
        }

        return false;
    }
}
