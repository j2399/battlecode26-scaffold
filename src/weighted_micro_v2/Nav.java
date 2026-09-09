package weighted_micro_v2;

import battlecode.common.*;

public class Nav extends Unit {
    static MapLocation currentTarget;

    // TODO implement real navigation (this is just a greedy stub)
    public static boolean moveTo(MapLocation target) throws GameActionException {
        if (target == null || myLoc.equals(target)) return false;
        currentTarget = target;

        Direction toTarget = myLoc.directionTo(target);
        if (rc.canMove(toTarget)) {
            rc.move(toTarget);
            return true;
        }

        Direction left = toTarget.rotateLeft();
        Direction right = toTarget.rotateRight();
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
