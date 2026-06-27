package rush;


import battlecode.common.Direction;
import battlecode.common.GameActionException;
import battlecode.common.MapInfo;
import battlecode.common.MapLocation;
import rush.Globals;
import rush.WbugNav;

/* ??????????????? FIX
Adapted from Just Woke Up's 2025 submission, posted on BattleCode
 */


public class Navigator extends Globals {
    private static MapLocation currentTarget;
    private static int lastDistance = Integer.MAX_VALUE;
    private static int stuckTurns = 0;

    public static void moveTo(MapLocation target) throws GameActionException {
        MapLocation myLoc = rc.getLocation();
        if (myLoc.equals(target)) return;

        if (currentTarget == null || !currentTarget.equals(target)) {
            reset();
        }
        currentTarget = target;

        if (!rc.isMovementReady()) return;

        int curDist = myLoc.distanceSquaredTo(target);

        // Progress tracking
        if (curDist < lastDistance) {
            lastDistance = curDist;
            stuckTurns = 0;
            WbugNav.reset();
        } else {
            stuckTurns++;
        }

        // 1️⃣ Try greedy move
        Direction greedy = myLoc.directionTo(target);
        if (rc.canMove(greedy)) {
            if (rc.canTurn() && greedy!=Direction.CENTER){
                rc.turn(greedy);
            }
            rc.move(greedy);
            return;
        }

        // 2️⃣ Try any move that DOES NOT increase distance
        for (Direction d : Direction.values()) {
            if (!rc.canMove(d)) continue;
            MapLocation next = myLoc.add(d);
            if (next.distanceSquaredTo(target) <= curDist) {
                if (rc.canTurn() && d!=Direction.CENTER){
                    rc.turn(d);
                }
                rc.move(d);
                return;
            }
        }

        // 3️⃣ If blocked by robots only → WAIT
        boolean terrainBlocked = false;
        for (Direction d : Direction.values()) {
            MapLocation next = myLoc.add(d);
            if (rc.canSenseLocation(next)) {
                MapInfo info = rc.senseMapInfo(next);
                if (!info.isPassable()) {
                    terrainBlocked = true;
                    break;
                }
            }
        }

        if (!terrainBlocked) {
            // Only robots are blocking — do NOT wander
            return;
        }

        // 4️⃣ Absolute last resort: bug nav
        WbugNav.moveTo(target);

        // 5️⃣ Emergency reset
        if (stuckTurns > 10) {
            reset();
        }
    }

    public static void reset() {
        currentTarget = null;
        lastDistance = Integer.MAX_VALUE;
        stuckTurns = 0;
        WbugNav.reset();
    }
}

