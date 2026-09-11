package weighted_micro_evo150_c4;

import battlecode.common.*;

/** Active when an attackable enemy rat is in range. */
public class CombatState extends Unit {

    // TODO implement combat state

    // Diagnostics for this turn's end-of-turn debug print: which action(s)
    // actually executed (comma-separated, in execution order) and total
    // damage dealt this turn. Reset at the top of run(), appended to by
    // whichever action method(s) actually fire below.
    static String turnActions = "";
    static int turnDamage = 0;

    public static void run() throws GameActionException {
        turnActions = "";
        turnDamage = 0;

        // act after moving
        int best_act_score=Integer.MIN_VALUE;
        Direction best_act_dir=Direction.NORTH;
        // no act after moving
        int best_no_act_score=Integer.MIN_VALUE;
        Direction best__no_act_dir=Direction.NORTH;

        StringBuilder noActLog = new StringBuilder();
        StringBuilder actLog = new StringBuilder();

        for(Direction dir : adjacentDirections){
                MapLocation loc=myLoc.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.tile_score(loc, false);
                    noActLog.append(dir).append("=").append(score).append(" ");
                    if (score>best_no_act_score){
                        best_no_act_score=score;
                        best__no_act_dir=dir;
                    }
                } else {
                    noActLog.append(dir).append("=x ");
                }
            }

        int[] scores = {0, 0, 0, 0};

        if (canAct){
             for(Direction dir : adjacentDirections){
                MapLocation loc=myLoc.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.tile_score(loc, true);
                    actLog.append(dir).append("=").append(score).append(" ");
                    if (score>best_act_score){
                        best_act_score=score;
                        best_act_dir=dir;
                    }
                } else {
                    actLog.append(dir).append("=x ");
                }
            }
            scores =CombatTileScore.action_score(myLoc);
            int act_score=Math.max(Math.max(scores[0],scores[1]),Math.max(scores[2],scores[3]));
            if(best_no_act_score+act_score>best_act_score){
                act(scores);
                attackMove(best__no_act_dir);
                if(canAct){
                    scores =CombatTileScore.action_score(myLoc);
                    act(scores);
                }
            }
            else{
                attackMove(best_act_dir);
                scores =CombatTileScore.action_score(myLoc);
                act(scores);
            }
        }
        else{
            attackMove(best__no_act_dir);
        }

        }

    /**
 * Moves toward the closest enemy, turning to face them.
 * If no enemy is within distance sqrt(3) of where we're about to land,
 * faces the move direction instead of the enemy -- otherwise keeps
 * facing the enemy regardless of how we're moving.
 */
public static void attackMove(Direction idealDir) throws GameActionException {

    RobotInfo closest = null;
    int closestDistSq = Integer.MAX_VALUE;
    for (RobotInfo enemy : enemyRats) {
        int distSq = myLoc.distanceSquaredTo(enemy.getLocation());
        if (distSq < closestDistSq) {
            closestDistSq = distSq;
            closest = enemy;
        }
    }
    if (closest == null) return;


    // Figure out which direction we can actually move in, falling back to
    // a 45-degree rotation off the ideal direction if it's blocked.
    Direction moveDir = null;
    if (rc.canMove(idealDir)) {
        moveDir = idealDir;
    } else {
        Direction left = idealDir.rotateLeft();
        Direction right = idealDir.rotateRight();
        if (rc.canMove(left)) {
            moveDir = left;
        } else if (rc.canMove(right)) {
            moveDir = right;
        }
    }

    if (moveDir != null && rc.canMove(moveDir)) {
        rc.move(moveDir);
        update();
    }
     Direction turnDir = myLoc.directionTo(closest.getLocation());
     // Turn after moving.
    if (rc.canTurn(turnDir)) {
        rc.turn(turnDir);
        selfDir=rc.getDirection();
    }
}

    // scores is a list of 4 elements: attacking, laying a mine, throwing, and carrying
    public static void act(int[] scores) throws GameActionException{
        // Try actions in score order (ties broken carry > throw > attack >
        // trap, matching the old priority). Each action method already
        // guards its own legality, so when the top scorer can't actually
        // fire this turn (e.g. trap blocked by insufficient cheese, or a
        // tile canPlaceRatTrap rejects) we now fall through to the
        // next-best action instead of a fixed carry->throw->attack order
        // that used to run unconditionally afterward and could silently
        // execute a far worse-scoring action (e.g. a low-value carry)
        // ahead of a much better one (e.g. a high-value attack) that
        // never got its turn.
        int[] order = {3, 2, 0, 1}; // carry, throw, attack, trap
        for (int i = 1; i < order.length; i++) {
            int key = order[i];
            int keyScore = scores[key];
            int j = i - 1;
            while (j >= 0 && scores[order[j]] < keyScore) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        for (int i = 0; i < order.length && turnActions.isEmpty(); i++) {
            // A negative score means the action's own valuation judged it a
            // bad trade (e.g. attackValue() found retaliation risk exceeds
            // the damage dealt) -- once we've fallen through this far,
            // taking it anyway is worse than doing nothing, so stop instead
            // of using it as a last-resort fallback.
            if (scores[order[i]] < 0) break;
            switch (order[i]) {
                case 3: carryBestTarget(); break;
                case 2: throwAtBestTarget(); break;
                case 0: attackHighestHealthInRange(); break;
                case 1: if (scores[1] > 0) placeTrapTowardClosestEnemy(); break;
            }
        }
    }

    /**
 * Ratnaps the highest-HP enemy rat we can legally carry. rc.canCarryRat
 * already covers adjacency, the facing-away-or-lower-health eligibility
 * rule, the same-robot recent-carry cooldown, and not already carrying
 * someone, so it's the authoritative check here rather than the distance
 * heuristics CombatTileScore uses for scoring purposes.
 */
    public static void carryBestTarget() throws GameActionException {
        if (!canAct || rc.getCarrying() != null) return;

        RobotInfo target = null;
        int bestHealth = -1;

        for (RobotInfo enemy : enemyRats) {
            if (enemy.getType() != UnitType.BABY_RAT) continue;
            if (!rc.canCarryRat(enemy.getLocation())) continue;

            if (enemy.getHealth() > bestHealth) {
                bestHealth = enemy.getHealth();
                target = enemy;
            }
        }

        if (target != null) {
            rc.carryRat(target.getLocation());
            canAct=false;
            turnActions += (turnActions.isEmpty() ? "" : ",") + "carry";
        }
    }

    /**
 * Places a rat trap on the tile adjacent to us, in the direction of the
 * closest enemy rat.
 */
public static void placeTrapTowardClosestEnemy() throws GameActionException {
    if (rc.getAllCheese()<300) return;
    RobotInfo closest = null;
    int closestDistSq = Integer.MAX_VALUE;

    for (RobotInfo enemy : enemyRats) {
        int distSq = myLoc.distanceSquaredTo(enemy.getLocation());
        if (distSq < closestDistSq) {
            closestDistSq = distSq;
            closest = enemy;
        }
    }

    if (closest == null) return;

    Direction dir = myLoc.directionTo(closest.getLocation());
    MapLocation trapLoc = myLoc.add(dir);

    if (rc.canPlaceRatTrap(trapLoc)) {
        rc.placeRatTrap(trapLoc);
        turnActions += (turnActions.isEmpty() ? "" : ",") + "trap";
    }
}

    /**
 * Attacks the highest-HP enemy rat within distance 3 of us, if any.
 */
public static void attackHighestHealthInRange() throws GameActionException {
    if (!canAct) return;

    RobotInfo enemyKing = null;
    for (RobotInfo e : enemyRats) {
            if (e.getType() == UnitType.RAT_KING) {
                enemyKing = e;
                break;
            }
        }
    if (enemyKing != null) {
        MapLocation atk_loc = enemyKing.getLocation().subtract(myLoc.directionTo(enemyKing.getLocation()));
        if(rc.canAttack(atk_loc)){
            if (rc.canTurn()) {
                Direction face = myLoc.directionTo(enemyKing.getLocation());
                if (face != Direction.CENTER && rc.getDirection() != face) rc.turn(face);
            }
            rc.attack(atk_loc);
            canAct=false;
            turnActions += (turnActions.isEmpty() ? "" : ",") + "attack";
            turnDamage += GameConstants.RAT_BITE_DAMAGE;
            return;
        }
    }

    final int MAX_DIST_SQ = 3;

    RobotInfo target = null;
    int bestHealth = -1;

    for (RobotInfo enemy : enemyRats) {
        int distSq = myLoc.distanceSquaredTo(enemy.getLocation());
        if (distSq > MAX_DIST_SQ) continue;

        if (enemy.getHealth() > bestHealth) {
            bestHealth = enemy.getHealth();
            target = enemy;
        }
    }

    if (target != null && rc.canAttack(target.getLocation())) {
        // TODO determine how mcuh to attack with instead of default
        rc.attack(target.getLocation());
        canAct=false;
        turnActions += (turnActions.isEmpty() ? "" : ",") + "attack";
        turnDamage += GameConstants.RAT_BITE_DAMAGE;
    }
}

    /**
 * Throws whatever rat we're currently carrying at the best enemy target:
 * normally the closest non adjacent enemy, but if 2+ enemies are within
 * distance sqrt([4, 8]), throws at the highest-HP one among those instead.
 */
    public static void throwAtBestTarget() throws GameActionException {
        if (rc.getCarrying() == null || !canAct) return;

        final int MIN_DIST_SQ = 3;
        final int RANGE_LO_SQ = 4;
        final int RANGE_HI_SQ = 8;

        RobotInfo closest = null;
        int closestDistSq = Integer.MAX_VALUE;

        RobotInfo bestInRange = null;
        int bestInRangeHealth = -1;
        int inRangeCount = 0;

        for (RobotInfo enemy : enemyRats) {
            int distSq = myLoc.distanceSquaredTo(enemy.getLocation());
            if (distSq < MIN_DIST_SQ) continue;

            if (distSq < closestDistSq) {
                closestDistSq = distSq;
                closest = enemy;
            }

            if (distSq >= RANGE_LO_SQ && distSq <= RANGE_HI_SQ) {
                inRangeCount++;
                if (enemy.getHealth() > bestInRangeHealth) {
                    bestInRangeHealth = enemy.getHealth();
                    bestInRange = enemy;
                }
            }
        }

        RobotInfo target = (inRangeCount >= 2) ? bestInRange : closest;
        if (target == null) return;

        Direction dir = myLoc.directionTo(target.getLocation());
        if (rc.canTurn(dir)) {
            rc.turn(dir);
            if (rc.canThrowRat()) {
                rc.throwRat();
                canAct=false;
                turnActions += (turnActions.isEmpty() ? "" : ",") + "throw";
            }
        }
        else if (selfDir==dir){
            if (rc.canThrowRat()) {
                rc.throwRat();
                canAct=false;
                turnActions += (turnActions.isEmpty() ? "" : ",") + "throw";
            }
        }
    }
}
