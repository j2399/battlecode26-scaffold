package prev_weighted_micro;

import battlecode.common.*;

/** Active when an attackable enemy rat is in range. */
public class CombatState extends Unit {

    // TODO implement combat state 

    public static void run() throws GameActionException {
        // act after moving
        int best_act_score=Integer.MIN_VALUE;
        Direction best_act_dir=Direction.NORTH;
        // no act after moving
        int best_no_act_score=Integer.MIN_VALUE;
        Direction best__no_act_dir=Direction.NORTH;

        for(Direction dir : adjacentDirections){
                MapLocation loc=myLoc.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.tile_score(loc, false);
                    if (score>best_no_act_score){
                        best_no_act_score=score;
                        best__no_act_dir=dir;
                    }
                }
            }

        if (canAct){
             for(Direction dir : adjacentDirections){
                MapLocation loc=myLoc.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.tile_score(loc, true);
                    if (score>best_act_score){
                        best_act_score=score;
                        best_act_dir=dir;
                    }
                }
            }
            int[] scores =CombatTileScore.action_score(myLoc);
            int act_score=Math.max(scores[0],Math.max(scores[1],scores[2]));
            if(best_no_act_score+act_score>best_act_score){
                act(scores);
                attackMove(best__no_act_dir);
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

    Direction turnDir = myLoc.directionTo(closest.getLocation());
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

    if (moveDir != null) {
        MapLocation destination = myLoc.add(moveDir);
        boolean enemyNearDestination = false;
        for (RobotInfo enemy : enemyRats) {
            if (destination.distanceSquaredTo(enemy.getLocation()) <= 3) {
                enemyNearDestination = true;
                break;
            }
        }
        if (!enemyNearDestination) {
            turnDir = moveDir;
        }
    }

    // Turn before moving.
    if (rc.canTurn(turnDir)) {
        rc.turn(turnDir);
        selfDir=rc.getDirection();
    }
    if (moveDir != null && rc.canMove(moveDir)) {
        rc.move(moveDir);
        update();
    }
}

    // scores is a list of 3 elements: attacking, laying a mine, and throwing
    public static void act(int[] scores) throws GameActionException{
        int attack_score=scores[0];
        int mine_score=scores[1];
        int throw_score=scores[2];

        if (throw_score>=attack_score && throw_score>=mine_score){
            throwAtBestTarget();
        }
        else if (attack_score>=mine_score){
             attackHighestHealthInRange();
        }
        else{
            placeTrapTowardClosestEnemy();
        }
    }

    /**
 * Places a rat trap on the tile adjacent to us, in the direction of the
 * closest enemy rat.
 */
public static void placeTrapTowardClosestEnemy() throws GameActionException {
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
    }
}

    /**
 * Attacks the highest-HP enemy rat within distance 3 of us, if any.
 */
public static void attackHighestHealthInRange() throws GameActionException {
    if (!canAct) return;
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
            }
        }
        else if (selfDir==dir){
            if (rc.canThrowRat()) {
                rc.throwRat();
                canAct=false;
            }
        }
    }
}
