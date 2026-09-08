package weightedMicro;

import battlecode.common.*;

/** Active when an attackable enemy rat is in range. */
public class CombatState extends Unit {

    // TODO implement combat state 

    public static void run() throws GameActionException {
        // act after moving
        int actScoreBest=Integer.MIN_VALUE;
        Direction actDirBest=Direction.NORTH;
        // no act after moving
        int noActScoreBest=Integer.MIN_VALUE;
        Direction noActDirBest=Direction.NORTH;

        for(Direction dir : directionsAdjacent){
                MapLocation loc=locMy.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.scoreTile(loc, false);
                    if (score>noActScoreBest){
                        noActScoreBest=score;
                        noActDirBest=dir;
                    }
                }
            }

        if (actCan){
             for(Direction dir : directionsAdjacent){
                MapLocation loc=locMy.add(dir);
                if(rc.canMove(dir)){
                    int score=CombatTileScore.scoreTile(loc, true);
                    if (score>actScoreBest){
                        actScoreBest=score;
                        actDirBest=dir;
                    }
                }
            }
            int[] scores =CombatTileScore.scoreAction(locMy);
            int scoreAct=Math.max(Math.max(scores[0],scores[1]),Math.max(scores[2],scores[3]));
            if(noActScoreBest+scoreAct>actScoreBest){
                act(scores);
                moveAttack(noActDirBest);
                if(actCan){
                    scores =CombatTileScore.scoreAction(locMy);
                    act(scores);
                }
            }
            else{
                moveAttack(actDirBest);
                scores =CombatTileScore.scoreAction(locMy);
                act(scores);
            }
        }
        else{
            moveAttack(noActDirBest);
        }
    }

    /**
 * Moves toward the closest enemy, turning to face them.
 * If no enemy is within distance sqrt(3) of where we're about to land,
 * faces the move direction instead of the enemy -- otherwise keeps
 * facing the enemy regardless of how we're moving.
 */
public static void moveAttack(Direction dirIdeal) throws GameActionException {

    RobotInfo closest = null;
    int distSqClosest = Integer.MAX_VALUE;
    for (RobotInfo enemy : ratsEnemy) {
        int sqDist = locMy.distanceSquaredTo(enemy.getLocation());
        if (sqDist < distSqClosest) {
            distSqClosest = sqDist;
            closest = enemy;
        }
    }
    if (closest == null) return;


    // Figure out which direction we can actually move in, falling back to
    // a 45-degree rotation off the ideal direction if it's blocked.
    Direction dirMove = null;
    if (rc.canMove(dirIdeal)) {
        dirMove = dirIdeal;
    } else {
        Direction left = dirIdeal.rotateLeft();
        Direction right = dirIdeal.rotateRight();
        if (rc.canMove(left)) {
            dirMove = left;
        } else if (rc.canMove(right)) {
            dirMove = right;
        }
    }

    if (dirMove != null && rc.canMove(dirMove)) {
        rc.move(dirMove);
        update();
    }
     Direction dirTurn = locMy.directionTo(closest.getLocation());
     // Turn after moving.
    if (rc.canTurn(dirTurn)) {
        rc.turn(dirTurn);
        dirMy=rc.getDirection();
    }
}

    // scores is a list of 4 elements: attacking, laying a mine, throwing, and carrying
    public static void act(int[] scores) throws GameActionException{
        int scoreAttack=scores[0];
        int scoreMine=scores[1];
        int scoreThrow=scores[2];
        int scoreCarry=scores[3];

        if (scoreCarry>=scoreAttack && scoreCarry>=scoreMine && scoreCarry>=scoreThrow){
            bestTargetCarry();
        }
        else if (scoreThrow>=scoreAttack && scoreThrow>=scoreMine){
            atBestTargetThrow();
        }
        else if (scoreAttack>=scoreMine){
             highestHealthInRangeAttack();
        }
        else{
            trapTowardClosestEnemyPlace();
        }
        bestTargetCarry();
        atBestTargetThrow();
        highestHealthInRangeAttack();
    }

    /**
 * Ratnaps the highest-HP enemy rat we can legally carry. rc.canCarryRat
 * already covers adjacency, the facing-away-or-lower-health eligibility
 * rule, the same-robot recent-carry cooldown, and not already carrying
 * someone, so it's the authoritative check here rather than the distance
 * heuristics CombatTileScore uses for scoring purposes.
 */
    public static void bestTargetCarry() throws GameActionException {
        if (!actCan || rc.getCarrying() != null) return;

        RobotInfo target = null;
        int healthBest = -1;

        for (RobotInfo enemy : ratsEnemy) {
            if (enemy.getType() != UnitType.BABY_RAT) continue;
            if (!rc.canCarryRat(enemy.getLocation())) continue;

            if (enemy.getHealth() > healthBest) {
                healthBest = enemy.getHealth();
                target = enemy;
            }
        }

        if (target != null) {
            rc.carryRat(target.getLocation());
            actCan=false;
        }
    }

    /**
 * Places a rat trap on the tile adjacent to us, in the direction of the
 * closest enemy rat.
 */
public static void trapTowardClosestEnemyPlace() throws GameActionException {
    if (rc.getAllCheese()<300) return;
    RobotInfo closest = null;
    int distSqClosest = Integer.MAX_VALUE;

    for (RobotInfo enemy : ratsEnemy) {
        int sqDist = locMy.distanceSquaredTo(enemy.getLocation());
        if (sqDist < distSqClosest) {
            distSqClosest = sqDist;
            closest = enemy;
        }
    }

    if (closest == null) return;

    Direction dir = locMy.directionTo(closest.getLocation());
    MapLocation locTrap = locMy.add(dir);

    if (rc.canPlaceRatTrap(locTrap)) {
        rc.placeRatTrap(locTrap);
    }
}

    /**
 * Attacks the highest-HP enemy rat within distance 3 of us, if any.
 */
public static void highestHealthInRangeAttack() throws GameActionException {
    if (!actCan) return;

    RobotInfo kingEnemy = null;
    for (RobotInfo e : ratsEnemy) {
            if (e.getType() == UnitType.RAT_KING) {
                kingEnemy = e;
                break;
            }
        }
    if (kingEnemy != null) {
        MapLocation locAtk = kingEnemy.getLocation().subtract(locMy.directionTo(kingEnemy.getLocation()));
        if(rc.canAttack(locAtk)){
            if (rc.canTurn()) {
                Direction face = locMy.directionTo(kingEnemy.getLocation());
                if (face != Direction.CENTER && rc.getDirection() != face) rc.turn(face);
            }
            rc.attack(locAtk);
            actCan=false;
            return;
        }
    }

    final int distSqMax = 3;

    RobotInfo target = null;
    int healthBest = -1;

    for (RobotInfo enemy : ratsEnemy) {
        int sqDist = locMy.distanceSquaredTo(enemy.getLocation());
        if (sqDist > distSqMax) continue;

        if (enemy.getHealth() > healthBest) {
            healthBest = enemy.getHealth();
            target = enemy;
        }
    }

    if (target != null && rc.canAttack(target.getLocation())) {
        // TODO determine how mcuh to attack with instead of default
        rc.attack(target.getLocation());
        actCan=false;
    }
}

    /**
 * Throws whatever rat we're currently carrying at the best enemy target:
 * normally the closest non adjacent enemy, but if 2+ enemies are within
 * distance sqrt([4, 8]), throws at the highest-HP one among those instead.
 */
    public static void atBestTargetThrow() throws GameActionException {
        if (rc.getCarrying() == null || !actCan) return;

        final int distSqMin = 3;
        final int loSqRange = 4;
        final int hiSqRange = 8;

        RobotInfo closest = null;
        int distSqClosest = Integer.MAX_VALUE;

        RobotInfo inRangeBest = null;
        int inRangeHealthBest = -1;
        int rangeCountIn = 0;

        for (RobotInfo enemy : ratsEnemy) {
            int sqDist = locMy.distanceSquaredTo(enemy.getLocation());
            if (sqDist < distSqMin) continue;

            if (sqDist < distSqClosest) {
                distSqClosest = sqDist;
                closest = enemy;
            }

            if (sqDist >= loSqRange && sqDist <= hiSqRange) {
                rangeCountIn++;
                if (enemy.getHealth() > inRangeHealthBest) {
                    inRangeHealthBest = enemy.getHealth();
                    inRangeBest = enemy;
                }
            }
        }

        RobotInfo target = (rangeCountIn >= 2) ? inRangeBest : closest;
        if (target == null) return;

        Direction dir = locMy.directionTo(target.getLocation());
        if (rc.canTurn(dir)) {
            rc.turn(dir);
            if (rc.canThrowRat()) {
                rc.throwRat();
                actCan=false;
            }
        }
        else if (dirMy==dir){
            if (rc.canThrowRat()) {
                rc.throwRat();
                actCan=false;
            }
        }
    }
}
