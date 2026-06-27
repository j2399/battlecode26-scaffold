package econ5;

import battlecode.common.*;

public class TileScore {

    /* ===============================
       CONSTANTS
       =============================== */

    /*private static final int ADJACENT_DIST_SQ = 1;

    private static final int HP_ADVANTAGE_BONUS = 200;

    private static final int ADJACENT_PENALTY = 100;

    private static final int FACING_AWAY_BONUS = 200;
    private static final int FACING_AWAY_THRESHOLD = 1;
    static final int FRIEND_SUPPORT_BONUS = 150; // tune*/

    private static final int ADJACENT_DIST_SQ = 1;

    private static final int HP_ADVANTAGE_BONUS = 150;

    private static final int ADJACENT_PENALTY = 100;

    private static final int FACING_AWAY_BONUS = 250;
    private static final int FACING_AWAY_THRESHOLD = 1;
    static final int FRIEND_SUPPORT_BONUS = 450; // tune

    /* ===============================
       HELPERS
       =============================== */

    private static boolean facingAway(Direction toMe, Direction enemyDir) {
        return Math.abs(
                toMe.getDirectionOrderNum() - enemyDir.getDirectionOrderNum()
        ) > FACING_AWAY_THRESHOLD;
    }

    /* ===============================
       SCORE FUNCTION
       =============================== */

    public static int score(
            RobotController rc,
            RobotInfo[] enemyRats,
            RobotInfo[] friendlyRats,
            MapLocation tile, MapLocation thrown,
            MapLocation mine) throws GameActionException {

        int score = 0;
        boolean carrying=rc.getCarrying()!=null;
        MapLocation myLoc = rc.getLocation();

        if(thrown!=null) {
            int x = myLoc.x-thrown.x;
            int y =myLoc.y-thrown.y;
            if (x==0 || y==0){
                score-=100;
            }
            else if (Math.abs(x)==Math.abs(y)){
                score-=100;
            }
        }

        if(mine!=null) {
            int dist = tile.distanceSquaredTo(mine);
            score += 50 * (dist==0 ? -3 : dist < 3 ? 3 : dist < 9 ? 2 : dist < 16 ? 1 : 0);
        }

        RobotInfo closestEnemy = null;
        RobotInfo secondEnemy = null;
        int closestDist = Integer.MAX_VALUE;
        int secondDist = Integer.MAX_VALUE;

        for (RobotInfo e : enemyRats) {
            int d = tile.distanceSquaredTo(e.getLocation());
            if(carrying && e.ID==rc.getCarrying().ID){ continue;}

            if (d < closestDist) {
                // demote current closest to second
                secondDist = closestDist;
                secondEnemy = closestEnemy;

                closestDist = d;
                closestEnemy = e;
            } else if (d < secondDist) {
                secondDist = d;
                secondEnemy = e;
            }
        }

        if (closestEnemy == null) return 0;


        /* ===============================
           ADJACENCY PENALTY
           =============================== */

        Direction dir_to_closest = tile.directionTo(closestEnemy.location);

        if (closestDist == ADJACENT_DIST_SQ) {
            score -= ADJACENT_PENALTY;


            MapLocation myleft = tile.add(dir_to_closest.rotateLeft().rotateLeft());
            MapLocation myright = tile.add(dir_to_closest.rotateRight().rotateRight());

            if (rc.canSenseLocation(myleft)) {
                if (!rc.sensePassability(myleft)) {
                    for (RobotInfo fr : friendlyRats) {
                        if (fr == null) continue;

                        // Ignore carried/self just in case
                        if (fr.getID() == rc.getID()) continue;

                        if (myleft.distanceSquaredTo(fr.getLocation()) <= 2) {
                            score += FRIEND_SUPPORT_BONUS;
                            break; // only count once
                        }
                    }
                }
            } else {
                for (RobotInfo fr : friendlyRats) {
                    if (fr == null) continue;

                    // Ignore carried/self just in case
                    if (fr.getID() == rc.getID()) continue;

                    if (myleft.distanceSquaredTo(fr.getLocation()) <= 2) {
                        score += FRIEND_SUPPORT_BONUS;
                        break; // only count once
                    }
                }
            }

            /* ===============================
       LINE / FAN-OUT FORMATION
       =============================== */
            Direction toEnemy = tile.directionTo(closestEnemy.getLocation());

            for (RobotInfo fr : friendlyRats) {
                if (fr == null || fr.getID() == rc.getID()) continue;

                int d = tile.distanceSquaredTo(fr.getLocation());
                if (d <= 1 || d > 14) continue;

                Direction toAlly = tile.directionTo(fr.getLocation());

                if (toEnemy != Direction.CENTER &&
                        (toAlly == toEnemy.rotateLeft()
                                || toAlly == toEnemy.rotateRight()
                                || toAlly == toEnemy.rotateLeft().rotateLeft()
                                || toAlly == toEnemy.rotateRight().rotateRight())
                                || toAlly == toEnemy.rotateLeft().rotateLeft().rotateLeft()
                                 || toAlly == toEnemy.rotateRight().rotateRight().rotateRight()) {

                    score += 180; // lateral spacing bonus
                }

                if (d <= 3) {
                    score -= 120; // discourage blobs
                }
            }


            if (rc.canSenseLocation(myright)) {
                if (!rc.sensePassability(myright)) {
                    for (RobotInfo fr : friendlyRats) {
                        if (fr == null) continue;

                        // Ignore carried/self just in case
                        if (fr.getID() == rc.getID()) continue;

                        if (myright.distanceSquaredTo(fr.getLocation()) <= 2) {
                            score += FRIEND_SUPPORT_BONUS;
                            break; // only count once
                        }
                    }
                }
            } else {
                for (RobotInfo fr : friendlyRats) {
                    if (fr == null) continue;

                    // Ignore carried/self just in case
                    if (fr.getID() == rc.getID()) continue;

                    if (myright.distanceSquaredTo(fr.getLocation()) <= 2) {
                        score += FRIEND_SUPPORT_BONUS;
                        break; // only count once
                    }
                }
            }

        }


        // range score
        if ((closestDist > 8 && (closestDist < 15 || closestDist == 18) && (enemyRats.length+1 >= friendlyRats.length || carrying))) {
            score += 350;
        } else if (enemyRats.length+1 <= friendlyRats.length ) {
            score += 351 - closestDist + (closestDist < 3 ? 40 : 0) + (closestDist < 9 ? 40 : 0);
        } else {
            score += (closestDist < 3 ? 40 : 0) + (closestDist < 9 ? 40 : 0) - closestDist;
        }

        /* ===============================
         SECOND CLOSEST ENEMY PENALTY
        =============================== */
        if (secondEnemy != null) {
            // Base penalty: closer second enemy => larger penalty
            score -= secondDist + (secondDist < 3 ? 100 : 0) + (secondDist < 9 ? 50 : 0);
        }

        /* ===============================
        CLOSEST ALLY PROXIMITY BONUS
        =============================== */
        int closestAllyDist = Integer.MAX_VALUE;

        for (RobotInfo fr : friendlyRats) {
            if (fr == null) continue;
            if (fr.getID() == rc.getID()) continue; // ignore self
            int d = tile.distanceSquaredTo(fr.getLocation());
            if (d < closestAllyDist) closestAllyDist = d;
        }

        if (closestAllyDist != Integer.MAX_VALUE) {
            // Closer ally => bigger bonus (cap it so it doesn't dominate)
            score += 100 - closestAllyDist * 2;
        }



        /* ===============================
           HEALTH DIFFERENCE
           =============================== */

        int myHP = rc.getHealth();
        int enemyHP = closestEnemy.getHealth();

        if (enemyHP < myHP && !carrying) {
            score += HP_ADVANTAGE_BONUS*((closestDist<3? 3 : (closestDist < 9 ? 2 : 1)));
        } else {
            Direction enemyToMe = closestEnemy.getLocation().directionTo(tile);
            if (!carrying && facingAway(enemyToMe, closestEnemy.getDirection())) {
                score += FACING_AWAY_BONUS*((closestDist<3? 3 : (closestDist < 9 ? 2 : 1)));;
            }
            else  if (enemyHP > myHP || carrying){
                score -= 350*((closestDist<3? 3 : (closestDist < 9 ? 2 : 1)));
            }
        }

        if (secondEnemy!=null){
         enemyHP = secondEnemy.getHealth();

        if (!carrying && enemyHP < myHP) {
            score += HP_ADVANTAGE_BONUS*((secondDist<3? 3 : (secondDist < 9 ? 2 : 1)));
        } else {
            Direction enemyToMe = secondEnemy.getLocation().directionTo(tile);
            if (!carrying && facingAway(enemyToMe, secondEnemy.getDirection())) {
                score += FACING_AWAY_BONUS*((secondDist<3? 3 : (secondDist < 8 ? 2 : 1)));;
            }
            else  if (carrying || enemyHP > myHP){
                score -= 350*((secondDist<3? 3 : (secondDist < 9 ? 2 : 1)));
            }
        }
    }


        return score;
    }
}
