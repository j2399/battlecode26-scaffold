package weightedMicro;

import battlecode.common.*;

/**
 * Heuristic scoring for CombatState: how good is a candidate tile to move
 * to, and how good is each combat action (attack / rat trap / throw) right
 * now. Values are expressed on an "HP of damage" scale -- a plain 10-damage
 * bite is worth 10 -- so every bonus/penalty below is picked as a multiple
 * or fraction of that base unit rather than as an arbitrary number.
 */
public class CombatTileScore extends Unit {

    // ---- constants, all in HP-equivalent "value" units ----

    /** Extra value for a bite that kills outright, on top of the raw
     *  damage: a kill permanently removes a threat and drops their cheese,
     *  so it's worth more than the last few HP of damage alone suggest. */
    private static final int bonusKill = 30;

    /** Bonus for attacking an enemy that is currently carrying one of our
     *  allies -- neutralizing whoever is holding a teammate captive. */
    private static final int targetBonusRescue = 15;

    /** Flat discount for an action that would trigger a backstab while
     *  still in cooperation, since it forfeits coop-weighted scoring.
     *  Trap placement gets a smaller discount than attacking because
     *  triggering the backstab is delayed and uncertain -- it only happens
     *  if and when the enemy actually steps on it -- whereas a bite starts
     *  the backstab immediately and for sure. */
    private static final int attackDiscountCoop = 15;
    private static final int trapDiscountCoop = 8;

    /** Below this HP, retaliation risk is weighted more heavily -- losing
     *  HP matters more the less of it we have left. */
    private static final int healthThresholdLow = 30;

    /** Below this much combined cheese, discount a trap's 20-cheese cost --
     *  it's competing with spawning and the king's upkeep. */
    private static final int scarceThresholdCheese = 60;
    private static final int scarceTrapDiscountCheese = 10;

    /** Ongoing mobility tax for as long as we're carrying anyone: cooldowns
     *  get multiplied by GameConstants-level CARRY_COOLDOWN_MULTIPLIER
     *  (1.5x) while carrying, on top of whatever else is slowing us down. */
    private static final int mobilityTaxCarry = 8;

    /** Flat value for taking an enemy out of play while we carry them: they
     *  can't move, attack, trap, or dig -- only squeak and sense -- for as
     *  long as we hold them (or until MAX_CARRY_DURATION forces a drop).
     *  Not a kill, so valued well below one, but still a real tempo swing. */
    private static final int carryBaseValueEnemy = 25;

    /** Ratnapping an enemy that is itself currently carrying one of our
     *  allies forces them to immediately drop that ally -- a guaranteed,
     *  damage-free rescue, worth more than the plain denial value alone. */
    private static final int disruptBonusCaptor = 20;

    /** Same reasoning as attackDiscountCoop: ratnapping an enemy rat is
     *  one of the three explicit backstab triggers and happens immediately
     *  and for sure, unlike a trap sitting dormant until it's triggered. */
    private static final int carryDiscountCoop = 15;

    public static int scoreTile(MapLocation locTile, boolean actCan) throws GameActionException {
        int score = valuePositional(locTile);

        if (actCan) {
            // Best action we could take if we ended up at this tile, as a
            // bonus on top of plain positioning -- ties the two functions
            // together instead of scoring movement blind to combat value.
            int attack = valueAttack(locTile);
            int trap = valueTrap(locTile);
            int valThrow = valueThrow(locTile);
            int carry = valueCarry(locTile);
            score += Math.max(Math.max(attack, trap), Math.max(valThrow, carry));
        } else {
            // We won't be able to act once we're there, so there's no
            // offensive upside this turn to offset any risk -- lean on
            // safety/setup considerations instead of combat value.
            score += positionalValueCautious(locTile);
        }

        return score;
    }

    // returns score of attacking, laying a rat trap, throwing, and carrying
    public static int[] scoreAction(MapLocation locBot) throws GameActionException {
        return new int[]{
            valueAttack(locBot),
            valueTrap(locBot),
            valueThrow(locBot),
            valueCarry(locBot)
        };
    }

    // ---------------------------------------------------------------
    // Positional / movement value
    // ---------------------------------------------------------------

    /** Safety/positioning value of standing at loc, independent of whether
     *  we can act there this turn. */
    private static int valuePositional(MapLocation loc) throws GameActionException {
        int score = 0;

        RobotInfo enemyNearest = nearestFind(ratsEnemy, loc);
        if (enemyNearest != null) {
            int nowDist = locMy.distanceSquaredTo(enemyNearest.getLocation());
            int thenDist = loc.distanceSquaredTo(enemyNearest.getLocation());
            int delta = thenDist - nowDist; // positive = this tile is farther away

            // Advance toward the fight when healthy, retreat when hurt.
            // Halved so positioning nudges the choice without swamping the
            // actual combat-action values folded in above (10-40 range).
            boolean healthy = rc.getHealth() > UnitType.BABY_RAT.getHealth() / 2;
            score += (healthy ? -delta : delta) / 2;

            // We can never see enemy rat traps. A tile right next to a
            // visible enemy is exactly where a rational opponent would
            // have pre-placed one, so treat close proximity as a small
            // unseen-trap risk on top of the plain distance preference.
            if (loc.distanceSquaredTo(enemyNearest.getLocation()) <= TrapType.RAT_TRAP.triggerRadiusSquared) {
                score -= 5;
            }
        }

        // Modest bonus for staying near allies -- support in a fight.
        // Capped so a big blob of allies doesn't dominate every other term.
        int nearAllies = 0;
        for (RobotInfo ally : ratsAlly) {
            if (loc.distanceSquaredTo(ally.getLocation()) <= 8) nearAllies++;
        }
        score += Math.min(nearAllies, 3) * 3;

        return score;
    }

    /**
     * Extra safety/setup value for a tile we're moving to but can't act
     * from -- since there's no combat value this turn to offset risk, this
     * leans on things that matter for surviving to (and being ready for)
     * a future turn instead: staying unseen, not letting several enemies
     * converge unopposed, keeping room to keep retreating, and lining up
     * a blind-spot position for a ratnap later once our cooldown clears.
     */
    private static int positionalValueCautious(MapLocation loc) throws GameActionException {
        int score = 0;

        // Being seen while we can't act is pure downside -- we can't
        // capitalize on the exposure, so it only invites a free look from
        // whoever's watching. Being unseen costs nothing this turn and
        // preserves options for later.
        int toCountVisible = 0;
        for (RobotInfo enemy : ratsEnemy) {
            if (toEnemyVisible(enemy, loc)) toCountVisible++;
        }
        score -= toCountVisible * 8;

        // One of the three ratnap conditions is the target facing away
        // from us. Landing in the nearest enemy's blind spot doesn't help
        // this turn (we can't act), but it sets up a possible ratnap next
        // turn once our action cooldown clears.
        RobotInfo enemyNearest = nearestFind(ratsEnemy, loc);
        if (enemyNearest != null && !toEnemyVisible(enemyNearest, loc)) {
            score += 4;
        }

        // Aggregate danger from every enemy that could plausibly reach us
        // over the next turn or two, not just the single closest one --
        // with no action available this turn, several converging threats
        // compound risk even if none of them can reach us just yet.
        final int radiusSqThreat = 18; // roughly "a couple of moves away"
        int threats = 0;
        for (RobotInfo enemy : ratsEnemy) {
            if (loc.distanceSquaredTo(enemy.getLocation()) <= radiusSqThreat) threats++;
        }
        score -= threats * 3;

        // Avoid boxing ourselves in behind walls/dirt/other robots -- fewer
        // open neighboring tiles means less room to keep retreating on a
        // future turn. Only worth the extra sensing cost when we already
        // know we can't fight back this turn anyway.
        int tilesOpen = adjacentTilesOpen(loc);
        score += (tilesOpen - 3) * 3;

        return score;
    }

    /** Whether `loc` falls inside `enemy`'s current vision (radius + facing
     *  cone), based on their last-sensed position and direction. */
    private static boolean toEnemyVisible(RobotInfo enemy, MapLocation loc) {
        int sqDist = enemy.getLocation().distanceSquaredTo(loc);
        if (sqDist > enemy.getType().getVisionRadiusSquared()) return false;
        if (enemy.getType().getVisionAngle() >= 360) return true;

        Direction locTo = enemy.getLocation().directionTo(loc);
        Direction facing = enemy.getDirection();
        return locTo == facing || locTo == facing.rotateLeft() || locTo == facing.rotateRight();
    }

    /** Count of the 8 tiles around `loc` that are passable and unoccupied
     *  (or simply unsensed, which we don't penalize since we can't tell). */
    private static int adjacentTilesOpen(MapLocation loc) throws GameActionException {
        int open = 0;
        for (Direction dir : directionsAdjacent) {
            MapLocation adj = loc.add(dir);
            if (!rc.canSenseLocation(adj)) {
                open++;
                continue;
            }
            if (!rc.sensePassability(adj)) continue;
            if (occupiedIs(adj)) continue;
            open++;
        }
        return open;
    }

    private static boolean occupiedIs(MapLocation loc) {
        for (RobotInfo r : ratsEnemy) if (r.getLocation().equals(loc)) return true;
        for (RobotInfo r : ratsAlly) if (r.getLocation().equals(loc)) return true;
        return false;
    }

    // ---------------------------------------------------------------
    // Attack
    // ---------------------------------------------------------------

    private static int valueAttack(MapLocation from) throws GameActionException {
        RobotInfo target = bestAttackTargetFind(from);
        if (target == null) return 0;

        int damage = Math.min(GameConstants.RAT_BITE_DAMAGE, target.getHealth());
        int value = damage;

        boolean lethal = target.getHealth() <= GameConstants.RAT_BITE_DAMAGE;
        if (lethal) value += bonusKill;

        if (target.getCarryingRobot() != null && target.getCarryingRobot().getTeam() == teamMy) {
            value += targetBonusRescue;
        }

        value -= riskRetaliation(from, lethal ? target : null);

        if (rc.isCooperation()) {
            value -= attackDiscountCoop;
        }

        return value;
    }

    /** Best (highest-HP) enemy within bite range of `from`. */
    private static RobotInfo bestAttackTargetFind(MapLocation from) {
        RobotInfo best = null;
        for (RobotInfo enemy : ratsEnemy) {
            if (from.distanceSquaredTo(enemy.getLocation()) > GameConstants.ATTACK_DISTANCE_SQUARED) continue;
            if (best == null || enemy.getHealth() > best.getHealth()) {
                best = enemy;
            }
        }
        return best;
    }

    /**
     * Expected damage we could take back next turn from enemies adjacent to
     * `from`, excluding a target we're about to kill (a dead rat can't bite
     * back). Scaled up when our own HP is low, and scaled up further when
     * we're already stuck in place for another round (stunned from a prior
     * trap or throw landing) -- there's no disengaging either way, so any
     * nearby enemy is a guaranteed exposure next turn, not just a likely one.
     */
    private static int riskRetaliation(MapLocation from, RobotInfo excluding) throws GameActionException {
        int enemiesAdjacent = 0;
        for (RobotInfo enemy : ratsEnemy) {
            if (excluding != null && enemy.getID() == excluding.getID()) continue;
            if (from.distanceSquaredTo(enemy.getLocation()) <= GameConstants.ATTACK_DISTANCE_SQUARED) {
                enemiesAdjacent++;
            }
        }
        if (enemiesAdjacent == 0) return 0;

        int risk = enemiesAdjacent * GameConstants.RAT_BITE_DAMAGE;

        if (rc.getHealth() < healthThresholdLow) {
            risk *= 2;
        }
        if (rc.getMovementCooldownTurns() >= GameConstants.COOLDOWN_LIMIT) {
            risk += GameConstants.RAT_BITE_DAMAGE;
        }
        return risk;
    }

    // ---------------------------------------------------------------
    // Rat trap
    // ---------------------------------------------------------------

    private static int valueTrap(MapLocation from) throws GameActionException {
        if (rc.getNumberRatTraps() >= TrapType.RAT_TRAP.maxCount) return 0;

        RobotInfo closest = nearestFind(ratsEnemy, from);
        if (closest == null) return 0;

        Direction dir = from.directionTo(closest.getLocation());
        MapLocation locTrap = from.add(dir);

        // Rough trigger-probability tiers based on how close the nearest
        // enemy already is to the trap tile -- closer means more likely to
        // wander into the trigger radius soon. Capped well below 100% since
        // we can never be sure they'll actually walk there.
        int toTrapDist = closest.getLocation().distanceSquaredTo(locTrap);
        int percentProbability;
        if (toTrapDist <= TrapType.RAT_TRAP.triggerRadiusSquared) {
            percentProbability = 60;
        } else if (toTrapDist <= 8) {
            percentProbability = 30;
        } else {
            percentProbability = 10;
        }

        // More potential traffic through the area raises the odds something
        // eventually triggers it, even if not this specific enemy.
        int nearbyTraffic = 0;
        for (RobotInfo enemy : ratsEnemy) {
            if (locTrap.distanceSquaredTo(enemy.getLocation()) <= 8) nearbyTraffic++;
        }
        percentProbability = Math.min(80, percentProbability + (nearbyTraffic - 1) * 10);

        // Blind-spot bonus: enemies can never see our traps regardless, but
        // one placed behind their current facing is even less likely to be
        // routed around by a cautious opponent.
        Direction facingEnemy = closest.getDirection();
        Direction trapTo = closest.getLocation().directionTo(locTrap);
        boolean enemyFacingConeIn = (trapTo == facingEnemy
                || trapTo == facingEnemy.rotateLeft() || trapTo == facingEnemy.rotateRight());
        if (!enemyFacingConeIn) percentProbability += 10;

        int value = (TrapType.RAT_TRAP.damage * percentProbability) / 100;

        int cheeseTotal = rc.getRawCheese() + rc.getGlobalCheese();
        if (cheeseTotal < scarceThresholdCheese) {
            value -= scarceTrapDiscountCheese;
        }

        if (rc.isCooperation()) {
            value -= trapDiscountCoop;
        }

        return Math.max(0, value);
    }

    // ---------------------------------------------------------------
    // Throw
    // ---------------------------------------------------------------

    private static int valueThrow(MapLocation from) throws GameActionException {
        RobotInfo carried = rc.getCarrying();
        if (carried == null) return 0;

        if (carried.getTeam() == teamMy) {
            return throwValueRescue(carried);
        }
        return throwValueOffensive(from, carried);
    }

    /** Throwing a captured enemy: valuable mainly if it can collide with a
     *  second enemy along the way for splash damage; otherwise a small
     *  baseline for freeing ourselves up rather than staying tied down
     *  carrying them. */
    private static int throwValueOffensive(MapLocation from, RobotInfo carried) {
        int distSqBest = Integer.MAX_VALUE;
        RobotInfo targetSecond = null;

        for (RobotInfo enemy : ratsEnemy) {
            if (enemy.getID() == carried.getID()) continue;
            int sqDist = from.distanceSquaredTo(enemy.getLocation());
            if (sqDist < distSqBest) {
                distSqBest = sqDist;
                targetSecond = enemy;
            }
        }

        if (targetSecond == null) return 5; // baseline: relieve the carry burden

        int flightTilesMax = GameConstants.TILES_FLOWN_PER_TURN * GameConstants.THROW_DURATION;
        int awayTiles = (int) Math.round(Math.sqrt(distSqBest));
        if (awayTiles > flightTilesMax) return 5; // out of range, no collision possible

        int tilesRemaining = Math.max(0, flightTilesMax - awayTiles);
        int damageCollision = GameConstants.THROW_DAMAGE_PER_TILE * tilesRemaining;
        return Math.min(damageCollision, targetSecond.getHealth());
    }

    /** Throwing a hurt ally toward safety: more urgent the less HP they
     *  have left to lose. */
    private static int throwValueRescue(RobotInfo carried) {
        int healthMissing = UnitType.BABY_RAT.getHealth() - carried.getHealth();
        return healthMissing / 3;
    }

    // ---------------------------------------------------------------
    // Carry (ratnap)
    // ---------------------------------------------------------------

    /**
     * Best value of ratnapping an adjacent rat from `from` -- an ally
     * (always a legal target) or an enemy (legal only if they're facing
     * away from us, or have less health than us, matching the real game
     * rule). Only baby rats can be carried at all -- rat kings can't.
     */
    private static int valueCarry(MapLocation from) throws GameActionException {
        if (rc.getCarrying() != null) return 0; // can only carry one at a time

        int best = 0;

        for (RobotInfo candidate : ratsEnemy) {
            if (candidate.getType() != UnitType.BABY_RAT) continue;
            if (from.distanceSquaredTo(candidate.getLocation()) > GameConstants.ATTACK_DISTANCE_SQUARED) continue;

            boolean awayFacing = !toEnemyVisible(candidate, from);
            boolean healthLower = candidate.getHealth() < rc.getHealth();
            if (!awayFacing && !healthLower) continue; // not a legal ratnap target

            int value = carryValueEnemy(candidate, from);
            if (value > best) best = value;
        }

        for (RobotInfo candidate : ratsAlly) {
            if (candidate.getType() != UnitType.BABY_RAT) continue;
            if (from.distanceSquaredTo(candidate.getLocation()) > GameConstants.ATTACK_DISTANCE_SQUARED) continue;

            int value = carryValueAlly(candidate);
            if (value > best) best = value;
        }

        return best;
    }

    /** Ratnapping an enemy: denies them all action while carried, at the
     *  cost of our own mobility tax and becoming a target for their allies
     *  to try to punish/rescue them from us. */
    private static int carryValueEnemy(RobotInfo target, MapLocation from) throws GameActionException {
        int value = carryBaseValueEnemy;

        if (target.getCarryingRobot() != null && target.getCarryingRobot().getTeam() == teamMy) {
            value += disruptBonusCaptor;
        }

        value -= mobilityTaxCarry;
        value -= riskRetaliation(from, target); // target itself can't retaliate once carried

        if (rc.isCooperation()) {
            value -= carryDiscountCoop;
        }

        return Math.max(0, value);
    }

    /** Ratnapping an ally: carrying moves them at our speed instead of
     *  their own (helpful if they're slowed by a heavy raw-cheese stash),
     *  and gets a hurt ally out of harm's way since they're immune to
     *  attacks while carried. Both benefits can apply to the same rat at
     *  once, so they add rather than take the max of each other. */
    private static int carryValueAlly(RobotInfo target) {
        int valueRescue = (UnitType.BABY_RAT.getHealth() - target.getHealth()) / 3;

        int cheeseRaw = target.getRawCheeseAmount();
        int rushValueCheese = Math.min(cheeseRaw, 100) / 4;

        int value = valueRescue + rushValueCheese - mobilityTaxCarry;
        return Math.max(0, value);
    }

    // ---------------------------------------------------------------
    // Shared helpers
    // ---------------------------------------------------------------

    private static RobotInfo nearestFind(RobotInfo[] robots, MapLocation from) {
        RobotInfo nearest = null;
        int distSqNearest = Integer.MAX_VALUE;
        for (RobotInfo r : robots) {
            int sqDist = from.distanceSquaredTo(r.getLocation());
            if (sqDist < distSqNearest) {
                distSqNearest = sqDist;
                nearest = r;
            }
        }
        return nearest;
    }
}
