package econ5;

import battlecode.common.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class BabyRat extends Globals {

    static final Random rng = new Random(6147);

    public static MapLocation target;

    public static int last_closest_dist;
    public static int turns_since_progress;
    public static MapLocation old_target;

    public static MapLocation explore_target;
    public static MapLocation cur_target;
    public static boolean isnew = true;

    public static int[] mines = new int[100];
    public static int minecount = 0;
    public static int[] global;

    public static RobotInfo[] last_turn_enemy_rats;
    public static RobotInfo[] last_turn_friend_rats;

    // ===== SEEN MAP =====
    static class SeenRobot {
        RobotInfo info;
        int lastSeenRound;

        SeenRobot(RobotInfo info, int round) {
            this.info = info;
            this.lastSeenRound = round;
        }
    }

    static final Map<Integer, SeenRobot> seenRobots = new HashMap<>();
    static final int SEEN_TTL = 50;

    static final Direction[] directions = {
            Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
            Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST
    };

    static enum turns{
        Left1,
        Left2,
        Right1,
        Right2,
    }

    public static turns turn = turns.Left1;

    public static void pickupdirt(RobotController rc) throws GameActionException {
        MapLocation loc = my_loc.add(rc.getDirection());
        if (rc.canRemoveDirt(loc)) {
            if (rc.canTurn()) {
                rc.turn(rc.getDirection());
            }
            rc.removeDirt(loc);
            if(rc.canMove(rc.getDirection())){
                rc.move(rc.getDirection());
                get_info();
            }
            return;
        }
        loc = my_loc.add(rc.getDirection().rotateLeft());
        if (rc.canRemoveDirt(loc)) {
            if (rc.canTurn()) {
                rc.turn(rc.getDirection().rotateLeft());
            }
            rc.removeDirt(loc);
            if(rc.canMove(rc.getDirection().rotateLeft())){
                rc.move(rc.getDirection().rotateLeft());
                get_info();
            }
            return;
        }
        loc = my_loc.add(rc.getDirection().rotateRight());
        if (rc.canRemoveDirt(loc)) {
            if (rc.canTurn()) {
                rc.turn(rc.getDirection().rotateRight());
            }
            rc.removeDirt(loc);
            if(rc.canMove(rc.getDirection().rotateRight())){
                rc.move(rc.getDirection().rotateRight());
                get_info();
            }
            return;
        }
        for (Direction d : directions) {
            MapLocation l = rc.getLocation().add(d);
            if (rc.canRemoveDirt(l)) {
                if (rc.canTurn()) {
                    rc.turn(d);
                }
                rc.removeDirt(l);
                if(rc.canMove(d)){
                    rc.move(d);
                    get_info();
                }
            }
            break;
        }
    }


    public static Direction get_dir(Direction dir) throws GameActionException {
        switch (turn) {
            case Left1:
                turn = turns.Left2;
                return dir.rotateLeft();
            case Left2:
                turn = turns.Right1;
                return dir;
            case Right1:
                turn = turns.Right2;
                return dir.rotateRight();
            case Right2:
                turn = turns.Left1;
                return dir;
        }
        return dir;
    }


    public static boolean turned_left=true;
    public static Direction line;
    public static boolean left_line=true;
    public static MapLocation prev_target;
    public static int tar_dist=Integer.MAX_VALUE;
    public static Direction last_move_dir=Direction.CENTER;
    public static MapLocation old_loc;

    // bugnav to the target turning left and right to get better vision around
    public static void move_and_turn() throws GameActionException {

        Direction dir = WbugNav.moveTo(target);
        //   rc.setIndicatorString("bugnav is: "+dir);
        if (dir== null || dir== Direction.CENTER) {
            return;
        }

        boolean move=true;

        if(rc.canTurn() && state!=State.combat) {
            if((mapWidth+mapHeight>85)) {
                if (rc.canTurn() && rc.getDirection() != dir) {
                    rc.turn(dir);
                    if (rc.canMove(dir)) {
                        rc.move(dir);
                        get_info();
                    }
                } else if (rc.canTurn()) {
                    if (rc.canMove(dir)) {
                        rc.move(dir);
                        get_info();
                    }
                    if (turned_left) {
                        rc.turn(dir.rotateRight().rotateRight());
                        turned_left = false;
                    } else {
                        rc.turn(dir.rotateLeft().rotateLeft());
                        turned_left = true;
                    }
                }
            }
            else{
                Direction new_dir =get_dir(dir);
                if(rc.canMove(new_dir)) {
                    rc.turn(new_dir);
                    if(move) {
                        rc.move(new_dir);
                        get_info();
                    }
                }
                else {
                    if (rc.canTurn() && rc.getDirection() != dir) {
                        rc.turn(dir);
                    }
                    if(move) {
                        if (rc.canMove(dir)) {
                            rc.move(dir);
                            get_info();
                        }
                    }
                }
            }
        }
        else {
            if (rc.canTurn() && rc.getDirection() != dir) {
                rc.turn(dir);
            }
            if(move) {
                if (rc.canMove(dir)) {
                    rc.move(dir);
                    get_info();
                }
            }
        }


    }


    public static RobotInfo[] friendlyRats;
    public static RobotInfo[] enemyRats;
    public static RobotInfo[] cats;
    public static Team my_team=rc.getTeam();
    public static MapLocation king_loc=null;
    public static MapInfo[] mapInfos;
    public static MapLocation my_loc;
    public static MapInfo closest_impassable;

    public static void get_info() throws  GameActionException {
        int bytes=Clock.getBytecodesLeft();
        friendlyRats=rc.senseNearbyRobots(-1, my_team);
        enemyRats=rc.senseNearbyRobots(-1, my_team.opponent());
        cats=rc.senseNearbyRobots(-1,Team.NEUTRAL);
        mapInfos=rc.senseNearbyMapInfos();
        my_loc=rc.getLocation();
        if (rc.getCarrying()!=null){
            carried=rc.getCarrying().ID;
        }

        for(MapInfo info: mapInfos){
            if(closest_impassable==null && info.isWall()){
                closest_impassable=info;
            }
            else if(closest_impassable!=null) {

                if (my_loc.distanceSquaredTo(info.getMapLocation()) < my_loc.distanceSquaredTo(closest_impassable.getMapLocation())) {
                    if (info.isWall()) {
                        closest_impassable = info;
                    }
                }
            }
        }

        king_loc=get_king_loc();
    }


    static int visitedCorners = 0; // bitmask 0b0000–0b1111

    // 0 = BL, 1 = BR, 2 = TL, 3 = TR
    public static int cornerIndex(MapLocation l){
        int cx = (l.x < mapWidth / 2) ? 0 : 1;
        int cy = (l.y < mapHeight / 2) ? 0 : 2;
        return cx + cy;
    }

    public static MapLocation cornerFromIndex(int idx){
        int x = (idx & 1) == 0 ? 3 : mapWidth - 4;
        int y = (idx & 2) == 0 ? 3 : mapHeight - 4;
        return new MapLocation(x, y);
    }

    public static MapLocation been_to_loc;
    public static boolean been_to=false;

    public static void exploring_set_target() throws GameActionException{
        try {
            if (digger && rc.canSenseLocation(my_loc.add(rc.getDirection()))) {
                MapLocation loc = my_loc.add(rc.getDirection()).add(rc.getDirection());
                if (rc.senseMapInfo(loc).isDirt()) {
                    target = loc;
                    target_memory=target;
                    return;
                }
                loc = my_loc.add(rc.getDirection());
                if (rc.senseMapInfo(loc).isDirt()) {
                    target = loc;
                    target_memory=target;
                    return;
                }
            }
        }catch (GameActionException e){}

        if(rc.readSqueaks(-1).length!=0){
            return;
        }

        king_loc=get_king_loc();

        Message[] msgs=rc.readSqueaks(rc.getRoundNum());
        if (msgs.length!=0){
            target=msgs[0].getSource();
            target_memory=target;
            if(last_mine == null || my_loc.isAdjacentTo(last_mine)) {
                last_mine = null;
            }
            return;
        }
        msgs=rc.readSqueaks(rc.getRoundNum()-1);
        if (msgs.length!=0){
            target=msgs[0].getSource();
            target_memory=target;
            if(last_mine == null || my_loc.isAdjacentTo(last_mine)) {
                last_mine = null;
            }
            return;
        }

        // no enemy rats
        if (enemyRats.length==0) {
            if(rc.getRawCheese()>Math.sqrt(rc.getLocation().distanceSquaredTo(king_loc))/cheese_return_const){
                //  beginBacktrackToKing();
                target = king_loc;        // target will get overridden by applyBacktrackTargetIfNeeded()
                target_memory = target;
                return;
            }



            if(last_mine == null || my_loc.isAdjacentTo(last_mine) ||
                    (was_making_king && (rc.readSharedArray(36)==0 || rc.readSharedArray(36)==1))) {
                last_mine=null;
                if (rc.canSenseLocation(target) || turns_since_progress>50) { // change
                    new_target();
                }
            }
            else {
                target = last_mine;
                target_memory=target;
                explore_direction = rc.getDirection();
            }


        }


    }

    public static boolean facing_away(Direction toMe, Direction enemyFacing) {
        int a = toMe.getDirectionOrderNum();
        int b = enemyFacing.getDirectionOrderNum();
        int diff = Math.abs(a - b);
        diff = Math.min(diff, 8 - diff); // wrap
        // diff 0 = facing me, 1 = near-facing me
        return diff >= 3; // 3-4 means mostly away
    }



    public static void attack(MapLocation loc) throws GameActionException {
        if (rc.canAttack(loc)) {
            Globals.attackTargetDir = rc.getLocation().directionTo(loc).ordinal();
            if(rc.getGlobalCheese()<100){
                rc.attack(loc,rc.getRawCheese());
                Globals.turnValue += rc.getRawCheese();
                return;
            }
            int cheese_value=(int)Math.sqrt(rc.getCurrentRatCost())/100;
            if(rc.canSenseLocation(loc)) {
                RobotInfo enemy = rc.senseRobotAtLocation(loc);
                if (enemy.getHealth()>rc.getHealth()+10+Math.sqrt(cheese_value)) {
                    if(Math.pow(enemy.getHealth()-rc.getHealth()-10,2)<=100){
                        cheese_value=(int)Math.pow(enemy.getHealth()-rc.getHealth()-10,2);
                    }
                }
                else if(Math.pow(enemy.getHealth(),2)<cheese_value){
                    cheese_value=(int)Math.pow(enemy.getHealth(),2);
                }
                Globals.attackTargetDir = rc.getLocation().directionTo(loc).ordinal();
                rc.attack(loc, cheese_value);
                Globals.turnValue += cheese_value;
            }
        }
    }

    public static Direction turn_dir;

    public static void enemy() throws GameActionException {
        // Sense nearby enemy rats (keep it fresh)
        enemyRats = rc.senseNearbyRobots(-1, my_team.opponent());

        // 1) autothrow first
        auto_throw();
        RobotInfo enemyKing = null;
        for (RobotInfo e : enemyRats) {
            if (e.getType() == UnitType.RAT_KING) {
                enemyKing = e;
                break;
            }
        }
        rc.setIndicatorString("enemy len "+ enemyRats.length);
        if (enemyKing != null) {
            MapLocation atk_loc = enemyKing.getLocation().subtract(my_loc.directionTo(enemyKing.getLocation()));
            if(rc.canAttack(atk_loc)){
                if (rc.canTurn()) {
                    Direction face = my_loc.directionTo(enemyKing.getLocation());
                    if (face != Direction.CENTER && rc.getDirection() != face) rc.turn(face);
                }
                attack(atk_loc);
                // still allow throw/carry later if action remains, but usually attack consumes it
            }
        }

        // 2) try to carry a rat (current location)
        for (RobotInfo rat : enemyRats) {
            if (rc.canCarryRat(rat.getLocation())) {
                Globals.carryTargetDir = rc.getLocation().directionTo(rat.getLocation()).ordinal();
                rc.carryRat(rat.getLocation());
                Globals.turnValue += 40;
                break;
            }
        }

        // List to store locations that match criteria
        MapLocation thrown=null;

        for (MapInfo info : mapInfos) {
            MapLocation loc = info.getMapLocation();

            // 1. Must be impassable
            if (!info.isPassable()) {

                // 2. Must NOT have enemy NPCs, wall, or dirt
                boolean hasObstacle = false;

                if(info.isWall()) hasObstacle = true;
                else  if (info.isDirt()) hasObstacle = true;
                else if (rc.senseRobotAtLocation(loc)!=null) hasObstacle=true;

                if (!hasObstacle) {
                    thrown=loc;
                    break;
                }
            }
        }

        // 3) use TileScore to pick best adjacent tile to go to
        Direction bestDir = Direction.CENTER;
        int bestScore = Integer.MIN_VALUE;

        String str="";

        if (rc.isMovementReady()) {
            for (Direction d : Direction.allDirections()) {
                if (!rc.canMove(d)) continue;
                MapLocation tile = my_loc.add(d);
                int s = TileScore.score(rc, enemyRats, friendlyRats, tile,thrown,null);
                str+=""+d+" score: " + s + "\n";
                if (s > bestScore) {
                    bestScore = s;
                    bestDir = d;
                }
            }
        }

        // If no legal move found, treat as "stay"
        MapLocation bestTile = my_loc;
        if (bestDir != null && rc.isMovementReady() && rc.canMove(bestDir)) {
            bestTile = my_loc.add(bestDir);
        }

        // 4) check if can carry at that location (simulate by checking enemies adjacent-to bestTile)
        //    If yes, do NOT attack first (per your ordering); we'll move and then carry.
        boolean canCarryAfterMove = false;
        for (RobotInfo rat : enemyRats) {
            // carry range appears to be adjacency-based in your code (you check canCarryRat(ratLoc))
            // We approximate: if we'd be adjacent (or same tile, unlikely) after moving.
            if (bestTile.distanceSquaredTo(rat.getLocation()) <= 2
                    && (rat.getHealth()<rc.getHealth() || facing_away(rat.getDirection(),rc.getDirection()))) {
                canCarryAfterMove = true;
                break;
            }
        }

        RobotInfo closest = null;
        int bestDist = Integer.MAX_VALUE;
        for (RobotInfo rat : enemyRats) {
            int d = my_loc.distanceSquaredTo(rat.getLocation());
            if (d < bestDist) {
                bestDist = d;
                closest = rat;
            }
        }


        // 5) if NOT going to be able to carry after move, then try to attack now
        if (!canCarryAfterMove && closest!=null) {

            boolean may_trap=true;

            int bytes1=Clock.getBytecodesLeft();

            for (Direction dir: adjacentDirections){
                MapLocation loc=closest.getLocation().add(dir);
                if (rc.canSenseLocation(loc)){
                    if(rc.senseMapInfo(loc).getTrap()==TrapType.RAT_TRAP){
                        may_trap=false;
                        break;
                    }
                }
            }

            int bytes2=Clock.getBytecodesLeft();
         //   System.out.println("rolled for loop bytes: "+(bytes1-bytes2));


            if(may_trap && my_loc.distanceSquaredTo(closest.location)<9) {
                Direction to_closest = my_loc.directionTo(closest.getLocation());
                Direction op = to_closest.opposite();

                if (closest.health > 70) {
                    if (rc.isActionReady()) {
                        if (rc.canPlaceRatTrap(my_loc.add(to_closest))) {
                            rc.placeRatTrap(my_loc.add(to_closest));
                            mapInfos=rc.senseNearbyMapInfos();
                            if (rc.isMovementReady()) {
                                for (Direction d : new Direction[]{to_closest,to_closest.rotateLeft(),to_closest.rotateRight()}) {
                                    if (!rc.canMove(d)) continue;
                                    MapLocation tile = my_loc.add(d);
                                    int s = TileScore.score(rc, enemyRats, friendlyRats, tile,thrown,my_loc.add(to_closest));
                                    str+=""+d+" score: " + s + "\n";
                                    if (s > bestScore) {
                                        bestScore = s;
                                        bestDir = d;
                                    }
                                }
                            }
                        } else if (rc.canPlaceRatTrap(my_loc.add(to_closest.rotateLeft()))) {
                            rc.placeRatTrap(my_loc.add(to_closest.rotateLeft()));
                            mapInfos=rc.senseNearbyMapInfos();
                            if (rc.isMovementReady()) {
                                for (Direction d : new Direction[]{to_closest.rotateLeft(),to_closest.rotateLeft().rotateLeft(),to_closest}) {
                                    if (!rc.canMove(d)) continue;
                                    MapLocation tile = my_loc.add(d);
                                    int s = TileScore.score(rc, enemyRats, friendlyRats, tile,thrown,my_loc.add(to_closest.rotateLeft()));
                                    str+=""+d+" score: " + s + "\n";
                                    if (s > bestScore) {
                                        bestScore = s;
                                        bestDir = d;
                                    }
                                }
                            }
                        } else if (rc.canPlaceRatTrap(my_loc.add(to_closest.rotateRight()))) {
                            rc.placeRatTrap(my_loc.add(to_closest.rotateRight()));
                            mapInfos=rc.senseNearbyMapInfos();
                            if (rc.isMovementReady()) {
                                for (Direction d : new Direction[]{to_closest,to_closest.rotateRight().rotateRight(),to_closest.rotateRight()}) {
                                    if (!rc.canMove(d)) continue;
                                    MapLocation tile = my_loc.add(d);
                                    int s = TileScore.score(rc, enemyRats, friendlyRats, tile,thrown,my_loc.add(to_closest.rotateRight()));
                                    str+=""+d+" score: " + s + "\n";
                                    if (s > bestScore) {
                                        bestScore = s;
                                        bestDir = d;
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }
        if (!canCarryAfterMove) {
            for (RobotInfo rat : enemyRats) {
                if (rc.canAttack(rat.getLocation())) {
                    attack(rat.getLocation());
                    break;
                }
            }
        }

        // 6) move to that location
        MapLocation new_loc=my_loc.add(bestDir);

        for (RobotInfo rat : enemyRats) {
            int d = new_loc.distanceSquaredTo(rat.getLocation());
            if (d < bestDist) {
                bestDist = d;
                closest = rat;
            }
        }
        if (bestDir != null && rc.isMovementReady() && rc.canMove(bestDir)) {
            if (closest!=null && rc.canTurn()) {
                Direction nextDir = new_loc.directionTo(closest.getLocation());
                if (nextDir == bestDir && nextDir!=Direction.CENTER) {
                    rc.turn(nextDir);
                }
            }
            rc.move(bestDir);
            get_info();
            my_loc = rc.getLocation();
        }

        // Refresh enemies after moving (important for the next steps)
        enemyRats = rc.senseNearbyRobots(-1, my_team.opponent());

        // 7) face the closest enemy
        for (RobotInfo rat : enemyRats) {
            int d = my_loc.distanceSquaredTo(rat.getLocation());
            if (d < bestDist) {
                bestDist = d;
                closest = rat;
            }
        }
        if (closest != null) {
            Direction face = my_loc.directionTo(closest.getLocation());
            if (face != Direction.CENTER && rc.canTurn() && rc.getDirection() != face) {
                rc.turn(face);
            }
        }

        // 8) autothrow again
        auto_throw();

        // 9) try to carry (after move)
        for (RobotInfo rat : enemyRats) {
            if (rc.canCarryRat(rat.getLocation())) {
                Globals.carryTargetDir = rc.getLocation().directionTo(rat.getLocation()).ordinal();
                rc.carryRat(rat.getLocation());
                Globals.turnValue += 40;
                break;
            }
        }

        if (!canCarryAfterMove && closest!=null) {

            boolean may_trap=true;
            for (Direction dir: adjacentDirections){
                MapLocation loc=closest.getLocation().add(dir);
                if (rc.canSenseLocation(loc)){
                    if(rc.senseMapInfo(loc).getTrap()==TrapType.RAT_TRAP){
                        may_trap=false;
                        break;
                    }
                }
            }


            if(may_trap && my_loc.distanceSquaredTo(closest.location)<9) {
                Direction to_closest = my_loc.directionTo(closest.getLocation());
                Direction op = to_closest.opposite();

                if (closest.health > 70) {
                    if (rc.isActionReady()) {
                        if (rc.canPlaceRatTrap(my_loc.add(to_closest))) {
                            rc.placeRatTrap(my_loc.add(to_closest));
                        } else if (rc.canPlaceRatTrap(my_loc.add(to_closest.rotateLeft()))) {
                            rc.placeRatTrap(my_loc.add(to_closest.rotateLeft()));
                        } else if (rc.canPlaceRatTrap(my_loc.add(to_closest.rotateRight()))) {
                            rc.placeRatTrap(my_loc.add(to_closest.rotateRight()));
                        }
                    }
                }
            }

        }

        // 10) try to attack (after move)
        for (RobotInfo rat : enemyRats) {
            if (rc.canAttack(rat.getLocation())) {
                attack(rat.getLocation());
                break;
            }
        }
    }


    static int threatLevel(MapLocation tile) {
        int t = 0;
        for (RobotInfo e : enemyRats) {
            int d = tile.distanceSquaredTo(e.getLocation());
            if (d <= 9) t++; // replace with real threat radius
        }
        return t;
    }


    public static MapLocation last_mine;
    public static boolean new_king=false;
    public static MapLocation signal_mine;

    public static void look_for_cheese_and_mine() throws GameActionException {

        int num_kings=0;
        for (int i=0; i<5; i++) {
            MapLocation loc = read_loc(2 * i);
            if (loc.x == 0 && loc.y == 0) {
                break;
            }
            num_kings++;
        } // chnaged

        MapInfo[] info=mapInfos;
        int total_cheese=0;
        for (MapInfo mi : info) {
            total_cheese+=mi.getCheeseAmount();
        }
        if (last_mine!=null && total_cheese>60+last_mine.distanceSquaredTo(new MapLocation(mapWidth/2,mapHeight/2))
                && my_loc.distanceSquaredTo(king_loc)>133
                && rc.getRoundNum()>250-total_cheese && num_kings<2){
            new_king=true;
            signal_mine=last_mine;
        }
        for (MapInfo mi : info) {
            if (mi.hasCheeseMine()) {
                if(rc.canSenseLocation(mi.getMapLocation())) {
                    RobotInfo rat = rc.senseRobotAtLocation(mi.getMapLocation());
                    if (rat == null || rat.getTeam() == opponentTeam || rat.getType() != UnitType.RAT_KING) {
                        last_mine = mi.getMapLocation();
                    }
                } else if (last_mine == null ){
                    last_mine = mi.getMapLocation();
                }
            }
            if (mi.getCheeseAmount()>0){
                if (rc.canPickUpCheese(mi.getMapLocation()) && rc.getRawCheese()<60) {
                    rc.pickUpCheese(mi.getMapLocation());

                    if(last_mine==null){
                        target=my_loc.add(rc.getDirection()).add(rc.getDirection()).add(rc.getDirection()).add(rc.getDirection());
                        target_memory=target;
                    }
                }
                else if (enemyRats.length==0 && rc.getRawCheese()<60) {
                    target=mi.getMapLocation();
                    target_memory=target;
                }
            }
        }
        if (last_mine!=null && signal_mine==null){
            signal_mine=last_mine;
        }
    }


    public static int have_been_carrying=0;

    public static void auto_throw() throws GameActionException {


        if (!rc.canThrowRat() || rc.getCarrying().getTeam()==my_team){return;}

        if(rc.canThrowRat() && enemyRats.length!=0 && rc.canTurn() ) {
            for (RobotInfo rat: enemyRats){
                int dist=my_loc.distanceSquaredTo(rat.getLocation());
                if(dist>2 && dist<9 && dist!=5){
                    Direction dir = rc.getLocation().directionTo(rat.getLocation());
                    if (dir != Direction.CENTER && rc.canDropRat(dir)) {
                        rc.turn(dir);
                        if(rc.canThrowRat()) {
                            doThrow();
                            return;
                        }
                    }
                }
            }
        }

        if(rc.canThrowRat()) {
            MapLocation next1 = rc.getLocation().add(rc.getDirection());
            MapLocation next2 = rc.getLocation().add(rc.getDirection()).add(rc.getDirection());
            if (rc.canSenseLocation(next1) && rc.canSenseLocation(next2)) {
                if (rc.senseMapInfo(next1).isPassable() && (!rc.senseMapInfo(next2).isPassable()
                        || (rc.senseRobotAtLocation(next2)!=null &&  rc.senseRobotAtLocation(next2).getTeam()==opponentTeam))) {
                    doThrow();
                    return;
                }
            }
        }
        if(rc.canTurn() && rc.isActionReady()) {
            for (Direction d : adjacentDirections) {
                MapLocation next1 = rc.getLocation().add(d);
                MapLocation next2 = rc.getLocation().add(d).add(d);
                if (rc.canSenseLocation(next1) && rc.canSenseLocation(next2)) {
                    if (rc.senseMapInfo(next1).isPassable() && !rc.senseMapInfo(next2).isPassable()) {
                        rc.turn(d);
                        if(rc.canThrowRat()) {
                            doThrow();
                            return;
                        }
                        break;
                    }
                }
            }
        }
        if(have_been_carrying>8){
            if (rc.canThrowRat()){
                doThrow();
            }
        }
    }

    public static void doThrow() throws GameActionException {
        Globals.throwTargetDir = rc.getDirection().ordinal();
        rc.throwRat();
        Globals.turnValue += 40;
        MapLocation dir1 = rc.getLocation().add(rc.getDirection());
        if (rc.canSenseLocation(dir1) && rc.senseMapInfo(dir1).isWall()) {
            Globals.turnValue += 40;
            return;
        }
        MapLocation dir2 = dir1.add(rc.getDirection());
        if (rc.canSenseLocation(dir2) && (rc.senseMapInfo(dir2).isWall() ||
                (rc.senseRobotAtLocation(dir2) != null && rc.senseRobotAtLocation(dir2).getTeam() == opponentTeam))) {
            Globals.turnValue += 40;
        }
    }

    public static void king_signal() throws GameActionException {
        // bits 0-2 symmetry
        int msg = 0;
        // bits 3-12 cheese/mine loc
        if (signal_mine!=null){
            msg+=translate_coords_to_int(signal_mine)*8;
        }
        // bits 13-22 enemy king loc
        if(enemy_king!=null){
            msg+=translate_coords_to_int(enemy_king)*8*1024;
        }
        if (new_king){
            msg+=1*1024*1024*8;
        }
        rc.squeak(msg);
        signal_mine=null;

    }

    public static void combat_signal() throws GameActionException {
        // squeak 1 to signal there is a fight
        if (enemyRats.length!=0) {
            RobotInfo closest = null;
            int bestDist = Integer.MAX_VALUE;

            for (RobotInfo rat: enemyRats) {
                int d = my_loc.distanceSquaredTo(rat.getLocation());
                if (d < bestDist) {
                    bestDist = d;
                    closest = rat;
                }
            }
            rc.squeak(translate_coords_to_int(closest.getLocation()));
        }
        else if (last_turn_enemy_rats.length!=0) {
            RobotInfo closest = null;
            int bestDist = Integer.MAX_VALUE;

            for (RobotInfo rat: last_turn_enemy_rats) {
                int d = my_loc.distanceSquaredTo(rat.getLocation());
                if (d < bestDist) {
                    bestDist = d;
                    closest = rat;
                }
            }
            rc.squeak(translate_coords_to_int(closest.getLocation()));
        }
    }

    public static MapLocation[] signaled_rats = new MapLocation[0];


    public static void read_signal() throws GameActionException {
        Message[] msgs1 = rc.readSqueaks(rc.getRoundNum() - 1);
        Message[] msgs2 = rc.readSqueaks(rc.getRoundNum());

        int count = 0;
        for (Message msg : msgs1) count++;
        for (Message msg : msgs2) count++;

        if (count == 0) {
            signaled_rats = new MapLocation[0];
            return;
        }

        MapLocation[] result = new MapLocation[count];
        int idx = 0;

        // collect sources (may include duplicates)
        MapLocation[] sources = new MapLocation[msgs1.length + msgs2.length];
        int sIdx = 0;

        for (Message msg : msgs1) {
            int b = msg.getBytes();
            if (b != 0) result[idx++] = translate_int_to_coords(b);
            MapLocation src = msg.getSource();
            if (src != null) sources[sIdx++] = src;
        }
        for (Message msg : msgs2) {
            int b = msg.getBytes();
            if (b != 0) result[idx++] = translate_int_to_coords(b);

            MapLocation src = msg.getSource();
            if (src == null) continue;

            // If we can sense the source tile, use robot ID to dedupe via seenRobots

            int id = msg.getSenderID();

            // If already seen, skip adding this source again
            if (seenRobots.containsKey(id)) {
                // still refresh "last seen" info if you want
                seenRobots.put(id, new SeenRobot(
                        new RobotInfo(id,my_team,UnitType.BABY_RAT,100,src,rc.getDirection(),1,0,null), rc.getRoundNum()));
                continue;
            }

            // Mark as seen and include the source
            seenRobots.put(id, new SeenRobot( new RobotInfo(id,my_team,UnitType.BABY_RAT,100,src,rc.getDirection(),1,0,null), rc.getRoundNum()));
            sources[sIdx++] = src;
        }


        signaled_rats = result;

        // ---- Merge squeak senders into friendlyRats ----
        if (friendlyRats == null) friendlyRats = new RobotInfo[0];

        // small headroom; if you expect lots of squeakers, bump 64 -> 128
        RobotInfo[] merged = new RobotInfo[friendlyRats.length + 64];
        int m = 0;

        // copy existing
        for (RobotInfo r : friendlyRats) {
            if (r != null) merged[m++] = r;
        }

        for (int i = 0; i < sIdx; i++) {
            MapLocation src = sources[i];
            if (src == null) continue;

            // If we can sense the location, already see the rat
            if (rc.canSenseLocation(src)) continue;

            merged[m++] = new RobotInfo(1,my_team,UnitType.BABY_RAT,100,src,rc.getDirection(),1,0,null);
        }

        RobotInfo[] finalFriends = new RobotInfo[m];
        for (int i = 0; i < m; i++) finalFriends[i] = merged[i];
        friendlyRats = finalFriends;
    }




    public static MapLocation enemy_king=null;
    public static Message[] friendly;
    public static int reached_edge=0;
    public static double cheese_return_const=rc.getAllCheese()/40;
    public static int king_const=15;
    public static MapLocation init_loc;
    public static Direction explore_direction;

    static int spawnCorner = -1;
    static boolean signal=true;

    static void initSpawnCorner(MapLocation spawn){
        spawnCorner = cornerIndex(spawn);
        visitedCorners = (1 << spawnCorner); // permanently exclude
    }

    public static boolean facing_same_way(Direction cat_dir, Direction rat_dir){
        int cat_x=cat_dir.getDeltaX();
        int cat_y=cat_dir.getDeltaY();
        int rat_x=rat_dir.getDeltaX();
        int rat_y=rat_dir.getDeltaY();
        if(cat_x==rat_x && Math.abs(cat_y-rat_y)<=1){
            return true;
        }
        if( cat_y==rat_y && Math.abs(cat_x-rat_x)<=1){
            return true;
        }
        return false;
    }

    public static RobotInfo cat_mem;
    public static int cat_mem_dur=0;

    public static void prio_move(Direction turn_dir) throws GameActionException {
        Direction dir = WbugNav.moveTo(target);
        if(dir==null || dir==Direction.CENTER) {
            if (rc.canTurn() && turn_dir!=Direction.CENTER) {
                rc.turn(turn_dir);
            }
            return;
        }

        if(rc.canTurn() && rc.getDirection()!=dir){
            rc.turn(dir);
        }
        if(rc.canMove(dir)){
            rc.move(dir);
            get_info();
        }
        if (rc.canTurn() && turn_dir!=Direction.CENTER) {
            rc.turn(turn_dir);
        }
    }

    public static void forced_move(Direction turn_dir, Direction move_dir) throws GameActionException {

        if(rc.canTurn() && rc.getDirection()!=move_dir  && move_dir!=Direction.CENTER){
            rc.turn(move_dir);
        }
        if(rc.canMove(move_dir)){
            rc.move(move_dir);
            get_info();
        }
        if (rc.canTurn() && turn_dir!=Direction.CENTER) {
            rc.turn(turn_dir);
        }
    }

    static boolean outsideCatVision(RobotInfo cat) {
        return !my_loc.isWithinDistanceSquared(
                cat.getLocation(),
                21,
                cat.getDirection(),
                Math.PI
        );
    }


    static void cat_target(RobotInfo[] cats) throws GameActionException {
        if (cats.length == 0 && cat_mem == null) return;

        RobotInfo cat;
        if (cats.length > 0) {
            cat = cats[0];
            cat_mem_dur = 3;
        } else {
            cat = cat_mem;
            cat_mem_dur--;
        }

        MapLocation catLoc = cat.getLocation();
        Direction catDir = cat.getDirection();
        if(cat_mem_dur!=3){
            catLoc=catLoc.add(catDir);
        }
        cat_mem = cat;

        boolean coop = rc.isCooperation();
        int round = rc.getRoundNum();

        // ===============================
        // FORBIDDEN ZONE IN FRONT OF CAT
        // ===============================
        MapLocation front = catLoc.add(catDir);
        MapLocation frontLeft = catLoc.add(catDir.rotateLeft());
        MapLocation frontRight = catLoc.add(catDir.rotateRight());

        MapLocation[] forbidden;

        if (catDir!=Direction.SOUTH && catDir!=Direction.WEST && catDir!=Direction.SOUTHWEST){
            front=front.add(catDir);
            frontLeft=frontLeft.add(catDir);
            frontRight=frontRight.add(catDir);
        }

        //  rc.setIndicatorLine(my_loc,front,255,0,255);
        //  rc.setIndicatorLine(my_loc,frontLeft,255,0,255);
        // rc.setIndicatorLine(my_loc,frontRight,255,0,255);
        forbidden = new MapLocation[] { front, frontLeft, frontRight };

        // ===============================
        // EARLY / NON-COOP MODE
        // ===============================
        if (round < 50 || (round < 500 && !coop)) {
            final int DESIRED_DIST_SQ = 16;
            Direction bestDir = null;
            int bestScore = Integer.MIN_VALUE;

            for (Direction d : adjacentDirections) {
                if (!rc.canMove(d)) continue;

                MapLocation next = my_loc.add(d);

                // Skip forbidden tiles
                boolean bad = false;
                for (MapLocation f : forbidden) {
                    if (next.equals(f)) {
                        bad = true;
                        break;
                    }
                }
                if (bad) continue;

                int nextDist = next.distanceSquaredTo(catLoc);
                int score = (nextDist < DESIRED_DIST_SQ) ? -Math.abs(nextDist - DESIRED_DIST_SQ) * 200 : 0;
                if (nextDist <= 4) score -= 5000;
                score -= next.distanceSquaredTo(target);

                // Orbiting bonus
                Direction toCat = next.directionTo(catLoc);
                score += (toCat == d.rotateLeft().rotateLeft() || toCat == d.rotateRight().rotateRight()) ? 200 : 0;
                score += (toCat == d.rotateLeft().rotateLeft().rotateLeft() || toCat == d.rotateRight().rotateRight().rotateRight()) ? 100 : 0;
                score += (toCat == d.rotateLeft() || toCat == d.rotateRight()) ? 100 : 0;

                if (score > bestScore) {
                    bestScore = score;
                    bestDir = d;
                }
            }

            if (bestDir != null) {
                if (rc.canTurn() && rc.getDirection() != bestDir) rc.turn(bestDir);
                if (rc.canMove(bestDir)) {rc.move(bestDir);
                    get_info();}
            }

            if (rc.canAttack(catLoc)) { Globals.attackTargetDir = rc.getLocation().directionTo(catLoc).ordinal(); rc.attack(catLoc); Globals.turnValue += GameConstants.RAT_BITE_DAMAGE; return; }

            MapLocation desired = catLoc.add(catDir).add(catDir);
            MapLocation[] trapPriority = { front, desired, frontLeft, frontRight };
            for (MapLocation t : trapPriority) {
                if (rc.canPlaceCatTrap(t)) {
                    rc.placeCatTrap(t);
                    return;
                }
            }
            return;
        }

        // ===============================
        // NORMAL MODE
        // ===============================
        MapLocation desired = catLoc.add(catDir).add(catDir);
        target = desired;
        target_memory = target;

        if (rc.canAttack(catLoc)) { Globals.attackTargetDir = rc.getLocation().directionTo(catLoc).ordinal(); rc.attack(catLoc); Globals.turnValue += GameConstants.RAT_BITE_DAMAGE; return; }

        MapLocation[] trapPriority = { front, desired, frontLeft, frontRight };
        for (MapLocation t : trapPriority) {
            if (rc.canPlaceCatTrap(t)) {
                rc.placeCatTrap(t);
                return;
            }
        }

        Direction bestDir = null;
        int bestScore = Integer.MIN_VALUE;
        int curToDesired = my_loc.distanceSquaredTo(desired);
        int curToCat = my_loc.distanceSquaredTo(catLoc);

        for (Direction dir : adjacentDirections) {
            if (!rc.canMove(dir)) continue;

            MapLocation next = my_loc.add(dir);

            // Skip forbidden tiles
            boolean bad = false;
            for (MapLocation f : forbidden) {
                if (next.equals(f)) {
                    bad = true;
                    break;
                }
            }
            if (bad) continue;

            int score = (curToDesired - next.distanceSquaredTo(desired)) * 1000 +
                    (next.distanceSquaredTo(catLoc) - curToCat) * 10;

            boolean entersVision = next.isWithinDistanceSquared(catLoc, 30 * 30, catDir, Math.PI);
            if (entersVision) score -= 5000;

            if (score > bestScore) {
                bestScore = score;
                bestDir = dir;
            }
        }

        if (bestDir != null) {
            if (rc.canTurn()) rc.turn(bestDir);
            if (rc.canMove(bestDir)){ rc.move(bestDir);
                get_info();}
        }

        for (MapLocation t : trapPriority) {
            if (rc.canPlaceCatTrap(t)) {
                rc.placeCatTrap(t);
                return;
            }
        }

        if (rc.canAttack(catLoc)) { Globals.attackTargetDir = rc.getLocation().directionTo(catLoc).ordinal(); rc.attack(catLoc); Globals.turnValue += GameConstants.RAT_BITE_DAMAGE; }
    }








    public static int carried=0;
    public static MapLocation target_memory;

    public static void explore_state() throws GameActionException {
        if(!rc.isActionReady()) {
            MapLocation map_loc = my_loc.add(rc.getDirection());
            if (rc.canSenseLocation(map_loc)) {
                if (rc.senseMapInfo(map_loc).isDirt()){
                    if(rc.canTurn()){
                        rc.turn(rc.getDirection());
                    }
                    if(rc.canMove(Direction.CENTER)){
                        rc.move(Direction.CENTER);
                    }
                }
            }
        }
        // read_signal();
        exploring_set_target();
        target_memory=target;
        boolean combat_allowed=can_get_target();
        if(combat_allowed){
            look_for_cheese_and_mine();
        }
        if(rc.getRawCheese()>0 && my_loc.distanceSquaredTo(king_loc)<=19) {
            forced_move(my_loc.directionTo(king_loc),my_loc.directionTo(king_loc));
        }
        else if(!isnew){
            int bytes=Clock.getBytecodesLeft();
            if (target==target_memory && combat_allowed) {
                boolean found_dirt=false;
                int dist=Integer.MAX_VALUE;
                MapInfo[] close_infos=rc.senseNearbyMapInfos(9);
                MapInfo[] infos = new MapInfo[64 * 64];
                for (MapInfo ci : close_infos) {
                    if (ci != null) {
                        MapLocation cl = ci.getMapLocation();
                        infos[cl.x + 64 * cl.y] = ci;
                    }
                }
                for (int i = 0; i < 26; i++) {
                    if (close_infos == null || i >= close_infos.length) break;

                    MapInfo info = close_infos[i];
                    if (info == null) continue;

                    if (info.isWall()) {
                        MapLocation loc = info.getMapLocation();

                        MapLocation locN = loc.add(Direction.NORTH);
                        if (infos[locN.x + 64 * locN.y] != null) {
                            if (infos[locN.x + 64 * locN.y].isDirt()) {
                                found_dirt = true;
                                if (my_loc.distanceSquaredTo(locN) <= dist) {
                                    dist = my_loc.distanceSquaredTo(locN);
                                    target = locN;
                                    Direction dir = my_loc.directionTo(target);
                                    if (dir != Direction.CENTER && rc.canTurn()) {
                                        rc.turn(dir);
                                    }
                                }
                            }
                        }

                        MapLocation locS = loc.add(Direction.SOUTH);
                        if (locS.x + 64 * locS.y > 0 && infos[locS.x + 64 * locS.y] != null) {
                            if (infos[locS.x + 64 * locS.y].isDirt()) {
                                found_dirt = true;
                                if (my_loc.distanceSquaredTo(locS) <= dist) {
                                    dist = my_loc.distanceSquaredTo(locS);
                                    target = locS;
                                    Direction dir = my_loc.directionTo(target);
                                    if (dir != Direction.CENTER && rc.canTurn()) {
                                        rc.turn(dir);
                                    }
                                }
                            }
                        }

                        MapLocation locE = loc.add(Direction.EAST);
                        if (infos[locE.x + 64 * locE.y] != null) {
                            if (infos[locE.x + 64 * locE.y].isDirt()) {
                                found_dirt = true;
                                if (my_loc.distanceSquaredTo(locE) <= dist) {
                                    dist = my_loc.distanceSquaredTo(locE);
                                    target = locE;
                                    Direction dir = my_loc.directionTo(target);
                                    if (dir != Direction.CENTER && rc.canTurn()) {
                                        rc.turn(dir);
                                    }
                                }
                            }
                        }

                        MapLocation locW = loc.add(Direction.WEST);
                        if (locW.x + 64 * locW.y > 0 && infos[locW.x + 64 * locW.y] != null) {
                            if (infos[locW.x + 64 * locW.y].isDirt()) {
                                found_dirt = true;
                                if (my_loc.distanceSquaredTo(locW) <= dist) {
                                    dist = my_loc.distanceSquaredTo(locW);
                                    target = locW;
                                    Direction dir = my_loc.directionTo(target);
                                    if (dir != Direction.CENTER && rc.canTurn()) {
                                        rc.turn(dir);
                                    }
                                }
                            }
                        }
                    }
                }

                if (!found_dirt){
                    target_memory=target;
                }
            }
            if(target!=target_memory){
                if(rc.canRemoveDirt(target)){
                    rc.removeDirt(target);
                    target=target_memory;
                }
                else if(rc.canTurn() && my_loc.distanceSquaredTo(target)<2){
                    Direction dir=my_loc.directionTo(target);
                    if(dir!=null && dir!=Direction.CENTER){
                        rc.turn(dir);
                    }
                }
            }
            move_and_turn();
        }
        get_info();

    }

    public static void combat_state() throws GameActionException {
        // Update enemy info
        read_signal();
        rc.squeak(0);

        enemy();

        // Retreat if adjacent enemies block movement
        if (!rc.isActionReady()) {
            for (RobotInfo rat : enemyRats) {
                if (rat.getID()!=carried && rat.getLocation().isAdjacentTo(rc.getLocation())) {
                    Direction away = rc.getLocation().directionTo(rat.getLocation()).opposite();
                    if (rc.canMove(away)) {
                        rc.move(away); // move away but still face enemies later
                        get_info();
                        break; // move once per turn
                    }
                }
            }
        }

        // Determine closest enemy (current or last turn)
        MapLocation closest = null;
        int minDist = Integer.MAX_VALUE;

        // Current enemies
        for (RobotInfo rat : enemyRats) {
            int d = rat.getLocation().distanceSquaredTo(my_loc);
            if (d < minDist) {
                minDist = d;
                closest = rat.getLocation();
            }
        }

        // If no current enemies, check last turn
        if (closest == null && (last_turn_enemy_rats!=null && last_turn_enemy_rats.length != 0)) {
            for (RobotInfo rat : last_turn_enemy_rats) {
                int d = rat.getLocation().distanceSquaredTo(my_loc);
                if (d < minDist) {
                    minDist = d;
                    closest = rat.getLocation();
                }
            }
        }

        // refine move

        if (minDist <9) {
            Direction dir = WbugNav.moveTo(target);


            if (dir==null || dir==Direction.CENTER){
                dir=my_loc.directionTo(closest);
            }
            if (dir==null || dir==Direction.CENTER){
                dir=rc.getDirection();
            }

            int score=Integer.MIN_VALUE;
            int left_score=Integer.MIN_VALUE;
            int right_score=Integer.MIN_VALUE;

            score = TileScore.score(rc, enemyRats, friendlyRats, my_loc.add(dir),null,null);
            if(rc.canMove(dir.rotateLeft())) {
                left_score = TileScore.score(rc, enemyRats, friendlyRats, my_loc.add(dir.rotateLeft()),null,null);
            }
            if (rc.canMove(dir.rotateRight()))
            {
                right_score = TileScore.score(rc, enemyRats, friendlyRats, my_loc.add(dir.rotateRight()),null,null);
            }

            Direction init_dir=dir;

            if (left_score > score) {
                dir=init_dir.rotateLeft();
                score = left_score;
            }
            if (right_score > score) {
                dir=init_dir.rotateRight();
            }


            closest = null;
            minDist = Integer.MAX_VALUE;

            // Current enemies
            for (RobotInfo rat : enemyRats) {
                int d = rat.getLocation().distanceSquaredTo(my_loc.add(dir));
                if (d < minDist) {
                    minDist = d;
                    closest = rat.getLocation();
                }
            }

            Direction turn_dir=my_loc.add(dir).directionTo(closest);
            if (turn_dir == null || turn_dir==Direction.CENTER) {
                turn_dir=rc.getDirection();
            }


            forced_move(turn_dir,dir);

        }

        // Face nearest enemy if any
        if (closest != null) {
            Direction dirToEnemy = my_loc.directionTo(closest);
            if (rc.getDirection() != dirToEnemy && rc.canTurn() && dirToEnemy != Direction.CENTER) {
                rc.turn(dirToEnemy);
            }
            // Move toward enemy if possible
            if (rc.isActionReady()) {
                prio_move(dirToEnemy);
            }
        }

        // Gather info and update enemy list again
        // enemy();
        auto_throw();
    }

    public static void cat_state() throws GameActionException {
        cat_target(cats);

        Direction dir=WbugNav.moveTo(target);
        if (dir!=null && dir !=Direction.CENTER && rc.canMove(dir)) {
            rc.move(dir);
            get_info();
        }
        if(rc.canTurn() && my_loc.directionTo(cat_mem.getLocation())!=Direction.CENTER){
            rc.turn(my_loc.directionTo(cat_mem.getLocation()));
        }
        //move_and_turn();
    }

    public enum State{
        explore,
        combat,
        cat,
        carrying,
        carried,
        thrown,
    }

    public static State state;

    static MapLocation getCarrySeekTarget() throws GameActionException {
        Message[] a = rc.readSqueaks(rc.getRoundNum());
        Message[] b = rc.readSqueaks(rc.getRoundNum() - 1);

        MapLocation best = null;
        int bestDist = Integer.MAX_VALUE;

        for (Message m : a) {
            MapLocation src = m.getSource();
            if (src == null) continue;

            if (rc.canSenseLocation(src)) {
                RobotInfo ri = rc.senseRobotAtLocation(src);
                if (ri != null && ri.getTeam() != my_team) continue;
            }

            int d = my_loc.distanceSquaredTo(src);
            if (d < bestDist) {
                bestDist = d;
                best = src;
            }
        }

        for (Message m : b) {
            MapLocation src = m.getSource();
            if (src == null) continue;

            if (rc.canSenseLocation(src)) {
                RobotInfo ri = rc.senseRobotAtLocation(src);
                if (ri != null && ri.getTeam() != my_team) continue;
            }

            int d = my_loc.distanceSquaredTo(src);
            if (d < bestDist) {
                bestDist = d;
                best = src;
            }
        }

        if (best != null) return best;
        return new MapLocation(mapWidth / 2, mapHeight / 2);
    }


    public static void determine_state() throws GameActionException {
        if(rc.isBeingCarried()){
            state=State.carried;
            return;
        }
        if(rc.isBeingThrown()) {
            state = State.thrown;
            return;
        }
        if(rc.getCarrying()!=null){
            state=State.carrying;
            return;
        }
        boolean combat_allowed=can_get_target();


        if(combat_allowed && (enemyRats.length>0 || (last_turn_enemy_rats!=null && last_turn_enemy_rats.length>0)) && rc.getRawCheese()<39){
            state= State.combat;
            signal=true;
        }
        else {
            if(rc.getLocation().distanceSquaredTo(king_loc)<17 && RobotPlayer.turnCount>5){
                signal=true;
            }else {
                signal=false;
            }
            if (combat_allowed && (cats.length!=0 || cat_mem_dur>0)) {
                state=State.cat;
            }
            else {
                state=State.explore;
            }
        }


    }

    public static MapLocation edge_from_dir(Direction dir) throws GameActionException {
        MapLocation loc=rc.getLocation();
        int x=loc.x;
        int y=loc.y;
        switch (dir) {
            case NORTH: return new MapLocation(x,mapHeight-1);
            case SOUTH: return new MapLocation(x,0);
            case EAST: return new MapLocation(mapWidth-1,y);
            case WEST: return new MapLocation(0,y);
            case NORTHEAST: {
                int slide=Math.min(mapWidth-x-1,mapHeight-y-1);
                return  new MapLocation(x+slide,y+slide);
            }
            case SOUTHEAST: {
                int slide=Math.min(mapWidth-x-1,y);
                return  new MapLocation(x+slide,y-slide);
            }
            case SOUTHWEST: {
                int slide=Math.min(x,y);
                return  new MapLocation(x-slide,y-slide);
            }
            case NORTHWEST: {
                int slide=Math.min(x,mapHeight-y-1);
                return  new MapLocation(x-slide,y+slide);
            }
        }
        return new MapLocation(mapWidth/2,mapHeight/2);
    }



    public static void carry_state() throws GameActionException {
        read_signal();
        auto_throw();

        if (enemyRats.length>0) {


            RobotInfo closestEnemy = null;
            int bestDist = Integer.MAX_VALUE;

            for (RobotInfo rat : enemyRats) {
                int d = my_loc.distanceSquaredTo(rat.getLocation());
                if (d < bestDist) {
                    bestDist = d;
                    closestEnemy = rat;
                }
            }

            if (closestEnemy != null) {
                MapLocation closestLoc = closestEnemy.getLocation();

                // At most adjacentDirections.length safe tiles
                MapLocation[] safeLocs = new MapLocation[adjacentDirections.length];
                int safeCount = 0;

                // 1) Collect safe locations
                for (Direction d : adjacentDirections) {
                    MapLocation testLoc = closestLoc.add(d).add(d);

                    boolean unsafe = false;
                    for (RobotInfo rat : enemyRats) {
                        if (rat == closestEnemy) continue;

                        if (rat.getLocation().distanceSquaredTo(testLoc) <= 8) {
                            unsafe = true;
                            break;
                        }
                    }

                    if (!unsafe) {
                        safeLocs[safeCount++] = testLoc;
                    }
                }

                // 2) Pick closest safe location to me
                if (safeCount > 0) {
                    MapLocation best = null;
                    bestDist = Integer.MAX_VALUE;

                    for (int i = 0; i < safeCount; i++) {
                        int d = my_loc.distanceSquaredTo(safeLocs[i]);
                        if (d < bestDist) {
                            bestDist = d;
                            best = safeLocs[i];
                        }
                    }

                    target = best;
                    target_memory=target;

                    if (target != null) {

                        // ---- MOVE TOWARD TARGET ----
                        Direction moveDir = my_loc.directionTo(target);
                        Direction turnDir = my_loc.directionTo(closestLoc);

                        if (moveDir != Direction.CENTER) {
                            if (rc.canTurn() && rc.getDirection() != moveDir && moveDir==turnDir) {
                                rc.turn(moveDir);
                            }
                            if (rc.canMove(moveDir)) {
                                rc.move(moveDir);
                                get_info();
                            }
                            if(rc.canMove(turnDir)) {
                                rc.move(turnDir);
                                get_info();
                            }
                        }

                        // ---- FIND NEW CLOSEST ENEMY AFTER MOVE ----
                        MapLocation newLoc = rc.getLocation(); // updated position
                        closestEnemy = null;
                        bestDist = Integer.MAX_VALUE;

                        for (RobotInfo rat : enemyRats) {
                            int d = newLoc.distanceSquaredTo(rat.getLocation());
                            if (d < bestDist) {
                                bestDist = d;
                                closestEnemy = rat;
                            }
                        }

                        // ---- TURN TO FACE THAT ENEMY ----
                        if (closestEnemy != null) {
                            Direction faceDir = newLoc.directionTo(closestEnemy.getLocation());
                            if (faceDir != Direction.CENTER && rc.canTurn()
                                    && rc.getDirection() != faceDir) {
                                rc.turn(faceDir);
                            }
                        }
                    }

                }




            }
            auto_throw();
            enemy();
        }
        else {
            // NEW: when no enemies are visible, seek fights:
            // 1) go to nearest friendly squeaker source, else
            // 2) go to center
            MapLocation seek = getCarrySeekTarget();
            target = seek;
            target_memory = target;

            Direction dir = my_loc.directionTo(target);
            if (dir == null || dir == Direction.CENTER) dir = rc.getDirection();

            // face toward seek
            if (rc.canTurn() && dir != Direction.CENTER) rc.turn(dir);

            // move toward seek using your existing move helper
            if (rc.isMovementReady()) {
                prio_move(dir);
            }

            // refresh + throw if possible
            get_info();
            auto_throw();

            // if we walked into enemies, immediately switch to enemy micro this same turn
            if (enemyRats != null && enemyRats.length > 0) {
                enemy();
            }
        }

        auto_throw();
    }

    public static void init_set_target() throws GameActionException {
        if(rc.readSharedArray(30)!=0){
            target=read_loc(30);
            target_memory=target;
        }
        else {
            int num_mines=0;
            for (int i=10; i<30; i++){
                if(rc.readSharedArray(i)!=0){
                    num_mines++;
                }
                else {
                    break;
                }
            }
            if( rc.readSharedArray(34)==0 && num_mines==0){
                target=edge_from_dir(explore_direction);
                target_memory=target;
                return;
            }
            if (rc.readSharedArray(34) != 0) {
                if (rc.getID() % (num_mines + 1) <= rc.getGlobalCheese()*20/2500) {
                    target = translate_int_to_coords(rc.readSharedArray(34));
                    target_memory=target;
                } else if (num_mines != 0) {
                    target = translate_int_to_coords(rc.readSharedArray(rc.getID() % (num_mines) + 10));
                    target_memory=target;
                }
            } else if (num_mines != 0) {
                target = translate_int_to_coords(rc.readSharedArray(rc.getID() % (num_mines) + 10));
                target_memory=target;
            }

        }
    }

    public static void carried_state() throws GameActionException {
        if(rc.canSenseLocation(my_loc)){
            RobotInfo rat=rc.senseRobotAtLocation(my_loc);
            if(rc.canTurn()) {
                rc.turn(rat.getDirection());
            }
        }
    }

    public static void thrown_state() throws GameActionException {
        for(Direction dir : adjacentDirections){
            if(rc.canSenseLocation(my_loc.add(dir))){
                if(rc.senseRobotAtLocation(my_loc.add(dir))!=null){
                    if(rc.senseRobotAtLocation(my_loc.add(dir)).getTeam()==my_team){
                        rc.disintegrate();
                    }
                }
            }
        }
    }
    // ===== MARKER TRAIL (every 5 turns) =====
    static final int MARKER_CAP = 60;
    static int[] markers = new int[MARKER_CAP];
    static int markerCount = 0;     // number of valid markers
    static int markerIdx = -1;      // backtrack pointer (from markerCount-1 down to 0)
    static boolean backtracking = false;

    // Tuning
    static final int MARKER_PERIOD = 5;
    static final int MARKER_REACHED_DIST_SQ = 2;  // "reached marker" threshold
    static final int KING_REACHED_DIST_SQ = 5;    // "reached king" threshold for reset

    static void maybeAddMarker() throws GameActionException {
        if (backtracking) return; // don't drop markers while backtracking
        if (king_loc == null) return;

        // every 5 turns
        if (rc.getRoundNum() % MARKER_PERIOD != 0) return;

        // Don't record if we're already basically at king
        if (my_loc.distanceSquaredTo(king_loc) <= KING_REACHED_DIST_SQ) return;

        int packed = translate_coords_to_int(my_loc);

        // Dedup: if last marker is basically same spot, skip
        if (markerCount > 0) {
            MapLocation last = translate_int_to_coords(markers[markerCount - 1]);
            if (last.distanceSquaredTo(my_loc) <= 2) return;
        }

        if (markerCount < MARKER_CAP) {
            markers[markerCount++] = packed;
        } else {
            // shift left (drop oldest) to keep the newest trail
            for (int i = 1; i < MARKER_CAP; i++) markers[i - 1] = markers[i];
            markers[MARKER_CAP - 1] = packed;
            markerCount = MARKER_CAP;
        }
    }

    static void beginBacktrackToKing() {
        if (backtracking) return;
        backtracking = true;
        markerIdx = markerCount - 1;
    }

    static void resetMarkersAndBacktrack() {
        backtracking = false;
        markerIdx = -1;
        markerCount = 0;
    }

    static void applyBacktrackTargetIfNeeded() throws GameActionException {
        if (!backtracking) return;

        // If we have markers left, target the current one
        if (markerIdx >= 0) {
            MapLocation m = translate_int_to_coords(markers[markerIdx]);

            // If already reached this marker, pop to the next
            if (my_loc.distanceSquaredTo(m) <= MARKER_REACHED_DIST_SQ) {
                markerIdx--;
                // fall through: if markerIdx < 0 we'll go to king directly
            } else {
                target = m;
                target_memory = target;
                return;
            }
        }

        // No markers left => just path to king directly
        target = king_loc;
        target_memory = target;
    }

    static void maybeFinishBacktrack() throws GameActionException {
        if (!backtracking) return;
        if (king_loc == null) return;

        // When we are close enough to king, consider it reached and reset
        if (my_loc.distanceSquaredTo(king_loc) <= KING_REACHED_DIST_SQ) {
            //   resetMarkersAndBacktrack();
        }
    }


    public static boolean digger=false;
    public static void digger_state() throws GameActionException {}
    public static MapLocation get_king_loc() throws GameActionException {
        MapLocation king_loc=null;
        for (RobotInfo rat:friendlyRats){
            if (rat.getType()==UnitType.RAT_KING){
                king_loc=rat.getLocation();
                return king_loc;
            }
        }
        king_loc=read_loc(0);
        for (int i=0; i<5; i++){
            MapLocation loc = read_loc(2*i);
            if (loc.x==0 && loc.y==0){
                break;
            }
            if (my_loc.distanceSquaredTo(loc)<my_loc.distanceSquaredTo(king_loc)){
                king_loc=loc;
            }
        }
        return king_loc;
    }
    public static boolean was_making_king=false;

    public static boolean new_king() throws GameActionException {
        if (rc.readSharedArray(36)!=0 && rc.readSharedArray(36)!=1) {
            was_making_king=true;
            target = translate_int_to_coords(rc.readSharedArray(rc.readSharedArray(36)));
            target_memory=target;
            if (rc.getLocation().distanceSquaredTo(target)<9){
                pickupdirt(rc);
            }
            if (rc.getLocation().distanceSquaredTo(target) < 3) {
                if (rc.getLocation().distanceSquaredTo(target) < 1 || (rc.canSenseLocation(target) && rc.senseRobotAtLocation(target) != null)) {
                    if (rc.canMove(Direction.CENTER)) {
                        rc.move(Direction.CENTER);

                    }
                    return true;
                }
            }

            if (rc.canSenseLocation(target) && rc.senseMapInfo(target).hasCheeseMine()) {
                if (rc.senseRobotAtLocation(target) != null) {
                    for (Direction dir : adjacentDirections) {
                        if (rc.canSenseLocation(target.add(dir)) && rc.senseRobotAtLocation(target.add(dir)) != null) {
                            target = target.add(dir);
                            target_memory=target;
                        }
                    }
                }
                if (rc.canMove(my_loc.directionTo(target))) {
                    if (rc.canTurn() && my_loc.directionTo(target)!=Direction.CENTER) {
                        rc.turn(my_loc.directionTo(target));
                    }
                    rc.move(my_loc.directionTo(target));
                    get_info();
                    return true;
                }
                if (rc.canMove(my_loc.directionTo(target).rotateLeft())) {
                    if (rc.canTurn() && my_loc.directionTo(target)!=Direction.CENTER) {
                        rc.turn(my_loc.directionTo(target).rotateLeft());
                    }
                    rc.move(my_loc.directionTo(target).rotateLeft());
                    get_info();
                    return true;
                }
                if (rc.canMove(my_loc.directionTo(target).rotateRight())) {
                    if (rc.canTurn() && my_loc.directionTo(target)!=Direction.CENTER) {
                        rc.turn(my_loc.directionTo(target).rotateRight());
                    }
                    rc.move(my_loc.directionTo(target).rotateRight());
                    get_info();
                    return true;
                }
                move_and_turn();
                return true;
            }
            return true;
        }

        return false;
    }

    public static boolean twice=false;

    public static void new_target() throws GameActionException {

        int[] mines=new int[30];
        int num_mines=0;
        MapLocation closest_mine=null;
        int closest=0;
        int closest_dist=Integer.MAX_VALUE;
        for(int i=10;i<30;i++){
            int mine=rc.readSharedArray(i);
            if(mine!=0){
                mines[i-10]=mine;
                num_mines++;
                MapLocation mine_loc=translate_int_to_coords(mine);
                if(my_loc.distanceSquaredTo(mine_loc)<closest_dist) {
                    if ((rc.getRoundNum() % (my_loc.distanceSquaredTo(mine_loc)+1) )< (my_loc.distanceSquaredTo(mine_loc) / 2)) {
                        closest_dist = my_loc.distanceSquaredTo(mine_loc);
                        closest_mine = mine_loc;
                        closest = i - 10;
                    }
                }
            }
            else {
                break;
            }
        }

        if (num_mines>0 && closest_mine!=null && rc.getRoundNum()%13<3) {
            target = closest_mine;
            target_memory = target;
            turns_since_progress = 0;
            last_closest_dist = my_loc.distanceSquaredTo(target);
            old_target = target;
        }
        else {
            Direction dir=my_loc.directionTo(new MapLocation(mapHeight/2,mapHeight/2));
            if(rc.getRoundNum()%4==0){
                dir=dir.rotateLeft();
            }
            else if (rc.getRoundNum()%4==1){
                dir=dir.rotateRight();
            }
            target = edge_from_dir(dir);
            target_memory=target;
            turns_since_progress =0;
            last_closest_dist=my_loc.distanceSquaredTo(target);
            old_target=target;
        }
    }

    public static boolean can_get_target() throws GameActionException {
        if(rc.canSenseLocation(my_loc.add(rc.getDirection()))) {
            if (!rc.senseMapInfo(my_loc.add(rc.getDirection())).isWall()) {
                return true;
            }
        }
        if(rc.canSenseLocation(my_loc.add(rc.getDirection().rotateLeft()))) {
            if (!rc.senseMapInfo(my_loc.add(rc.getDirection().rotateLeft())).isWall()) {
                return true;
            }
        }
        if(rc.canSenseLocation(my_loc.add(rc.getDirection().rotateRight()))) {
            if (!rc.senseMapInfo(my_loc.add(rc.getDirection().rotateRight())).isWall()) {
                return true;
            }
        }
        return false;
    }


    @SuppressWarnings("unused")
    public static void run(RobotController rc) throws GameActionException {

        try {
            if (rc.getID()%10==0){
                digger=true;
            }
            get_info();
            if (isnew) {
                SpawnManager.init(rc);
                turns_since_progress=0;
                init(rc);
                explore_direction=king_loc.directionTo(rc.getLocation());

                Direction dir = rc.getLocation().directionTo(king_loc).opposite();
                init_loc=translate_int_to_coords(rc.readSharedArray(0)).add(dir).add(dir);
                init_set_target();
                target_memory=target;
                if (rc.getRoundNum()>350){
                    new_target();
                }
                old_loc=my_loc;
            }

            // Drop a marker every 5 turns while exploring (not while backtracking)
            // maybeAddMarker();

// If we are backtracking, override target to follow markers in reverse
            //
            // applyBacktrackTargetIfNeeded();

            try {
                if (rc.getRoundNum() < 20) {
                    pickupdirt(rc);
                }


                boolean picked_up = false;

                MapLocation f = my_loc.add(rc.getDirection());
                MapLocation l = my_loc.add(rc.getDirection().rotateLeft());
                MapLocation r = my_loc.add(rc.getDirection().rotateRight());

                boolean fDirt = rc.canSenseLocation(f) && rc.senseMapInfo(f).isDirt();
                boolean lDirt = rc.canSenseLocation(l) && rc.senseMapInfo(l).isDirt();
                boolean rDirt = rc.canSenseLocation(r) && rc.senseMapInfo(r).isDirt();

                // If any of these are off-map, treat as dirt
                if (!rc.canSenseLocation(f)) fDirt = true;
                if (!rc.canSenseLocation(l)) lDirt = true;
                if (!rc.canSenseLocation(r)) rDirt = true;

                if (fDirt && lDirt && rDirt) {
                    if (twice) {
                        pickupdirt(rc);
                        picked_up = true;
                        twice = false;
                    } else {
                        twice = true;
                    }
                }

                if (!picked_up) {
                    Direction[] checks = {
                            Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
                    };

                    if (fDirt) {
                        for (Direction d : checks) {
                            MapLocation adj = f.add(d);
                            if (!rc.canSenseLocation(adj) || rc.senseMapInfo(adj).isWall()) {
                                pickupdirt(rc);
                                break;
                            }
                        }
                    }
                    if (digger) {

                        if (lDirt) {
                            for (Direction d : checks) {
                                MapLocation adj = l.add(d);
                                if (!rc.canSenseLocation(adj) || rc.senseMapInfo(adj).isWall()) {
                                    pickupdirt(rc);
                                    break;
                                }
                            }
                        }

                        if (rDirt) {
                            for (Direction d : checks) {
                                MapLocation adj = r.add(d);
                                if (!rc.canSenseLocation(adj) || rc.senseMapInfo(adj).isWall()) {
                                    pickupdirt(rc);
                                    break;
                                }
                            }
                        }
                    }
                }

            } catch (Exception e) {}



            read_signal();
            determine_state();
            auto_throw();
            new_king();

            switch (state) {
                case explore:
                    explore_state();
                    break;
                case combat:
                    combat_state();
                    break;
                case cat:
                    cat_state();
                    break;
                case carrying:
                    carry_state();
                    break;
                case carried:
                    carried_state();
                    break;
                case thrown:
                    thrown_state();
                    break;
            }

            boolean combat_allowed=can_get_target();
            if(combat_allowed){
                look_for_cheese_and_mine();
            }




            if(rc.canBecomeRatKing() && my_loc.distanceSquaredTo(king_loc)<3){
                rc.becomeRatKing();
            }
            auto_throw();

            if(true) {
                //      signal();
            }

            if(rc.canTransferCheese(king_loc,rc.getRawCheese())){
                king_signal();
                rc.transferCheese(king_loc,rc.getRawCheese());
                new_target();/*
                int dist=Integer.MAX_VALUE;
                if(rc.getID()%2==0){
                for (int mine:mines){
                    MapLocation min=translate_int_to_coords(mine);
                    int dis=min.distanceSquaredTo(edge);
                    if (dis<dist){
                        target=min;
                        dist=dis;
                    }
                }
                }*/
            }

            if(old_target== null || my_loc.distanceSquaredTo(old_target)==0){
                old_target=target;
                turns_since_progress=0;
                last_closest_dist=my_loc.distanceSquaredTo(target);
            }
            else {
                int dist = my_loc.distanceSquaredTo(target);
                if (dist<last_closest_dist) {
                    turns_since_progress=0;
                    last_closest_dist=dist;
                }
                else{
                    turns_since_progress++;
                }
            }

            if (last_mine!=null && king_loc.distanceSquaredTo(last_mine)<9) {
                last_mine=null;
            }

            have_been_carrying= rc.getCarrying()==null ? 0 : have_been_carrying+1;
            last_turn_friend_rats=friendlyRats;
            last_turn_enemy_rats=enemyRats;

            if (rc.readSharedArray(36)>1) {
                if (rc.canBecomeRatKing() && ((rc.getRoundNum() > 200 &&
                        my_loc.distanceSquaredTo(translate_int_to_coords(
                                rc.readSharedArray(rc.readSharedArray(36))))<14) || my_loc.distanceSquaredTo(king_loc)<3)) {
                    rc.becomeRatKing();
                }
            }

            if(rc.getDirt()<10){
                pickupdirt(rc);
            }

            if (rc.isMovementReady()){

                if (target!=null){
                    Direction dir=rc.getDirection();
                    if(rc.canRemoveDirt(my_loc.add(dir))){
                        rc.removeDirt(my_loc.add(dir));
                    }
                    if (rc.canMove(dir)){
                        if (rc.canTurn() && dir!=Direction.CENTER){
                            rc.turn(dir);
                        }
                        rc.move(dir);
                        get_info();
                    }
                    if (rc.canMove(dir.rotateRight())){
                        if (rc.canTurn() && dir!=Direction.CENTER){
                            rc.turn(dir.rotateRight());
                        }
                        rc.move(dir.rotateRight());
                        get_info();
                    }
                    if (rc.canMove(dir.rotateLeft())){
                        if (rc.canTurn() && dir!=Direction.CENTER){
                            rc.turn(dir.rotateLeft());
                        }
                        rc.move(dir.rotateLeft());
                        get_info();
                    }
                    if (rc.canMove(dir.rotateRight().rotateRight())){
                        if (rc.canTurn() && dir!=Direction.CENTER){
                            rc.turn(dir.rotateRight().rotateRight());
                        }
                        rc.move(dir.rotateRight().rotateRight());
                        get_info();
                    }
                    if (rc.canMove(dir.rotateLeft().rotateLeft())){
                        if (rc.canTurn() && dir!=Direction.CENTER){
                            rc.turn(dir.rotateLeft().rotateLeft());
                        }
                        rc.move(dir.rotateLeft().rotateLeft());
                        get_info();
                    }
                }
            }



            if(new_king){
                if (signal_mine!=null) {
                }
            }

            prev_target=target;
            // If we reached the king while backtracking, reset trail
            maybeFinishBacktrack();
            isnew = false;
            if(old_loc!=my_loc){
                last_move_dir=old_loc.directionTo(my_loc);
                old_loc=my_loc;
            }



            try {
                rc.setIndicatorLine(rc.getLocation(), new MapLocation(Math.max(0, Math.min(mapWidth - 1, target.x)),
                        Math.max(0, Math.min(mapHeight - 1, target.y))), 0, 255, 0);
                if (last_mine != null) {
                    rc.setIndicatorLine(rc.getLocation(), last_mine, 255, 0, 0);
                }
            } catch (Exception e) {}

        } catch (GameActionException e) {e.printStackTrace();}
    }
}
