package learner;

import battlecode.common.*;
import learner.Globals;

public class RatKing extends Globals {

    public static int target = 0;
    public static int random_explore = 10;
    public static int mine_choice = 0;
    public static int nearby_friendly_rats;
    public static int spawn_dir = 0;
    public static int king_counter;
    public static MapLocation kings_goal;
    public static int ratsSpawned = 0;


//////////////////////// M variables/functions /////////////////////////

// initialize king ***********************
    public static void initialize_king() throws GameActionException {
        newbot = false;
        //initialize(rc);
        init(rc);
        rc.writeSharedArray(2,translate_coords_to_int(rc.getLocation()));  // index 2: coords of king1

        // count the number of kings
        king_counter=Math.max(rc.readSharedArray(0)-1,0);
        rc.writeSharedArray(0,king_counter); // index 0: counter
        kings=rc.readSharedArray(1)+1;
        rc.writeSharedArray(1,kings); // index 1: number of kings // ?????????? FIX
        king_num = kings;

        //location info
        height = rc.getMapHeight();
        width = rc.getMapWidth();
        xx = rc.getLocation().x;
        yy = rc.getLocation().y;
        quadrant = which_quadrant(xx, yy, width, height);
        subquadrant = which_subquadrant(xx, yy, width, height, quadrant);
        center=new MapLocation(width/2,height/2);
        myId = rc.getID();
        myTeam = rc.getTeam();
        opponentTeam = myTeam.opponent();
        int r_start = rc.getRoundNum();
        r= rc.getRoundNum();
        if (r==1){king_start=true;}  // this is the starting king g


/*
        // number of rats to have
        int minrats=0; // min number of rats to have
        int size=(height+width)/2;
        if (size<40){ minrats=18;}
        if (39<size & size<50){ minrats=22;}
        if (49<size) { minrats=26;}
        rats=minrats; // starting value of rats
        if (kings>1){ rats=rats +8*(kings-1);} // because making a king destroys 7 rats
*/

        SpawnManager.init(rc);

    } // end initialize

// starting code ***********************
    public static void start_king() throws GameActionException {
        // get constants
        r= rc.getRoundNum();
        price = rc.getCurrentRatCost();
        cheese = rc.getAllCheese();

        // move starting king towards center of quadrant
        if (king_start==true){center_king(width, height);}

    } // end of start

// turn and move towards location
    public static void turn_move_dir(Direction d) throws GameActionException {
        if (d == Direction.CENTER) return;
        if (rc.canMove(d) == true) {
            if (rc.canTurn() == true) {
                rc.turn(d);
                rc.move(d);
            }
        }
    }

// turn and move towards location
    public static void turn_move_loc(MapLocation target) throws GameActionException {
        Direction dir = rc.getLocation().directionTo(target);
        turn_move_dir(dir);
    }

//  move starting king to center of his quadrant
    public static void center_king(int w, int h)throws GameActionException {
        if  (subquadrant!=5){
            switch (quadrant) {
                case 1:
                    kings_goal = new MapLocation(w/4, h/4); break;
                case 2:
                    kings_goal = new MapLocation((3*w)/4, h/4); break;
                case 3:
                    kings_goal = new MapLocation((3*w)/4, (3*h)/4); break;
                case 4:
                    kings_goal = new MapLocation(w/4, (3*h)/4); break;
            }
        turn_move_loc(kings_goal);
        }
    }






// compute number of kings and record in shared array  ************************
// index 0 gives rolling counter, index 1 gives number of kings
    public static void num_kings() throws GameActionException{
        if (king_counter>=1023){
            king_counter%=1024;
            if(king_counter==rc.readSharedArray(0)){
                kings=rc.readSharedArray(1);
                rc.writeSharedArray(0,0);
                king_counter+=kings;
                king_counter%=1024;
            }
            else{
                kings-=1;
                rc.writeSharedArray(1,kings);
                rc.writeSharedArray(0,king_counter+1);
                king_counter+=kings;
                king_counter%=1024;
            }
        }
        if(king_counter==rc.readSharedArray(0)){
            kings=rc.readSharedArray(1);
            rc.writeSharedArray(0,king_counter+1);
            king_counter+=kings;
        }
        else{
            kings-=1;
            rc.writeSharedArray(1,kings);
            rc.writeSharedArray(0,king_counter+1);
            king_counter+=kings;
        }
    } // end of num_kings

//////////////////// end M variables/functions ///////////////////////////

    // GLOBAL ARRAY MEANING, index 0 is target for next rat,
    // 1,2,3,4 are first 4 cats found,
    // 5,6,7,8,9 are the ratkings,
    // 64- is the mines

    /*
    public static void move(RobotController rc) throws GameActionException
    {
        MapLocation myLoc = rc.getLocation();
        MapLocation moveTarget = null;

        // --- if no mines known, go to center ---
        if (rc.readSharedArray(63) == 0){
            moveTarget = new MapLocation(midX1, midY1);
            rc.setIndicatorString("moving to center: "+ moveTarget);
        }
        else {
            // --- if mines known, go to nearest mine ---
            int min_dist = Integer.MAX_VALUE;
            for(int i=63; i>9; i--){
                int code = rc.readSharedArray(i);
                if (code == 0) break;
                MapLocation mineLoc = translate_int_to_coords(code);
                int dist = myLoc.distanceSquaredTo(mineLoc);
                if(dist < min_dist){
                    min_dist = dist;
                    moveTarget = mineLoc;
                }
            }
            rc.setIndicatorString("moving to mine: " + moveTarget);
        }

        // --- check for nearby cats and move away ---
        for (int i = 1; i <= 4; i++) {
            int catCode = rc.readSharedArray(i);
            if (catCode == 0) continue;
            MapLocation catLoc = translate_int_to_coords(catCode);

            int dist = myLoc.distanceSquaredTo(catLoc);
            if (dist <= 16) { // if cat is within 4 tiles
                // move away from cat by creating a vector in opposite direction
                int dx = myLoc.x - catLoc.x;
                int dy = myLoc.y - catLoc.y;
                MapLocation away = new MapLocation(myLoc.x + dx, myLoc.y + dy);

                // prefer away target if passable
                if (rc.canMove(myLoc.directionTo(away))) {
                    moveTarget = away;
                    rc.setIndicatorString("moving away from cat: " + catLoc);
                    break; // prioritize first threatening cat
                }
            }
        }

        // --- finally move to chosen target ---
        rc.setIndicatorString("move target is null: " + moveTarget);
        if (moveTarget != null) {
            rc.setIndicatorString("moving to : " + moveTarget);
            Navigator.moveTo(moveTarget);
        }
    }

    public static void hear_squeak(RobotController rc) throws GameActionException{
        Message[] msgs = rc.readSqueaks(-1);
        if(msgs.length == 0) return;

        int content = 0;
        for (Message msg : msgs) {
            if (msg.getBytes() == 12345){
                nearby_friendly_rats++;
            } else {
                content = msg.getBytes();
            }
        }
        if (content == 0) return;

        if (content < Math.pow(2,30)) {
            // mine
            for (int i = 63; i > 4; i--) {
                if (rc.readSharedArray(i) == content) return;
                if (rc.readSharedArray(i) == 0) {
                    rc.writeSharedArray(i, content);
                    rc.setIndicatorString("learned of mine at " +translate_int_to_coords(content));
                    num_mines+=1;
                    return;
                }
            }
        } else {
            // cat
            content = content % 1024;
            for (int i = 1; i < 5; i++) {
                try {
                    MapLocation catloc = translate_int_to_coords(content);
                    MapLocation oldloc = translate_int_to_coords(rc.readSharedArray(i));
                    if (catloc.distanceSquaredTo(oldloc) <= 4) return;
                    if (catloc.distanceSquaredTo(oldloc) <= 16 || rc.readSharedArray(i) == 0) {
                        rc.writeSharedArray(i, content);
                        rc.setIndicatorString("learned of cat at " + catloc);
                        return;
                    }
                } catch (Exception e) {}
            }
        }
    }

    public static void signaltarget(RobotController rc) throws GameActionException {
        int code = translate_coords_to_int(exploreLocations[target]);
        rc.writeSharedArray(0, code);
        target += 1;
        target %= 8;
    }

    public static void spawnrat(RobotController rc) throws GameActionException{
        try {
            Direction direc = rc.getLocation().directionTo(exploreLocations[target]);
            MapLocation loca = rc.getLocation().add(direc).add(direc).add(direc);
            MapLocation loca2 = rc.getLocation().add(direc).add(direc);

            if (rc.canBuildRat(loca)) {
                signaltarget(rc);
                rc.buildRat(loca);
                return;
            } else if (rc.canBuildRat(loca2)) {
                signaltarget(rc);
                rc.buildRat(loca2);
                return;
            }
        } catch (Exception e){}

        for (Direction dir : Direction.values()) {
            MapLocation loc = rc.getLocation().add(dir).add(dir);
            if (rc.canBuildRat(loc)) {
                rc.buildRat(loc);
                break;
            }
        }
    }

    public static void put_self_in_array(RobotController rc) throws GameActionException {
        int myCode = translate_coords_to_int(rc.getLocation());

        for (int i = 5; i < 10; i++) {
            int v = rc.readSharedArray(i);
            if (v != 0 && translate_int_to_coords(v).distanceSquaredTo(rc.getLocation()) <= 4) {
                rc.writeSharedArray(i, myCode);
                return;
            }
        }

        for (int i = 5; i < 10; i++) {
            if (rc.readSharedArray(i) == 0) {
                rc.writeSharedArray(i, myCode);
                return;
            }
        }
    }
    */

    /*
    public static void spawnrat(RobotController rc) throws GameActionException{
        if (rc.getCurrentRatCost()<40+rc.getAllCheese()/1000){
            spawn_dir=rc.getRoundNum()%8;
            int intial_dir=spawn_dir;
            while(true){
                if(rc.canBuildRat(selfloc.add(adjacentDirections[spawn_dir]).add(adjacentDirections[spawn_dir]))){
                    rc.buildRat(selfloc.add(adjacentDirections[spawn_dir]).add(adjacentDirections[spawn_dir]));
                    spawn_dir+=2;
                    spawn_dir%=8;
                    return;
                }
                spawn_dir++;
                spawn_dir%=8;
                if (spawn_dir==intial_dir){return;}

            }
        }
    }
    */
    public static RobotInfo[] friendlyRats;
    public static RobotInfo[] enemyRats;
    public static RobotInfo[] Cats;
    public static MapLocation enemy_loc = null;
    public static MapInfo[] mapInfos;

    public static int cost_const = 25;

    public static void get_info() throws GameActionException {
        friendlyRats = rc.senseNearbyRobots(-1, myTeam);
        enemyRats = rc.senseNearbyRobots(-1, opponentTeam);
        Cats = rc.senseNearbyRobots(-1, Team.NEUTRAL);
        mapInfos = rc.senseNearbyMapInfos();
        selfloc = rc.getLocation();
    }

    public static void spawnrat(RobotController rc) throws GameActionException {
        boolean surge = false;
        if (rc.getRoundNum() > 70 && rc.getRoundNum() < 90 &&
                (rc.getLocation().distanceSquaredTo(new MapLocation(mapWidth / 2, mapHeight / 2)))
                        < mapWidth + mapHeight) {
            surge = true;
        }

        int num_kings = rc.readSharedArray(1);

        if (!(rc.getCurrentRatCost() < cost_const + 5 * num_kings + (surge ? 20 : 0) + (rc.readSharedArray(36) > 1 ? 20 : 0)
                || (enemyRats.length > friendlyRats.length && rc.getAllCheese() > 50))
                && rc.getGlobalCheese() < 2100) {
            return;
        }

        Direction dir = SpawnManager.get_spawn_direction();

        if (enemyRats.length != 0) {
            int bestDist = Integer.MAX_VALUE;
            RobotInfo closest = null;
            MapLocation myLoc = rc.getLocation();

            for (RobotInfo ri : enemyRats) {
                int d = myLoc.distanceSquaredTo(ri.location);
                if (d < bestDist) {
                    bestDist = d;
                    closest = ri;
                }
            }

            if (closest != null) {
                dir = rc.getLocation().directionTo(closest.location);
            }
        }

        if (rc.getGlobalCheese() > 50 * num_kings) {
            if (rc.canBuildRat(selfloc.add(dir).add(dir))) {
                rc.buildRat(selfloc.add(dir).add(dir));
            } else if (rc.canBuildRat(selfloc.add(dir).add(dir.rotateLeft()))) {
                rc.buildRat(selfloc.add(dir).add(dir.rotateLeft()));
            } else if (rc.canBuildRat(selfloc.add(dir).add(dir.rotateRight()))) {
                rc.buildRat(selfloc.add(dir).add(dir.rotateRight()));
            } else {
                for (Direction d : Direction.values()) {
                    for (Direction d2 : Direction.values()) {
                        if (rc.canBuildRat(selfloc.add(d).add(d2))) {
                            rc.buildRat(selfloc.add(d).add(d2));
                            return;
                        }
                    }
                }
            }
        }
    }

/////////////// RUN //////////////////////////////////////////
    public static void run(RobotController rc) throws GameActionException {

        try {
            if (newbot){
                cost_const = 25 + (mapHeight + mapWidth) / 10;
                initialize_king();
            }
            get_info();
            start_king();

            spawnrat(rc);

            for (RobotInfo info: rc.senseNearbyRobots()){
                if (info.getType()== UnitType.CAT){
                    if(rc.canMove(rc.getLocation().directionTo(info.getLocation()).opposite())){
                        rc.move(rc.getLocation().directionTo(info.getLocation()).opposite());
                    }
                }
            }

            for (RobotInfo info: rc.senseNearbyRobots(-1,rc.getTeam().opponent())){
                    if(rc.canMove(rc.getLocation().directionTo(info.getLocation()).opposite())){
                        rc.move(rc.getLocation().directionTo(info.getLocation()).opposite());
                }
            }


         // put_self_in_array(rc);
         //   signaltarget(rc);

            /* M
            if (quadrant==1){
                spawn1(xx,yy, width, height);}
            if (quadrant==2){spawn2(xx,yy, width, height);}
            if (quadrant==3){spawn3(xx,yy, width, height);}
            if (quadrant==4){spawn4(xx,yy, width, height);}
            else
            {spawnrat(rc);}
             */

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
