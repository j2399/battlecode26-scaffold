package econ5;

import battlecode.common.*;

public class RatKing extends Globals {

    public static boolean isnew=true;
    public static int target =0;
    public static int random_explore=10;
    public static int mine_choice=0;
    public static int spawn_dir=0;
    public static double[] mineWeights = new double[20];
    public static boolean mineWeightsInit = false;

    public static void init_mine_weights() throws GameActionException {
        MapLocation kingLoc = rc.getLocation();

        int bytes1=Clock.getBytecodesLeft();

        for (int i = 0; i < 20; i++) {
            if (mines[i] == 0) {
                mineWeights[i] = 0;
                continue;
            }

            MapLocation mineLoc = translate_int_to_coords(mines[i]);

            int dx = Math.abs(mineLoc.x - kingLoc.x);
            int dy = Math.abs(mineLoc.y - kingLoc.y);

            // Chebyshev distance (grid-consistent)
            int dist = Math.max(dx, dy);

            // Further mines start with higher weight
            mineWeights[i] = Math.pow(dist, 2.5);
        }

        int bytes2=Clock.getBytecodesLeft();
      //  System.out.println("rolled for loop bytes: " + (bytes1-bytes2) +" \n");

        mineWeightsInit = true;
    }

    public static void choose_target_mine_weighted() throws GameActionException {
        double bestWeight = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;

        for (int i = 0; i < 20; i++) {
            if (mines[i] == 0) continue;

            if (mineWeights[i] > bestWeight) {
                bestWeight = mineWeights[i];
                bestIndex = i;
            }
        }

        if (bestIndex == -1) return;

        // Decay like spawn weights
        mineWeights[bestIndex] /= 2.0;

        // Publish chosen mine index
        rc.writeSharedArray(35, bestIndex);
    }

    public static void handle_starving() throws GameActionException{
        if (rc.getAllCheese()==0){
            boolean may_dis=false;
            boolean wrote=false;
            int id=rc.getID()%1000;
            int lowest_health=Integer.MAX_VALUE;
            for (int i=0; i<5; i++){
                int read=rc.readSharedArray(63-2*i);
                if(read!=0 && read!=id) {
                    may_dis=true;
                    int other_health=rc.readSharedArray(63-2*i-1);
                    lowest_health=Math.min(lowest_health,other_health);
                }
                if ((read== 0 || read==id) && !wrote){
                    rc.writeSharedArray(63-2*i,id);
                    rc.writeSharedArray(63-2*i-1,rc.getHealth());
                    wrote=true;
                }
            }
            if(may_dis && rc.getHealth()<lowest_health){
                for (int i=0; i<10; i++){
                    rc.writeSharedArray(63-i,0);
                }
                rc.disintegrate();
            }
        }
        else {
            for (int i=0; i<10; i++){
                rc.writeSharedArray(63-i,0);

            }
        }
    }



    public static void spawnrat(RobotController rc) throws GameActionException{


        boolean surge=false;
        if (rc.getRoundNum()>70 && rc.getRoundNum()<90 &&
                (rc.getLocation().distanceSquaredTo(new MapLocation(mapWidth/2,mapHeight/2)))
                        < mapWidth+mapHeight) {surge=true;}

        int num_kings=0;
        for (int i=0; i<5; i++) {
            MapLocation loc = read_loc(2 * i);
            if (loc.x == 0 && loc.y == 0) {
                break;
            }
            num_kings++;
        }

        if (!(rc.getCurrentRatCost()<cost_const+5*num_kings+(surge ? 20 : 0) + (rc.readSharedArray(36) > 1 ? 20 : 0)|| (enemyRats.length>friendlyRats.length &&
                rc.getAllCheese()>50)) && rc.getGlobalCheese()<2100) {
            return;
        }

        Direction dir = SpawnManager.get_spawn_direction();


        if(enemyRats.length!=0){
            int bestDist=Integer.MAX_VALUE;
            RobotInfo closest=null;
            MapLocation myLoc = rc.getLocation();

            for (RobotInfo ri : enemyRats) {
                int d = myLoc.distanceSquaredTo(ri.location);
                if (d < bestDist) {
                    bestDist = d;
                    closest = ri;
                }
            }

            dir=rc.getLocation().directionTo(closest.location);
        }
        if (rc.getGlobalCheese()>50*num_kings) {
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

    public static RobotInfo[] friendlyRats;
    public static RobotInfo[] enemyRats;
    public static RobotInfo[] Cats;
    public static Team my_team=rc.getTeam();
    public static MapLocation enemy_loc=null;
    public static MapInfo[] mapInfos;

    public static void get_info() throws  GameActionException {
        friendlyRats=rc.senseNearbyRobots(-1, my_team);
        enemyRats=rc.senseNearbyRobots(-1, my_team.opponent());
        Cats=rc.senseNearbyRobots(-1, Team.NEUTRAL);
        mapInfos=rc.senseNearbyMapInfos();
        selfloc=rc.getLocation();
    }

    public static int cost_const=25+(mapHeight+mapWidth)/10;

    public static int[] mines= new int[20];


    public static void init_mines() throws GameActionException {
        for(int i=10;i<30;i++){
            int mine=rc.readSharedArray(i);
            if(mine!=0){
                mines[i-10]=mine;
            }
        }
    }

    static int getFurthestFromCenterMine(int dist) throws GameActionException {
        MapLocation center = new MapLocation(mapWidth / 2, mapHeight / 2);

        int bestMine = 10;                 // shared array index [10..29]
        int bestDist = Integer.MIN_VALUE;  // maximize distance from center

        for (int i = 10; i < 30; i++) {
            int raw = rc.readSharedArray(i);
            if (raw == 0) continue;

            MapLocation mine = translate_int_to_coords(raw);

            // skip mines too close to any king (reuse your filter)
            if (kingNear(mine, dist)) continue;

            int d = center.distanceSquaredTo(mine);
            if (d > bestDist) {
                bestDist = d;
                bestMine = i;
            }
        }

        return bestMine;
    }


    public static void update_mine_weights_on_new_mine() throws GameActionException {
        MapLocation kingLoc = rc.getLocation();

        for (int i = 0; i < 20; i++) {
            // New mine: exists but has no weight yet
            if (mines[i] != 0 && mineWeights[i] == 0) {
                MapLocation mineLoc = translate_int_to_coords(mines[i]);

                int dx = Math.abs(mineLoc.x - kingLoc.x);
                int dy = Math.abs(mineLoc.y - kingLoc.y);

                // Chebyshev distance (grid-consistent)
                int dist = Math.max(dx, dy);
                if (dist == 0) dist = 1;

                mineWeights[i] = Math.pow(dist, 2.5);
            }

            // Slot cleared (optional safety)
            if (mines[i] == 0) {
                mineWeights[i] = 0;
            }
        }
    }



    public static void read_king_signal() throws GameActionException {
        Message[] msgs=rc.readSqueaks(-1);
        for (Message msg : msgs) {
            // bits 0-2 symmetry
            // bits 3-12 cheese/mine loc
            int cheese = (msg.getBytes()/8)%1024;
            for(int i=0;i<20;i++){
                if (mines[i]==0){
                    mines[i]=cheese;
                    rc.writeSharedArray(i+10,cheese);

                    break;
                }
                if(translate_int_to_coords(mines[i]).distanceSquaredTo(translate_int_to_coords(cheese))<49){
                    break;
                }
            }

            // bits 13-22 enemy king loc
            int enemy_king = (msg.getBytes()%(8*1024*1024)/(8*1024));
            if ( enemy_king!=0){
                rc.writeSharedArray(34,enemy_king);
            }
            if (msg.getBytes()/(8*1024*1024)==1){
                cheese_new_king=translate_int_to_coords(cheese);
            }
            //  rc.setIndicatorString(" new king is " + cheese_new_king);
        }
    }
    public static MapLocation cheese_new_king;

    public static int mine_counter=0;


    // 0-9 king locs,  10-29 mine locs, 30/31 enemy attacking king, 32/33 being rushed, 33 sym, 34 is enemy king, 35 is mine counter
    public static void writes() throws GameActionException {
        for (int i = 0; i < 5; i++) {
            MapLocation loc=read_loc(2*i);
            if (loc.x==0 && loc.y==0) {
                write_loc(2*i,rc.getLocation());

                break;
            }
            if (loc.distanceSquaredTo(rc.getLocation()) < 9) {
                write_loc(2*i,rc.getLocation());
                break;
            }
        }
        if(enemy_loc!=null && enemyRats.length!=0){
            write_loc(30,enemy_loc);
        }
        else {
            rc.writeSharedArray(30,0);
        }

    }

    public static void enemy_response() throws GameActionException {
        if (enemyRats.length!=0 /*&& rc.getCurrentRatCost()>=cost_cosnt && rc.getAllCheese()>30*/) {
            int dist = Integer.MAX_VALUE;
            for (RobotInfo rat : enemyRats) {
                if (rc.getLocation().distanceSquaredTo(rat.getLocation()) < dist) {
                    dist = rc.getLocation().distanceSquaredTo(rat.getLocation());
                    enemy_loc = rat.getLocation();
                }
            }

            // rc.setIndicatorLine(rc.getLocation(), enemy_loc, 0, 0, 255);
            int num_kings=0;
            for (int i=0; i<5; i++) {
                MapLocation loc = read_loc(2 * i);
                if (loc.x == 0 && loc.y == 0) {
                    break;
                }
                num_kings++;
            } // chnaged

            if (rc.getAllCheese() > 50*num_kings && (num_kings==1 || rc.getCurrentRatCost()<cost_const+10+0.005*rc.getGlobalCheese()+5*num_kings)) {
                Direction dir = rc.getLocation().directionTo(enemy_loc);
                MapLocation loc = rc.getLocation().add(dir);
                if (rc.canBuildRat(loc.add(dir))) {
                    rc.buildRat(loc.add(dir));
                } else if (rc.canBuildRat(loc.add(dir.rotateLeft()))) {
                    rc.buildRat(loc.add(dir.rotateLeft()));
                } else if (rc.canBuildRat(loc.add(dir.rotateRight()))) {
                    rc.buildRat(loc.add(dir.rotateRight()));
                }

                if (rc.isActionReady()){
                    dir = rc.getLocation().directionTo(enemy_loc).rotateRight();
                    loc = rc.getLocation().add(dir);
                    if (rc.canBuildRat(loc.add(dir))) {
                        rc.buildRat(loc.add(dir));
                    } else if (rc.canBuildRat(loc.add(dir.rotateLeft()))) {
                        rc.buildRat(loc.add(dir.rotateLeft()));
                    } else if (rc.canBuildRat(loc.add(dir.rotateRight()))) {
                        rc.buildRat(loc.add(dir.rotateRight()));
                    }
                }
                if (rc.isActionReady()){
                    dir = rc.getLocation().directionTo(enemy_loc).rotateLeft();
                    loc = rc.getLocation().add(dir);
                    if (rc.canBuildRat(loc.add(dir))) {
                        rc.buildRat(loc.add(dir));
                    } else if (rc.canBuildRat(loc.add(dir.rotateLeft()))) {
                        rc.buildRat(loc.add(dir.rotateLeft()));
                    } else if (rc.canBuildRat(loc.add(dir.rotateRight()))) {
                        rc.buildRat(loc.add(dir.rotateRight()));
                    }
                }
                if (rc.isActionReady()){
                    dir = rc.getLocation().directionTo(enemy_loc).rotateRight().rotateRight();
                    loc = rc.getLocation().add(dir);
                    if (rc.canBuildRat(loc.add(dir))) {
                        rc.buildRat(loc.add(dir));
                    } else if (rc.canBuildRat(loc.add(dir.rotateLeft()))) {
                        rc.buildRat(loc.add(dir.rotateLeft()));
                    } else if (rc.canBuildRat(loc.add(dir.rotateRight()))) {
                        rc.buildRat(loc.add(dir.rotateRight()));
                    }
                }
                if (rc.isActionReady()){
                    dir = rc.getLocation().directionTo(enemy_loc).rotateLeft().rotateLeft();
                    loc = rc.getLocation().add(dir);
                    if (rc.canBuildRat(loc.add(dir))) {
                        rc.buildRat(loc.add(dir));
                    } else if (rc.canBuildRat(loc.add(dir.rotateLeft()))) {
                        rc.buildRat(loc.add(dir.rotateLeft()));
                    } else if (rc.canBuildRat(loc.add(dir.rotateRight()))) {
                        rc.buildRat(loc.add(dir.rotateRight()));
                    }
                }

            }

            if (rc.canMove(rc.getLocation().directionTo(enemy_loc).opposite())) {
                if (rc.canTurn()) {
                    rc.turn(rc.getLocation().directionTo(enemy_loc).opposite());
                }
                rc.move(rc.getLocation().directionTo(enemy_loc).opposite());
            } else {
                for (RobotInfo info : enemyRats) {
                    if (rc.canMove(rc.getLocation().directionTo(info.getLocation()).opposite())) {
                        if (rc.canTurn()) {
                            rc.turn(rc.getLocation().directionTo(info.getLocation()).opposite());
                        }
                        rc.move(rc.getLocation().directionTo(info.getLocation()).opposite());
                    }
                }
            }

        }
    }

    public static void cat_response() throws GameActionException {
        // rc.setIndicatorString("" + rc.isMovementReady());
        for (RobotInfo info: rc.senseNearbyRobots(-1, Team.NEUTRAL)){


            if (info.getType()== UnitType.CAT){

                MapLocation me = rc.getLocation();
                MapLocation catLoc = info.getLocation();

                int distCatMe = me.distanceSquaredTo(catLoc);
                if (distCatMe == 0) return;

                Direction toCat = me.directionTo(catLoc);
                if (toCat == Direction.CENTER) return;

                Direction left  = toCat.rotateLeft();
                Direction right = toCat.rotateRight();

// Count escape options BEFORE placing
                int escapeBefore = 0;
                for (Direction d : adjacentDirections) {
                    if (rc.canMove(d)) escapeBefore++;
                }

// Build candidates IN FRONT OF ME toward the cat.
// (Near-me first is usually best to immediately block a chase lane.)
                MapLocation[] candidates = new MapLocation[] {
                        me.add(toCat),
                        me.add(toCat).add(toCat),

                        // widen the block (still in front of me)
                        me.add(toCat).add(left),
                        me.add(toCat).add(right),
                        me.add(toCat).add(toCat).add(left),
                        me.add(toCat).add(toCat).add(right),

                        // deeper
                        me.add(toCat).add(toCat).add(toCat),
                        me.add(toCat).add(toCat).add(toCat).add(left),
                        me.add(toCat).add(toCat).add(toCat).add(right),
                };

                for (MapLocation p : candidates) {
                    if (!rc.canPlaceDirt(p)) continue;

                    // --- MUST BE BETWEEN ME AND CAT ---
                    // "Between" constraint: p must be closer to cat than me->cat distance,
                    // and closer to me than me->cat distance.
                    if (me.distanceSquaredTo(p) >= distCatMe) continue;
                    if (catLoc.distanceSquaredTo(p) >= distCatMe) continue;

                    // --- DON'T CORNER MYSELF ---
                    // If p is an adjacent tile I can currently move into and I have low mobility, skip it.
                    if (me.isAdjacentTo(p)) {
                        Direction into = me.directionTo(p);
                        if (into != Direction.CENTER && rc.canMove(into) && escapeBefore <= 2) {
                            continue;
                        }
                    }

                    // Extra safety: if I already have low mobility, don't place dirt very close to me.
                    if (escapeBefore <= 2 && me.distanceSquaredTo(p) <= 2) {
                        continue;
                    }

                    rc.placeDirt(p);
                    break;
                }




                Direction dir= rc.getLocation().directionTo(info.location);
                /*if (rc.canPlaceDirt(rc.getLocation().add(dir).add(dir))){
                    rc.placeDirt(rc.getLocation().add(dir).add(dir));
                }*/
                // ===== RUN AWAY FROM CAT WITHOUT GETTING CORNERED =====
                if (rc.isMovementReady()) {

                    final MapLocation here = rc.getLocation();
                    catLoc = info.getLocation();
                    final Direction catDir = info.getDirection();

                    // Current distance to cat
                    final int curDist = here.distanceSquaredTo(catLoc);

                    Direction bestMove = null;
                    int bestScore = Integer.MIN_VALUE;

                    // Helper: count impassable cardinal neighbors of a tile (higher => more "cage-like")
                    java.util.function.IntUnaryOperator impAdj = (int packed) -> {
                        // packed = (x<<16) | y
                        int x = (packed >>> 16);
                        int y = (packed & 0xFFFF);
                        MapLocation loc = new MapLocation(x, y);

                        int cnt = 0;

                        try {

                            MapLocation n = loc.add(Direction.NORTH);
                            if (rc.canSenseLocation(n) && !rc.senseMapInfo(n).isPassable()) cnt++;

                            MapLocation s = loc.add(Direction.SOUTH);
                            if (rc.canSenseLocation(s) && !rc.senseMapInfo(s).isPassable()) cnt++;

                            MapLocation e = loc.add(Direction.EAST);
                            if (rc.canSenseLocation(e) && !rc.senseMapInfo(e).isPassable()) cnt++;

                            MapLocation w = loc.add(Direction.WEST);
                            if (rc.canSenseLocation(w) && !rc.senseMapInfo(w).isPassable()) cnt++;

                        } catch (Exception e) {}


                        return cnt;
                    };

                    // Optional helper: edge penalty (avoid corners/edges)
                    java.util.function.IntUnaryOperator edgePenalty = (int packed) -> {
                        int x = (packed >>> 16);
                        int y = (packed & 0xFFFF);

                        int dx = Math.min(x, mapWidth - 1 - x);
                        int dy = Math.min(y, mapHeight - 1 - y);

                        // If close to an edge, penalty increases; corners (dx=0 or dy=0) worst
                        int minToEdge = Math.min(dx, dy);
                        if (minToEdge <= 0) return 2000;  // on border
                        if (minToEdge == 1) return 800;
                        if (minToEdge == 2) return 250;
                        return 0;
                    };

                    for (Direction d : Globals.adjacentDirections) {
                        if (!rc.canMove(d)) continue;

                        MapLocation nxt = here.add(d);

                        // Base: run away (maximize distance gain)
                        int nd = nxt.distanceSquaredTo(catLoc);
                        int score = (nd - curDist) * 200;     // big weight: increase distance

                        // Strongly avoid tiles that look like a pocket/cage
                        int packed = (nxt.x << 16) | (nxt.y & 0xFFFF);
                        int cage = impAdj.applyAsInt(packed);
                        score -= cage * 900;                  // 2 impassables is already bad

                        // Avoid hugging edges/corners
                        score -= edgePenalty.applyAsInt(packed);

                        // Prefer passable surroundings (more escape routes next turn)
                        int escapeRoutes = 0;
                        for (Direction d2 : Globals.adjacentDirections) {
                            MapLocation nn = nxt.add(d2);
                            if (rc.canSenseLocation(nn) && rc.senseMapInfo(nn).isPassable()) escapeRoutes++;
                        }
                        score += escapeRoutes * 120;


                        // Optional: avoid moving into cat vision cone if you have isWithinDistanceSquared(angle) available.
                        // If you do NOT have it in this ruleset, delete this block.
                        try {
                            boolean entersVision = nxt.isWithinDistanceSquared(catLoc, 21, catDir, Math.PI);
                            if (entersVision) score -= 4000;
                        } catch (Throwable t) {
                            // ignore if method not available
                        }

                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = d;
                        }
                    }

                    if (bestMove != null) {
                        if (rc.canTurn()) rc.turn(bestMove);
                        rc.move(bestMove);
                    } else {
                        // Fallback: if no move, at least face away from cat (prevents accidental "CENTER" turn issues elsewhere)
                        Direction away = here.directionTo(catLoc).opposite();
                        if (rc.canTurn() && away != Direction.CENTER) rc.turn(away);
                    }
                }


            }
        }
    }

    public static void standard()  throws GameActionException {
        Direction dir=null;

        if (dir==null && mines.length!=0){
            int i;
            if (rc.getID() < 1000) {
                i = getFurthestFromCenterMine(0);
            } else {
                i = getClosestSafeMine(0);
                int j = getClosestMiddleMine(33);

                if (rc.getLocation().distanceSquaredTo(translate_int_to_coords(j))<rc.getLocation().distanceSquaredTo(translate_int_to_coords(i))*4) {
                    i=j;
                }
            }
            int mine = rc.readSharedArray(i);
            if (mine!=0){
                dir=WbugNav.moveTo(translate_int_to_coords(mine));
            }
        }

        MapInfo[] infos=rc.senseNearbyMapInfos();
        MapLocation closestCheese = null;
        int bestDist = Integer.MAX_VALUE;

// 1) find closest cheese
        for (MapInfo info : infos) {
            if (info.getCheeseAmount() > 0) {
                int d = rc.getLocation().distanceSquaredTo(info.getMapLocation());
                if (d < bestDist) {
                    bestDist = d;
                    closestCheese = info.getMapLocation();
                }
            }
        }

// 2) act on closest cheese
        if (closestCheese != null) {
            if (rc.canPickUpCheese(closestCheese)) {
                rc.pickUpCheese(closestCheese);
            } else {
                dir = WbugNav.moveTo(closestCheese);
            }
        }


        if(dir==null) {
            for (MapInfo info : mapInfos) {
                if (info.hasCheeseMine()) {
                    dir = WbugNav.moveTo(info.getMapLocation());
                    break;
                }
            }
        }
        if(rc.getHealth()<UnitType.RAT_KING.getHealth()/4){
            dir=WbugNav.moveTo(init_loc);
        }
        if(dir==null) {
            dir = WbugNav.moveTo(new MapLocation(mapWidth / 2, mapHeight / 2));
        }

        if (dir== null || dir== Direction.CENTER) {
            return;
        }
        if(rc.canMove(dir)){
            if(rc.canTurn()){
                rc.turn(dir);
            }
            move_dir=dir;
            rc.move(dir);
        }



    }

    public static Direction move_dir;
    public static Direction last_move_dir=Direction.CENTER;
    public static MapLocation old_loc;

    public enum State{
        standard,
        combat,
        cat
    }

    public static RatKing.State state;

    public static void determine_state() throws GameActionException {
        if(enemyRats.length!=0 && enemyRats.length>Cats.length+1){
            state=RatKing.State.combat;
        }
        else if(Cats.length!=0){
            state=RatKing.State.cat;
        }
        else {
            state=RatKing.State.standard;
        }
    }

    public static MapLocation init_loc;

    static boolean kingNear(MapLocation mine, int distSq) throws GameActionException {
        for (int i = 0; i < 5; i++) {
            MapLocation king = read_loc(2 * i);
            if (king.x == 0 && king.y == 0) break;

            if (king.distanceSquaredTo(mine) <= distSq) {
                return true;
            }
        }
        return false;
    }

    static int getClosestSafeMine(int dist) throws GameActionException {
        MapLocation here = rc.getLocation();

        int bestMine = 10;
        int bestDist = Integer.MAX_VALUE;

        // adjust if your mine list size differs
        for (int i = 10; i < 30; i++) {
            int raw = rc.readSharedArray(i);
            if (raw == 0) continue;

            MapLocation mine = translate_int_to_coords(raw);

            // skip mines too close to any king
            if (kingNear(mine, dist)) continue;

            int d = here.distanceSquaredTo(mine);
            if (d < bestDist) {
                bestDist = d;
                bestMine = i;
            }
        }

        return bestMine; // null if none found
    }

    static int getClosestMiddleMine(int dist) throws GameActionException {
        MapLocation here = rc.getLocation();
        MapLocation center=new MapLocation(mapWidth/2,mapHeight/2);

        int bestMine = 10;
        int bestDist = Integer.MAX_VALUE;

        // adjust if your mine list size differs
        for (int i = 10; i < 30; i++) {
            int raw = rc.readSharedArray(i);
            if (raw == 0) continue;

            MapLocation mine = translate_int_to_coords(raw);

            // skip mines too close to any king
            if (kingNear(mine, dist)) continue;



            int d = center.distanceSquaredTo(mine);
            if (d < bestDist) {
                bestDist = d;
                bestMine = i;
            }
        }

        return bestMine; // null if none found
    }

    static int getFurthesstSafeMine(int dist) throws GameActionException {
        MapLocation here = rc.getLocation();

        int bestMine = 10;
        int bestDist = Integer.MIN_VALUE;

        // adjust if your mine list size differs
        for (int i = 10; i < 30; i++) {
            int raw = rc.readSharedArray(i);
            if (raw == 0) continue;

            MapLocation mine = translate_int_to_coords(raw);

            // skip mines too close to any king
            if (kingNear(mine, dist)) continue;

            int d = here.distanceSquaredTo(mine);
            if (d > bestDist) {
                bestDist = d;
                bestMine = i;
            }
        }

        return bestMine; // null if none found
    }




    public static void set_new_king_index() throws GameActionException {

        // Count known kings (shared array 0..9 holds up to 5 king locs as pairs)
        int num_kings = 0;
        for (int i = 0; i < 5; i++) {
            MapLocation k = read_loc(2 * i);
            if (k.x == 0 && k.y == 0) break;
            num_kings++;
        }

        // If we already have max kings (or late-game cap), disable “make new king”
        if (num_kings == 5 || (num_kings >= 2 && rc.getRoundNum() >= 1050)) {
            rc.writeSharedArray(36, 1);
            return;
        }

        int cheeseThresh = 1800;
        int costThresh   = (int)(30 * Math.pow(num_kings, 1.6));

        // If a requested "cheese_new_king" is set and there is just one king, prefer that mine index
        // Assumes cheese_new_king is a public static MapLocation (or similar) in your class.
        if (cheese_new_king != null && (num_kings==1 ||  (rc.getAllCheese() > cheeseThresh &&
                rc.getCurrentRatCost() >= costThresh))) {

            // If a king is already near that location (using global king locs), clear request
            for (int i = 0; i < 5; i++) {
                MapLocation k = read_loc(2 * i);
                if (k.x == 0 && k.y == 0) break;

                if (k.distanceSquaredTo(cheese_new_king) <= 33) { // "near" threshold; tune if needed
                    cheese_new_king = null;
                    rc.writeSharedArray(36, 1);
                    return;
                }
            }

            // Find the mine index (10..29) whose location matches/near cheese_new_king
            int bestMineIdx = -1;
            int bestDist = Integer.MAX_VALUE;
            int i=0;
            for (int mine: mines) {
                int raw = mine;
                if (raw == 0) continue;

                MapLocation m = translate_int_to_coords(raw);
                int d = m.distanceSquaredTo(cheese_new_king);

                // exact/near match: prefer smallest distance
                if (d < bestDist) {
                    bestDist = d;
                    bestMineIdx = i+10;
                }
                i++;
            }

            // If we found a sufficiently close mine slot, publish it
            // (49 = within 7 tiles; adjust threshold if you want tighter mapping)
            if (bestMineIdx != -1 && bestDist <= 49) {
                rc.writeSharedArray(36, bestMineIdx);
                return;
            } else {
                // Request doesn't map to a known mine; fall back to normal logic
                // do NOT clear cheese_new_king here; let it persist until a mine is known or a king appears
            }
        }

        int roundThresh  = -100 + 250 * num_kings;
        cheeseThresh = 1800;
        costThresh   = (int)(30 * Math.pow(num_kings, 1.6));

        rc.setIndicatorString(
                "NK gate | " +
                        "round: " + rc.getRoundNum() + "/" + roundThresh + " | " +
                        "cheese: " + rc.getAllCheese() + "/" + cheeseThresh + " | " +
                        "cost: " + rc.getCurrentRatCost() + "/" + costThresh + " | " +
                        "numK=" + num_kings + " 36 is " + rc.readSharedArray(36)
        );

        // ---------- Existing gating logic ----------
        if (rc.readSharedArray(36) < 2) {

            if (rc.getRoundNum() < roundThresh ||
                    rc.getAllCheese() < cheeseThresh ||
                    rc.getCurrentRatCost() < costThresh) {

                rc.writeSharedArray(36, 1);
                return;
            }
        }


        MapLocation firstKing = read_loc(10);
        if (firstKing.x == 0 && firstKing.y == 0) {
            return;
        }

        int mine = getClosestMiddleMine(80);
        MapLocation mineLoc = translate_int_to_coords(rc.readSharedArray(mine));

        if (rc.getLocation().distanceSquaredTo(mineLoc) < 9) {
            rc.writeSharedArray(36, 1);
        } else {
            rc.writeSharedArray(36, mine);
        }

        // rc.setIndicatorString("mine " + mine + " at " + mineLoc);
    }


    // 0 is self, 1 is enemy if being attacked, 2 is self, 3 is enemy king if not 0
    @SuppressWarnings("unused")
    public static void run(RobotController rc) throws GameActionException {
        try {
            if (isnew){
                rc.writeSharedArray(36,1);
                init_mines();
                init_loc = rc.getLocation();
                init(rc);
                SpawnManager.init(rc);
                isnew=false;
                enemy_loc=null;
                writes();
                old_loc=rc.getLocation();
            }
            get_info();
            enemy_loc=null;
            read_king_signal();
            update_mine_weights_on_new_mine();


            if (!mineWeightsInit) {
                init_mine_weights();
            }

            choose_target_mine_weighted();

            determine_state();
            switch (state) {
                case combat:  enemy_response();  break;
                case cat:    cat_response();  break;
                case standard:  standard();  break;
            }
            writes();
            spawnrat(rc);
            if(rc.canPickUpCheese(rc.getLocation())){
                rc.pickUpCheese(rc.getLocation());
            }
            // add adj

            int num_dirt=0;

            for (Direction dir: adjacentDirections){
                if (rc.canSenseLocation(rc.getLocation().add(dir).add(dir))) {
                    if (rc.senseMapInfo(rc.getLocation().add(dir).add(dir)).isDirt()) {
                        num_dirt++;
                    }
                }
                if (rc.canSenseLocation(rc.getLocation().add(dir).add(dir.rotateRight()))) {
                    if (rc.senseMapInfo(rc.getLocation().add(dir).add(dir.rotateRight())).isDirt()) {
                        num_dirt++;
                    }
                }
            }
            if (num_dirt>3 || rc.getRoundNum()<30) {

                for (Direction dir : adjacentDirections) {
                    if (rc.canRemoveDirt(rc.getLocation().add(dir).add(dir))) {
                        rc.removeDirt(rc.getLocation().add(dir).add(dir));
                        break;
                    }
                    if (rc.canRemoveDirt(rc.getLocation().add(dir).add(dir.rotateRight()))) {
                        rc.removeDirt(rc.getLocation().add(dir).add(dir.rotateRight()));
                        break;
                    }
                }
            }

            Direction moveDir = rc.getDirection();
            MapLocation here = rc.getLocation();

            if(Cats.length==0) {

                if (rc.isActionReady()) {
                    MapLocation[] targets = new MapLocation[]{
                            here.add(moveDir).add(moveDir), // straight ahead
                            here.add(moveDir).add(moveDir.rotateRight()),
                            here.add(moveDir).add(moveDir.rotateLeft()),
                            here.add(moveDir.rotateLeft()).add(moveDir.rotateLeft()),
                            here.add(moveDir.rotateRight()).add(moveDir.rotateRight())
                    };

                    for (MapLocation loc : targets) {
                        if (!rc.canSenseLocation(loc)) continue;
                        if (!rc.senseMapInfo(loc).isDirt()) continue;
                        if (!rc.canRemoveDirt(loc)) continue;

                        rc.removeDirt(loc);
                        break; // remove ONE tile per turn
                    }
                }


                moveDir = rc.getDirection();
                here = rc.getLocation();

                if (rc.isActionReady()) {
                    MapLocation[] targets = new MapLocation[]{
                            here.add(moveDir).add(moveDir), // straight ahead
                            here.add(moveDir).add(moveDir.rotateRight()),
                            here.add(moveDir).add(moveDir.rotateLeft()),
                            here.add(moveDir.rotateLeft()).add(moveDir.rotateLeft()),
                            here.add(moveDir.rotateRight()).add(moveDir.rotateRight())
                    };

                    for (MapLocation loc : targets) {
                        if (!rc.canSenseLocation(loc)) continue;
                        if (!rc.senseMapInfo(loc).isDirt()) continue;
                        if (!rc.canRemoveDirt(loc)) continue;

                        rc.removeDirt(loc);
                        break; // remove ONE tile per turn
                    }
                }

            }

            if(rc.getHealth()<100 && (enemyRats.length>0 || Cats.length>0 || rc.getHealth()<30)){
                for (int i=0; i<5; i++) {
                    MapLocation loc = read_loc(2 * i);

                    if (rc.getLocation().distanceSquaredTo(loc) < 16) {
                        rc.writeSharedArray(2 * i, 0);
                        rc.writeSharedArray(2 * i + 1, 0);
                    }
                }
            }



            String str="";
            for (int i=0; i<10; i++){
                str+=" king " + i + " is at " + translate_int_to_coords(rc.readSharedArray(i)) + "\n";
            }

            set_new_king_index();
            handle_starving();
            if(old_loc!=rc.getLocation()){
                last_move_dir=old_loc.directionTo(rc.getLocation());
                old_loc=rc.getLocation();
            }

            try {
                for (int mine : mines) {
                    if (mine != 0) {
                        rc.setIndicatorLine(rc.getLocation(), translate_int_to_coords(mine), 0, 255, 0);
                    }
                }
                if (rc.readSharedArray(34) != 0) {
                    rc.setIndicatorLine(rc.getLocation(), translate_int_to_coords(rc.readSharedArray(34)), 255, 0, 0);
                }
            } catch (Exception e) {}


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
