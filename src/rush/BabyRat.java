package rush;

import battlecode.common.*;
import rush.Globals;
import rush.WbugNav;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class BabyRat extends Globals {

    /// ///// M variables/functions /////////////////////

// variables
    public static MapLocation target1;
    public static MapLocation objective;
    public static boolean central=false;
    public static int progress_time=0;
    public static boolean end_explore=false;
    public static int coord_explore=0;
    public static int x_edge=0;
    public static int y_edge=0;
    public static int role; // role of rat to be spawned

// initialize rat ***********************
    public static void initialize_bot() throws GameActionException {
        newbot = false;

        //location info
        xx = rc.getLocation().x;
        yy = rc.getLocation().y;
        height = rc.getMapHeight();
        width = rc.getMapWidth();
        quadrant = which_quadrant(xx, yy, width, height);
        subquadrant = which_subquadrant(xx, yy, width, height, quadrant);
        center=new MapLocation(width/2,height/2);
        r = rc.getRoundNum();
        myId = rc.getID();
        myTeam = rc.getTeam();
        opponentTeam = myTeam.opponent();
        center = new MapLocation(width / 2, height / 2);
        role=rc.getRoundNum()-1;

        exploration_targets(width, height, quadrant); //determine the initial targets of the rats
        rc.setIndicatorLine(rc.getLocation(),objective,0,100,0);

    } // end initialize

// set exploration targets  ***********************
public static void exploration_targets(int w,int h,int q) throws GameActionException {
        if (q == 1) {exploration1(w,h); }
        if (q == 2)  {exploration2(w,h);  }
        if (q == 3)  {exploration3(w,h);}
        if (q == 4)  {exploration4(w,h);}
    }

// in quadrant 1 set exploration targets ***********************
public static void exploration1(int w, int h) throws GameActionException {
    // central cycle targets
    int[][] central_targets =  {{w / 2 + 3, h / 2 + 3}, {w / 2 - 3, h / 2 + 3},
            {w / 2 - 3, h / 2-3}, {w / 2 + 3, h / 2 - 3}};

    // parameter targets, counterclockwise
    int[][] targets = {{0, 0}, {w / 4, 0}, {w / 2, 0}, {(3 * w) / 4, 0},
            {w - 1, 0}, {w - 1, h / 4}, {w - 1, h / 2}, {w - 1, (3 * h) / 4},
            {w - 1, h - 1}, {(3 * w) / 4, h - 1}, {w / 2, h - 1}, {w / 4, h - 1},
            {0, h - 1}, {0, (3 * h) / 4}, {0, h / 2}, {0, h / 4}};

    // assign targets
        switch (role) {
            case 1: // point 1
            case 2:
                objective = new MapLocation(targets[7][0], targets[7][1]);
                x_edge=4; y_edge=0;
                break;
            case 3: // point 2 cycle
            case 4:
                objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 5:  // point 3
            case 6:
                objective = new MapLocation(targets[9][0], targets[9][1]);
                x_edge=0; y_edge=4;
                break;
            case 7:  // point 4 cycle
            case 8:
                objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 9:  // point 5
            case 10:
                objective = new MapLocation(targets[5][0], targets[5][1]);
                x_edge=4; y_edge=0;
                break;
            case 11:  // point 6
            case 12:
                objective = new MapLocation(targets[11][0], targets[11][1]);
                x_edge=0; y_edge=4;
                break;
            case 13:  // point 7
            case 14:
                objective = new MapLocation(targets[3][0], targets[3][1]);
                x_edge=0; y_edge=-4;
                break;
            case 15:  // point 8
            case 16:
                objective = new MapLocation(targets[13][0], targets[13][1]);
                x_edge=-4; y_edge=0;
                break;
            case 17:  // point 9 small optional
            case 18:
                if (w>29){objective = new MapLocation(targets[1][0], targets[1][1]);
                x_edge=0; y_edge=-4;}
                break;
            case 19:  // point 10  small optional
            case 20:
                if (h>29){objective = new MapLocation(targets[15][0], targets[15][1]);
                x_edge=-4; y_edge=0;}
                break;
            case 21:  // point  11 optional
            case 22:
                if (w>45){objective = new MapLocation(targets[2][0], targets[2][1]);
                    x_edge=0; y_edge=-4;}
                else{objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 23:  // point 12 optional
            case 24:
                if (w>45){objective = new MapLocation(targets[10][0], targets[10][1]);
                    x_edge=0; y_edge=4;}
                else{objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 25:  // point 13 optional
            case 26:
                if (h>45){objective = new MapLocation(targets[6][0], targets[6][1]);
                    x_edge=4; y_edge=0;}
                else{objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 27:  // point 14 optional
            case 28:
                if (h>45){objective = new MapLocation(targets[14][0], targets[14][1]);
                    x_edge=-4; y_edge=0; central=true;}
                break;
            default: // if r>28, send rat to center
                objective = new MapLocation(central_targets[0][0], central_targets[0][1]);
                x_edge=0; y_edge=0; central=true;
        }
    }

    // in quadrant 2 set exploration targets ***********************
    public static void exploration2(int w, int h) throws GameActionException {

        // central cycle targets
        int[][] central_targets =  {{w / 2 + 3, h / 2 + 3}, {w / 2 - 3, h / 2 + 3},
                {w / 2 - 3, h / 2-3}, {w / 2 + 2, h / 2 - 3}};

        // parameter targets, counterclockwise
        int[][] targets = {{0, 0}, {w / 4, 0}, {w / 2, 0}, {(3 * w) / 4, 0},
                {w - 1, 0}, {w - 1, h / 4}, {w - 1, h / 2}, {w - 1, (3 * h) / 4},
                {w - 1, h - 1}, {(3 * w) / 4, h - 1}, {w / 2, h - 1}, {w / 4, h - 1},
                {0, h - 1}, {0, (3 * h) / 4}, {0, h / 2}, {0, h / 4}};

        // assign targets
        switch (role) {
            case 1: // point 1
            case 2:
                objective = new MapLocation(targets[11][0], targets[11][1]);
                x_edge=0; y_edge=4;
                break;
            case 3: // point 2 circle
            case 4:
                objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 5:  // point 3
            case 6:
                objective = new MapLocation(targets[13][0], targets[13][1]);
                x_edge=-4; y_edge=0;
                break;
            case 7:  // point 4 circle
            case 8:
                objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 9:  // point 5
            case 10:
                objective = new MapLocation(targets[9][0], targets[9][1]);
                x_edge=0; y_edge=4;
                break;
            case 11:  // point 6
            case 12:
                objective = new MapLocation(targets[15][0], targets[15][1]);
                x_edge=-4; y_edge=0;
                break;
            case 13:  // point 7
            case 14:
                objective = new MapLocation(targets[7][0], targets[7][1]);
                x_edge=4; y_edge=0;
                break;
            case 15:  // point 8
            case 16:
                objective = new MapLocation(targets[1][0], targets[1][1]);
                x_edge=4; y_edge=-4;
                break;
            case 17:  // point 9 small optional
            case 18:
                if (h>29){objective = new MapLocation(targets[5][0], targets[5][1]);
                x_edge=0; y_edge=0;}
                break;
            case 19:  // point 10 small optional
            case 20:
                if (w>29){objective = new MapLocation(targets[3][0], targets[3][1]);
                x_edge=0; y_edge=-4;}
                break;
            case 21:  // point  11 optional
            case 22:
                if (w>45){objective = new MapLocation(targets[10][0], targets[10][1]);
                x_edge=0; y_edge=4;}
                else{objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 23:  // point 12 optional
            case 24:
                if (w>45){objective = new MapLocation(targets[2][0], targets[2][1]);
                    x_edge=0; y_edge=0;}
                else{objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                    x_edge=0; y_edge=-4; central=true;}
                break;
                case 25:  // point 13 optional
            case 26:
                if (h>45){objective = new MapLocation(targets[6][0], targets[6][1]);
                x_edge=4; y_edge=0;}
                else{objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 27:  // point 14 optional
            case 28:
                if (h>45){objective = new MapLocation(targets[14][0], targets[14][1]);
                    x_edge=-4; y_edge=0;}
                else{objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            default: // if r>28, send rat to center
                objective = new MapLocation(central_targets[1][0], central_targets[1][1]);
                x_edge=0; y_edge=0; central=true;
        }
    }


// in quadrant 3 set exploration targets ***********************
public static void exploration3(int w, int h) throws GameActionException {

    // central cycle targets
    int[][] central_targets =  {{w / 2 + 3, h / 2 + 3}, {w / 2 - 3, h / 2 + 3},
            {w / 2 - 3, h /2- 3}, {w / 2 + 3, h / 2 - 3}};

    // parameter targets, counterclockwise
    int[][] targets = {{0, 0}, {w / 4, 0}, {w / 2, 0}, {(3 * w) / 4, 0},
            {w - 1, 0}, {w - 1, h / 4}, {w - 1, h / 2}, {w - 1, (3 * h) / 4},
            {w - 1, h - 1}, {(3 * w) / 4, h - 1}, {w / 2, h - 1}, {w / 4, h - 1},
            {0, h - 1}, {0, (3 * h) / 4}, {0, h / 2}, {0, h / 4}};

    // assign targets
    switch (role) {
        case 1: // point 1
        case 2:
            objective = new MapLocation(targets[15][0], targets[15][1]);
            x_edge=-4; y_edge=0;
            break;
        case 3: // point 2 circle
        case 4:
            objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
            x_edge=0; y_edge=0; central=true;
            break;
        case 5:  // point 3
        case 6:
            objective = new MapLocation(targets[1][0], targets[1][1]);
            x_edge=0; y_edge=-4;
            break;
        case 7:  // point 4 circle
        case 8:
            objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
            x_edge=0; y_edge=0; central=true;
            break;
        case 9:  // point 5
        case 10:
            objective = new MapLocation(targets[13][0], targets[13][1]);
            x_edge=-4; y_edge=0;
            break;
        case 11:  // point 6
        case 12:
            objective = new MapLocation(targets[3][0], targets[3][1]);
            x_edge=0; y_edge=-4;
            break;
        case 13:  // point 7
        case 14:
            objective = new MapLocation(targets[11][0], targets[11][1]);
            x_edge=0; y_edge=4;
            break;
        case 15:  // point 8
        case 16:
            objective = new MapLocation(targets[5][0], targets[5][1]);
            x_edge=4; y_edge=0;
            break;
        case 17:  // point 9 small optional
        case 18:
            if (w>29){objective = new MapLocation(targets[9][0], targets[9][1]);
            x_edge=0; y_edge=4;}
            break;
        case 19:  // point 10 small optional
        case 20:
            if (h>29){objective = new MapLocation(targets[7][0], targets[7][1]);
            x_edge=4; y_edge=0;}
            break;
        case 21:  // point  11 optional
        case 22:
            if (w>45){objective = new MapLocation(targets[10][0], targets[10][1]);
                x_edge=0; y_edge=4;}
            else{objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
                x_edge=0; y_edge=0; central=true;}
            break;
        case 23:  // point 12 optional
        case 24:
            if (w>45){objective = new MapLocation(targets[2][0], targets[2][1]);
            x_edge=0; y_edge=-4;}
            else{objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
                x_edge=0; y_edge=0; central=true;}
            break;
        case 25:  // point 13 optional
        case 26:
            if (h>45){objective = new MapLocation(targets[14][0], targets[14][1]);
                x_edge=-4; y_edge=0;}
            else{objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
                x_edge=0; y_edge=0; central=true;}
            break;
        case 27:  // point 14 optional
        case 28:
            if (h>45){objective = new MapLocation(targets[6][0], targets[6][1]);
            x_edge=4; y_edge=0;}
            else{objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
                x_edge=0; y_edge=0; central=true;}
            break;
        default: // if r>28, send rat to center
            objective = new MapLocation(central_targets[2][0], central_targets[2][1]);
            x_edge=0; y_edge=0; central=true;
    }
}

    // in quadrant 4 set exploration targets ***********************
    public static void exploration4(int w, int h) throws GameActionException {

        // central cycle targets
        int[][] central_targets =  {{w / 2 + 3, h / 2 + 3}, {w / 2 - 2, h / 2 + 3},
                {w / 2 - 3, h / 2-3}, {w / 2 + 3, h / 2 - 3}};

        // parameter targets, counterclockwise
        int[][] targets = {{0, 0}, {w / 4, 0}, {w / 2, 0}, {(3 * w) / 4, 0},
                {w - 1, 0}, {w - 1, h / 4}, {w - 1, h / 2}, {w - 1, (3 * h) / 4},
                {w - 1, h - 1}, {(3 * w) / 4, h - 1}, {w / 2, h - 1}, {w / 4, h - 1},
                {0, h - 1}, {0, (3 * h) / 4}, {0, h / 2}, {0, h / 4}};

        // assign targets
        switch (role) {
            case 1: // point 1
            case 2:
                objective = new MapLocation(targets[3][0], targets[3][1]);
                x_edge=0; y_edge=-4;
                break;
            case 3: // point 2  circle
            case 4:
                objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 5:  // point 3
            case 6:
                objective = new MapLocation(targets[5][0], targets[5][1]);
                x_edge=4; y_edge=0;
                break;
            case 7:  // point 4 circle
            case 8:
                objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                x_edge=0; y_edge=0; central=true;
                break;
            case 9:  // point 5
            case 10:
                objective = new MapLocation(targets[1][0], targets[1][1]);
                x_edge=0; y_edge=-4;
                break;
            case 11:  // point 6
            case 12:
                objective = new MapLocation(targets[7][0], targets[7][1]);
                x_edge=4; y_edge=0;
                break;
            case 13:  // point 7
            case 14:
                objective = new MapLocation(targets[15][0], targets[15][1]);
                x_edge=-4; y_edge=0;
                break;
            case 15:  // point 8
            case 16:
                objective = new MapLocation(targets[9][0], targets[9][1]);
                x_edge=0; y_edge=4;
                break;
            case 17:  // point 9 small optional
            case 18:
                if (h>29){objective = new MapLocation(targets[13][0], targets[13][1]);
                x_edge=-4; y_edge=0;}
                break;
            case 19:  // point 10 small optional
            case 20:
                if (w>29){objective = new MapLocation(targets[11][0], targets[11][1]);
                x_edge=0; y_edge=4;}
                break;
            case 21:  // point  11 optional
            case 22:
                if (w>45){objective = new MapLocation(targets[2][0], targets[2][1]);
                x_edge=0; y_edge=-4;}
                else{objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 23:  // point 12 optional
            case 24:
                if (w>45){objective = new MapLocation(targets[10][0], targets[10][1]);
                    x_edge=0; y_edge=4;}
                else{objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 25:  // point 13 optional
            case 26:
                if (h>45){objective = new MapLocation(targets[14][0], targets[14][1]);
                x_edge=-4; y_edge=0;}
                else{objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            case 27:  // point 14 optional
            case 28:
                if (h>45){objective = new MapLocation(targets[6][0], targets[6][1]);
                    x_edge=4; y_edge=0;}
                else{objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                    x_edge=0; y_edge=0; central=true;}
                break;
            default: // if r>28, send rat to center
                objective = new MapLocation(central_targets[3][0], central_targets[3][1]);
                x_edge=0; y_edge=0; central=true;
        }
    }




// stop exploring ***********************
    public static void stop_explore(MapLocation obj) throws GameActionException {
        MapLocation bot_location=rc.getLocation();
        int xnow = rc.getLocation().x;
        int ynow = rc.getLocation().y;
            // end central exploration
            if (central == true) {
                if (bot_location.distanceSquaredTo(center) <= 2) {
                    end_explore = true;
                }
                ;
            } else {
                if (coord_explore == 0) {
                    if (xnow + x_edge - obj.x <= 2) {
                        end_explore = true;
                    }
                } else {
                    if (ynow + y_edge - obj.y <= 2) {
                        end_explore = true;
                    } // end vertical exploration
                }
            }
    }

// rats cycle around the center counterclockwise
    public static void cycle(int w, int h) throws GameActionException {
        // central cycle targets
        int[][] central_targets =  {{w / 2 + 3, h / 2 + 3}, {w / 2 - 3, h / 2 + 3},
                {w / 2 - 3, h /2- 3}, {w / 2 + 3, h / 2 - 3}};
        int xnow = rc.getLocation().x;
        int ynow = rc.getLocation().y;
        int quadrantnow = which_quadrant(xnow, ynow, width, height);
        int goal=(quadrantnow +2)%4;
        objective=new MapLocation(central_targets[goal][0], central_targets[goal][1]);
    } // end cycle

//////// end M variables/functions /////////////////////
///
///
    static final Random rng = new Random(6147);

    public static MapLocation target;

    public static MapLocation explore_target;
    public static MapLocation cur_target;
    public static boolean isnew = true;

    public static int[] mines = new int[100];
    public static int minecount = 0;
    public static int[] global;

    public static RobotInfo[] ri;
    public static MapInfo[] mi;

    public static int cheese_value = 100;
    public static int cat_value = -150;
    public static int king_value = 100;
    public static int explore_value = 20;

    // ===== RAT COMBAT =====
    static final int ATTACK_HEURISTIC = 300;
    static final int FRIENDLY_ADVANTAGE_BONUS = 150;
    static final int ENEMY_OVERWHELM_PENALTY = 100;
    // ======================

    public static int lastExploreDist = Integer.MAX_VALUE;
    public static int exploreStuckTurns = 0;
    public static boolean returning = false;

    public static int nearby_friendly_rats;

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

    static final Map<MapLocation, Integer> location_heuristic = new HashMap<>();

    /*
    public static void init(RobotController rc) {
        try {
            global = new int[64];
            for (int i = 0; i < 64; i++) {
                global[i] = rc.readSharedArray(i);
            }
            readTarget(rc);
        } catch (Exception e) {}
    }



    // ===== SEEN MAP UPDATE =====
    public static void updateSeenRobots(RobotController rc, RobotInfo[] robots) {
        int round = rc.getRoundNum();

        for (RobotInfo info : robots) {
            seenRobots.put(info.ID, new SeenRobot(info, round));
        }

        Iterator<Map.Entry<Integer, SeenRobot>> it = seenRobots.entrySet().iterator();
        while (it.hasNext()) {
            if (round - it.next().getValue().lastSeenRound > SEEN_TTL) {
                it.remove();
            }
        }
    }

    // ===== RAT COMBAT =====
    public static boolean tryAttackEnemyRat(RobotController rc, RobotInfo[] robots)
            throws GameActionException {
        if (!rc.isActionReady()) return false;

        for (RobotInfo info : robots) {
            if (info.type == UnitType.BABY_RAT &&
                    info.team != rc.getTeam() &&
                    rc.canAttack(info.location)) {
                rc.attack(info.location);
                return true;
            }
        }
        return false;
    }

    public static void ratCombatHeuristic(RobotController rc, RobotInfo[] robots) {
        int friendly = 0, enemy = 0;

        for (RobotInfo info : robots) {
            if (info.type == UnitType.BABY_RAT) {
                if (info.team == rc.getTeam()) friendly++;
                else enemy++;
            }
        }

        if (enemy > 0 && friendly > enemy) {
            int advantage = friendly - enemy;

            for (RobotInfo info : robots) {
                if (info.type == UnitType.BABY_RAT && info.team != rc.getTeam()) {
                    int dist = rc.getLocation().distanceSquaredTo(info.location);
                    if (dist == 0) dist = 1;

                    int value = ATTACK_HEURISTIC + FRIENDLY_ADVANTAGE_BONUS * advantage
                            - ENEMY_OVERWHELM_PENALTY * enemy;
                    location_heuristic.put(info.location, value / dist);
                }
            }
        }
    }

    // ===== RATNAP WEAKER ENEMY =====
    public static boolean tryRatnapEnemy(RobotController rc, RobotInfo[] robots)
            throws GameActionException {
        if (!rc.isActionReady()) return false;
        int myHealth = rc.getHealth();

        for (RobotInfo info : robots) {
            if (info.type == UnitType.BABY_RAT &&
                    info.team != rc.getTeam() &&
                    info.health < myHealth &&
                    rc.canCarryRat(info.location)) {
                rc.carryRat(info.location);
                return true;
            }
        }
        return false;
    }

    // ===== THROW CARRIED RAT =====
    public static boolean tryThrowRat(RobotController rc) throws GameActionException {
        if (!rc.isActionReady() || !(rc.getCarrying()!=null)) return false;

        RobotInfo[] nearby = rc.senseNearbyRobots(24, null); // sense in throw range
        Direction bestDir = null;
        int bestScore = Integer.MIN_VALUE;

        for (Direction dir : directions) {
            int score = 0;
            MapLocation loc = rc.getLocation();
            for (int i = 1; i <= 3; i++) { // simulate throw 3 squares
                loc = loc.add(dir);
                for (RobotInfo r : nearby) {
                    if (r.location.equals(loc) && r.type == UnitType.BABY_RAT) {
                        if (r.team != rc.getTeam()) score += 10; // enemy
                        else score -= 8;                          // ally
                    }
                }
            }
            if (score > bestScore) {
                bestScore = score;
                bestDir = dir;
            }
        }

        if (bestDir != null && bestScore > 0) { // only throw if net positive
            if (rc.canTurn()){
                rc.turn(bestDir);
            }
            rc.throwRat();
            rc.setIndicatorString("threw rat " + bestDir + " score=" + bestScore);
            return true;
        }
        return false;
    }

    // ===== SQUEAKS =====
    public static void hear_squeak(RobotController rc) throws GameActionException {
        Message[] msgs = rc.readSqueaks(-1);
        if (msgs.length == 0) return;

        int newmine = 0;
        for (Message msg : msgs) {
            if (msg.getBytes() == 12345) nearby_friendly_rats++;
            else newmine = msg.getBytes();
        }

        if (newmine == 0) return;

        for (int mine : mines) if (mine == newmine) return;
        for (int i = 63; i > 4; i--) if (global[i] == newmine) return;

        mines[minecount++] = newmine;
        rc.squeak(newmine);
    }

    public static void found_mine(RobotController rc, MapLocation loc) throws GameActionException {
        int newmine = translate_coords_to_int(loc);

        for (int mine : mines) if (mine == newmine) return;
        for (int i = 63; i > 4; i--) {
            if (global[i] == newmine) return;
            if (global[i] == 0) break;
        }

        mines[minecount++] = newmine;
        rc.squeak(newmine);
        returning = true;
    }

    public static void found_cat(RobotController rc, RobotInfo info) throws GameActionException {
        MapLocation catloc = info.location;
        for (Direction dir : directions) {
            MapLocation loc = rc.getLocation().add(dir);
            if (rc.canPlaceCatTrap(loc) )rc.placeCatTrap(loc);
        }

        int cat = translate_coords_to_int(catloc) + (1 << 30);

        for (int i = 1; i < 5; i++) {
            if (global[i] == cat % 1024) return;
            if (global[i] == 0) break;
        }

        rc.squeak(cat);
        location_heuristic.put(catloc, cat_value / rc.getLocation().distanceSquaredTo(catloc));
    }

    public static void found_cheese(RobotController rc, MapLocation loc) throws GameActionException {
        if (rc.canPickUpCheese(loc)) {
            rc.pickUpCheese(loc);
            return;
        }
        location_heuristic.put(loc, cheese_value / rc.getLocation().distanceSquaredTo(loc));
    }

    public static void return_heuristic(RobotController rc) throws GameActionException {
        MapLocation best = translate_int_to_coords(global[5]);
        for (int i = 6; i < 10; i++) {
            if (global[i] != 0) {
                MapLocation k = translate_int_to_coords(global[i]);
                if (rc.getLocation().distanceSquaredTo(k)
                        < rc.getLocation().distanceSquaredTo(best)) {
                    best = k;
                }
            }
        }
        location_heuristic.put(best, rc.getRawCheese() * king_value);
    }

    public static void checkExploreProgress(RobotController rc) throws GameActionException {
        if (explore_target == null) return;
        int d = rc.getLocation().distanceSquaredTo(explore_target);

        if (d < lastExploreDist) {
            lastExploreDist = d;
            exploreStuckTurns = 0;
        } else if (++exploreStuckTurns >= 5) {
            readTarget(rc);
            lastExploreDist = Integer.MAX_VALUE;
            exploreStuckTurns = 0;
        }
    }

    public static void set_target(RobotController rc) throws GameActionException {
        location_heuristic.clear();
        ri = rc.senseNearbyRobots();
        mi = rc.senseNearbyMapInfos();

        updateSeenRobots(rc, ri);

        // ===== RATNAP WEAKER ENEMY =====
        if (tryRatnapEnemy(rc, ri)) return;
        // ===== IMMEDIATE ATTACK =====
        if (tryAttackEnemyRat(rc, ri)) return;

        for (RobotInfo info : ri) {
            if (info.type == UnitType.CAT) found_cat(rc, info);
            if (info.type == UnitType.RAT_KING &&
                    rc.canTransferCheese(info.location, rc.getRawCheese())) {
                rc.transferCheese(info.location, rc.getRawCheese());
            }
        }

        ratCombatHeuristic(rc, ri);

        for (MapInfo info : mi) {
            if (info.hasCheeseMine()) found_mine(rc, info.getMapLocation());
            if (info.isPassable() && info.getCheeseAmount() > 0)
                found_cheese(rc, info.getMapLocation());
        }

        return_heuristic(rc);
        location_heuristic.put(explore_target, explore_value);

        cur_target = explore_target;
        int best = Integer.MIN_VALUE;
        for (var e : location_heuristic.entrySet()) {
            if (e.getValue() > best) {
                best = e.getValue();
                cur_target = e.getKey();
            }
        }
    }

    public static void readTarget(RobotController rc) throws GameActionException {
        if (rc.getRoundNum() > first_phase && myId % 16 != 0) {
            int cnt = 0;
            for (int i = 63; i > 9; i--) {
                if (rc.readSharedArray(i) != 0) cnt++;
                else break;
            }
            explore_target = translate_int_to_coords(rc.readSharedArray(63 - myId % cnt));
        } else {
            explore_target = translate_int_to_coords(global[0]);
        }
        lastExploreDist = Integer.MAX_VALUE;
        exploreStuckTurns = 0;
    }

    public static void move(RobotController rc, MapLocation target) throws GameActionException {
        Navigator.moveTo(target);
    }

    public static void pickupdirt(RobotController rc) throws GameActionException {
        for (Direction d : directions) {
            MapLocation l = rc.getLocation().add(d);
            if (rc.canRemoveDirt(l)) {
                rc.removeDirt(l);
                break;
            }
        }
    }
    */

    public static void init(RobotController rc) {
        target=exploreLocations[rc.getRoundNum()%8];
    }

    public static boolean turned_left=false;

// bugnav to the target turning left and right to get better vision around
    public static void move_and_turn(MapLocation target) throws GameActionException {
        //rc.setIndicatorLine(rc.getLocation(),target,0,100,0);
        // Navigation.bfs(target,rc);
        // comment out above or below  /*

        Direction dir = WbugNav.moveTo(target);
        if (dir!= null && dir!= Direction.CENTER) {
            if(rc.canTurn() && rc.getDirection()!=dir) {
                rc.turn(dir);
            }
            rc.move(dir);

            if(rc.canTurn()) {
                if (!turned_left) rc.turn(dir.rotateLeft().rotateLeft());
                else rc.turn(dir.rotateRight().rotateRight());
                turned_left = !turned_left;
            }
        } // */ comment out up to here
    }

    public static int turns_since_progress_made=0;
    public static int dist_to_target=Integer.MAX_VALUE;
    public static int progress_magic_number=3;

    // sets the target to a new location if the current one is found or not making progress
    /*
    public static void set_target() throws GameActionException{

        // if found and close enough to target
        if (rc.getLocation().distanceSquaredTo(target)<3) {
            if (target==exploreLocations[rc.getRoundNum()%8])
                target=exploreLocations[(rc.getRoundNum()+1)%8];
            else target=exploreLocations[rc.getRoundNum()%8];
            return;
        }

        // get if we made progress
        if (rc.getLocation().distanceSquaredTo(target)<dist_to_target) {
            turns_since_progress_made=0;
        }
        else turns_since_progress_made++;
        dist_to_target=rc.getLocation().distanceSquaredTo(target);

        if (turns_since_progress_made >= progress_magic_number) {
            turns_since_progress_made=0;
            dist_to_target=Integer.MAX_VALUE;
            if (target==exploreLocations[rc.getRoundNum()%8])
                target=exploreLocations[(rc.getRoundNum()+1)%8];
            else target=exploreLocations[rc.getRoundNum()%8];
        }

    }
    */

    public static RobotInfo[] friendlyRats;
    public static RobotInfo[] enemyRats;
    public static RobotInfo[] Cats;
    public static MapInfo[] mapInfos;
    public static Team my_team=rc.getTeam();
    public static void get_info() throws  GameActionException {
        friendlyRats=rc.senseNearbyRobots(-1, my_team);
        enemyRats=rc.senseNearbyRobots(-1, my_team.opponent());
        Cats=rc.senseNearbyRobots(-1, Team.NEUTRAL);

    }
    public static int reached_edge=0;

    public static void set_target() throws GameActionException{

        // no enemy rats
        if (enemyRats.length==0 && king==null) {
            target=oppositeLocation;
            if (rc.canSenseLocation(oppositeLocation)) {
                if (reached_edge == 0) {
                    oppositeLocation=new MapLocation(oppositeLocation.x,mapHeight-oppositeLocation.y);
                    reached_edge=1;
                }
                else{
                    oppositeLocation=new MapLocation(mapWidth-oppositeLocation.x,mapHeight-oppositeLocation.y);
                }
            }
        }
        else{ // enemy rats
            enemy();
        }

    }
    public static MapLocation init_loc;

    public static boolean facing_away(Direction dir1, Direction dir2) throws GameActionException {
        if (Math.abs(dir1.getDirectionOrderNum()-dir2.getDirectionOrderNum())>2){
            return true;
        }
        return false;

    }

    public static MapLocation king=null;


    public static void enemy() throws GameActionException {

        // focus the king
        if (king!=null){
            if (rc.getDirection()!=rc.getLocation().directionTo(king)) {
                if (rc.canTurn()) {
                    rc.turn(rc.getLocation().directionTo(king));
                }
            }
            enemyRats=rc.senseNearbyRobots(-1, my_team.opponent());
        }


        // attack the king
        for (RobotInfo rat:enemyRats){
            if (rat.getType()==UnitType.RAT_KING){
                target=rat.getLocation();
                king=rat.getLocation();
                if (rc.getDirection()!=rc.getLocation().directionTo(target)) {
                    if(rc.canTurn()) {
                        rc.turn(rc.getLocation().directionTo(target));
                    }
                }
                if (rc.canThrowRat() && rc.getDirection()==rc.getLocation().directionTo(target)) {
                    rc.throwRat();
                }
                if (rc.canAttack(king.subtract(rc.getLocation().directionTo(target)))) {
                    rc.attack(king.subtract(rc.getLocation().directionTo(target)),rc.getRawCheese());

                }
                return;
            }
        }

        // attack a rat in range
        for (RobotInfo rat:enemyRats) {
            if (rc.canCarryRat(rat.getLocation())) {
                rc.carryRat(rat.getLocation());
            }
            if (rc.canAttack(rat.getLocation())) {
                rc.attack(rat.getLocation(), rc.getRawCheese());
            }
            if (rc.canCarryRat(rat.getLocation())) {
                rc.carryRat(rat.getLocation());
            }
            if (rc.canThrowRat() && rc.getDirection() == rc.getLocation().directionTo(target)) {
                rc.throwRat();
            }
        }

        // select target
        for(RobotInfo rat:enemyRats){
            if(facing_away(rc.getLocation().directionTo(rat.getLocation()),rat.getDirection())){
                target=rat.getLocation();
                break;
            }
            if (rat.getHealth()+rat.getRawCheeseAmount()<=rc.getHealth()+rc.getRawCheese()){
                target=rat.getLocation();
                break;
            }
            if (friendlyRats.length>=enemyRats.length){
                target=rat.getLocation();
            }
        }
    }

    public static void look_for_cheese() throws GameActionException {
        MapInfo[] info=rc.senseNearbyMapInfos();
        for (MapInfo mi : info) {
            if (mi.getCheeseAmount()>0){
                if (rc.canPickUpCheese(mi.getMapLocation())) {
                    rc.pickUpCheese(mi.getMapLocation());
                    return;
                }
            }
        }
    }

    public static void auto_throw() throws GameActionException {
        if(rc.canThrowRat() && enemyRats.length!=0 && rc.canTurn()) {
            int closest_dist=Integer.MAX_VALUE;
            RobotInfo closest=null;
            for (RobotInfo rat: enemyRats){
                if (rc.getLocation().distanceSquaredTo(rat.getLocation()) < closest_dist) {
                    closest_dist=rc.getLocation().distanceSquaredTo(rat.getLocation());
                    closest=rat;
                }
            }
            rc.turn(rc.getLocation().directionTo(closest.getLocation()));
            if(rc.canThrowRat()) {
                rc.throwRat();
            }
        }
    }

    public static boolean placed_trap=false;
    public static int health_const=30;

//////// RUN ///////////////////////////////
    public static void run(RobotController rc) throws GameActionException {

        if (newbot) {
            initialize_bot();
            init(rc);
            Direction dir = rc.getLocation().directionTo(translate_int_to_coords(rc.readSharedArray(0)));
            init_loc=translate_int_to_coords(rc.readSharedArray(0)).add(dir).add(dir);
        }

        try {
          //  rc.setIndicatorString("target is " + target);
            get_info();
            look_for_cheese();

            set_target();
            if(!rc.isActionReady()){
                for (RobotInfo rat: enemyRats){
                    if (rat.getLocation().isAdjacentTo(rc.getLocation())) {
                        if (rc.getID()<rat.getID()){
                            if (rc.canMove(rc.getLocation().directionTo(rat.getLocation()).opposite())) {
                                rc.move(rc.getLocation().directionTo(rat.getLocation()).opposite());
                            }
                        }
                    }
                }
            }

            auto_throw();

            // get an objective
            if ((end_explore == true)& (central==true)){cycle(width, height);}
            // move towards objective
            if ((end_explore == false)){ move_and_turn(objective);}
            else {
                if (central != true) {move_and_turn(target);}
                if (central == true) {move_and_turn(objective);} // add conditions ????????
            }
            if (end_explore== false) {stop_explore(objective);}

            if (rc.isActionReady()) {
                enemy();
            }






//rc.setIndicatorString("enemy rats is "+enemyRats.length + " king is " + king + "target is " + target);




            /*
            global = new int[64];
            for (int i = 0; i < 64; i++)
                global[i] = rc.readSharedArray(i);

            nearby_friendly_rats = 0;
            hear_squeak(rc);
            checkExploreProgress(rc);

            if (returning) {
                cur_target = translate_int_to_coords(global[5]);
            } else {
                set_target(rc);
            }

            rc.squeak(12345);
            pickupdirt(rc);

            // ==== THROW CARRIED RAT IF IT HITS MORE ENEMIES THAN FRIENDS ====
            tryThrowRat(rc);

            move(rc, cur_target);*/

        } catch (GameActionException e) {e.printStackTrace();}
    }
}
