package learner;

import battlecode.common.*;
import learner.FastSet;


// height, width, center, list of cengral destinations, list of parameter destinations

// is_new=true,
// initialize: number of rats, number of rounds, number of kings, number of known mines, amount of global cheese, amount of dirt, broadcasted info, past path, health, amount of local cheese,
//game mode, direction, destination


/*
Adapted from camel_case's 2024 submission:
https://github.com/jmerle/battlecode-2024
 */
public class Globals {
    public static RobotController rc;


//////////// M variables/functions ////////////////////////////
    static int r;
    static int height;
    static int width;
    static int size; // center
    static int xx; // initial xcoord of bot
    static int yy; // initial ycoord of bot
    static int quadrant; // initial quadrant of bot
    static int subquadrant;
    static MapLocation center;
    static boolean king_start=false;  // is this the starting king

    static int cheese; // global cheese
    static int price; // cost of new bot

    static int kings; // number of kings
    static int cheese_reserve=500; // treshold for spawning rats
    static int more_kings=80; // treshold of rounds after which we could add a king
    static int king_num; // number of the king (first, etc)
    // static int add_king=0; // values 0-9, #rats when we want to add king

    static int rats; //


    static boolean newbot=true;

// returns in which quadrant is (x,y) in a wxh map; 1=SW, 2=SE, 3=NE, 4=NW
    public static int which_quadrant(int x,int y,int w,int h) throws GameActionException {
        if (x<(w/2) & y<(h/2)){return 1;  }
        if (x>=(w/2) & y<(h/2)){ return 2; }
        if (x>=(w/2) & y>=(h/2)){ return 3; }
        if (x<(w/2) & y>=(h/2)){  return 4; }
        return 1;
    }

// returns in which subquadrant is (x,y) in a wxh map; 1=SW, 2=SE, 3=NE, 4=NW, 5=center
    public static int which_subquadrant(int x,int y,int w,int h,int q) throws GameActionException {
        int s=0;
        switch(q) {
            case 1:
                if ((w/6 < x)& (x <w/3) & (h/6<y) & (y<h/3)){s=5;}
                else{
                    if ((x<w/4) & (y<h/4)){s=1;}
                    if ((x>=w/4) & (y<h/4)){s=2;}
                    if ((x>=w/4) & (y>=h/4)){s=3;}
                    if ((x<w/4) & (y>=h/4)){s=4;}
                }
                break;
            case 2:
                if (((2*w)/3 < x)& (x <(5*w)/6) & (h/6<y) & (y<h/3)){s=5;}
                else{
                    if ((x<(3*w)/4) & (y<h/4)){s=1;}
                    if ((x>=(3*w)/4) & (y<h/4)){s=2;}
                    if ((x>=(3*w)/4) & (y>=h/4)){s=3;}
                    if ((x<(3*w)/4) & (y>=h/4)){s=4;}
                }
                break;
            case 3:
                if (((2*w)/3 < x)& (x <(5*w)/6) & ((2*h)/3<y) & (y<(5*h)/6)){s=5;}
                else{
                    if ((x<(3*w)/4) & (y<(3*h)/4)){s=1;}
                    if ((x>=(3*w)/4) & (y<(3*h)/4)){s=2;}
                    if ((x>=(3*w)/4) & (y>=(3*h)/4)){s=3;}
                    if ((x<(3*w)/4) & (y>=(3*h)/4)){s=4;}
                }
                break;
            case 4:
                if ((w/6 < x)& (x <w/3) & ((2*h)/3<y) & (y<(5*h)/6)){s=5;}
                else{
                    if ((x<w/4) & (y<(3*h)/4)){s=1;}
                    if ((x>=w/4) & (y<(3*h)/4)){s=2;}
                    if ((x>=w/4) & (y>=(3*h)/4)){s=3;}
                    if ((x<w/4) & (y>=(3*h)/4)){s=4;}
                }
                break;
        }
        return s;
    }


    ///////// end of M variables/functions  /////////////////////

    public static int first_phase;

    public static int num_mines;

    public static int selfx;
    public static int selfy;
    public static MapLocation selfloc;

    public static int mapWidth;
    public static int mapHeight;


    public static int midX1;
    public static int midX2;
    public static int midY1;
    public static int midY2;

    public static int myId;
    public static Team myTeam;
    public static Team opponentTeam;

    static FastSet friendlyTowerLocations;
    static FastSet enemyTowerLocations;
    static FastSet noRefillTowerLocations;
    static FastSet flickerTowerLocations;
    static FastSet ruinLocations; // mine locations
    static MapLocation oppositeLocation;


    final static int enemyLocMinDist = 16;
    static MapLocation[] latestEnemyLocations = new MapLocation[25];
    static int latestEnemyLocationIndex = 0;

    static MapLocation[] exploreLocations = new MapLocation[9];
    static MapLocation[] lateExploreLocations;
    static boolean[] exploreLocationsVisited = new boolean[9];
    static boolean wandering = false;
    static MapLocation wanderLocation;
    static int minWanderDistance = 100;
    static int minDistToTarget = 999999;
    static int wanderCount = 0;
    static int maxWanderingCounter = 5;
    static int noActionCounter = 0;
    static int noActionThreshold = 10;
    static int noFlickerCounter = 0;
    static MapLocation flipLocation = null;

    static FastSet taintedRuins;
    static MapLocation spawnLocation;
    // diag, vert, horz
    static MapLocation[] symmetryLocations = new MapLocation[3];
    static boolean[] symmetryLocationsVisited = new boolean[3];
    static int symmetry = -1;
    static boolean[] symmetryBroken = new boolean[3];
    static FastSet vertCheckedLocations;
    static FastSet horzCheckedLocations;
    static FastSet diagCheckedLocations;



    static boolean[][] nearbyAllies = new boolean[3][3];

    public static Direction[] adjacentDirections = {
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST,
            Direction.NORTHEAST,
            Direction.SOUTHEAST,
            Direction.SOUTHWEST,
            Direction.NORTHWEST
    };
    public static Direction [] cardinalDirections = {
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST
    };

    static MapLocation[] perpendicularEdgePoints(
            MapLocation me,
            MapLocation center,
            int xmin, int xmax,
            int ymin, int ymax) {

        int dx = center.x - me.x;
        int dy = center.y - me.y;

        // perpendicular direction
        int px = -dy;
        int py = dx;

        MapLocation[] result = new MapLocation[2];
        int count = 0;

        // vertical edges
        if (px != 0) {
            double t = (xmin - center.x) / (double) px;
            double y = center.y + t * py;
            if (y >= ymin && y <= ymax) {
                result[count++] = new MapLocation(xmin, (int)Math.round(y));
            }

            t = (xmax - center.x) / (double) px;
            y = center.y + t * py;
            if (y >= ymin && y <= ymax && count < 2) {
                result[count++] = new MapLocation(xmax, (int)Math.round(y));
            }
        }

        // horizontal edges
        if (py != 0 && count < 2) {
            double t = (ymin - center.y) / (double) py;
            double x = center.x + t * px;
            if (x >= xmin && x <= xmax) {
                result[count++] = new MapLocation((int)Math.round(x), ymin);
            }

            t = (ymax - center.y) / (double) py;
            x = center.x + t * px;
            if (x >= xmin && x <= xmax && count < 2) {
                result[count++] = new MapLocation((int)Math.round(x), ymax);
            }
        }

        return result;
    }

    public static void write_loc(int index, MapLocation loc) throws GameActionException {
        rc.writeSharedArray(index, loc.x);
        rc.writeSharedArray(index+1, loc.y);
    }

    public static MapLocation read_loc(int index) throws GameActionException {
        return new MapLocation(rc.readSharedArray(index), rc.readSharedArray(index+1));
    }

    public static int translate_coords_to_int(MapLocation mine) {
        int x=mine.x/2;
        int y=mine.y/2;
        return (int)(x * 32) + y;
    }

    public static MapLocation translate_int_to_coords(int i) {
        i=i%1024;
        return new MapLocation((int)(2*(i-i%32)/32), 2*(i%32));
    }

    public static int translate_coords_to_int_precise(MapLocation mine) {
        int x=mine.x;
        int y=mine.y;
        return (int)(x * 64) + y;
    }

    public static MapLocation translate_int_to_coords_precise(int i) {
        i=i%1024;
        return new MapLocation((int)((i-i%64)/64), (i%64));
    }





    public static void init(RobotController robotController) throws GameActionException {
        rc = robotController;

       mapWidth = rc.getMapWidth();
        mapHeight = rc.getMapHeight();
        if (mapWidth % 2 == 1) {
            midX1 = mapWidth / 2;
            midX2 = mapWidth / 2;
        } else {
            midX1 = mapWidth / 2 - 1;
            midX2 = mapWidth / 2;
        }
        if (mapHeight % 2 == 1) {
            midY1 = mapHeight / 2;
            midY2 = mapHeight / 2;
        } else {
            midY1 = mapHeight / 2 - 1;
            midY2 = mapHeight / 2;
        }

        selfx=rc.getLocation().x;
        selfy=rc.getLocation().y;
        selfloc=rc.getLocation();

        /*
        MapLocation[] edgepoints = perpendicularEdgePoints(rc.getLocation(),new MapLocation(midX1,midY1)
        ,0,mapWidth,0,mapHeight);



        // have 5 spread targets for the rats to explore towards at the start
        targets=new MapLocation[100];
        targets[0]=edgepoints[0];
        targets[1]=new MapLocation((midX1+edgepoints[0].x)/2,(midY1+edgepoints[0].y)/2);
        targets[2]=new MapLocation(midX1,midY1);
        targets[3]=new MapLocation((midX1+edgepoints[1].x)/2,(midY1+edgepoints[1].y)/2);
        targets[4]=edgepoints[1];
        */


        try {

            for (int i = 63; i > 9; i--) {
                if (rc.readSharedArray(i) != 0) {
                    num_mines+=1;
                }
            }
        } catch (Exception ex) {}




        /// /////////////////// ///////////////////////

        int minX = 2;
        int midX = mapWidth / 2;
        int maxX = mapWidth - 3;
        int minY = 2;
        int midY = mapHeight / 2;
        int maxY = mapHeight - 3;
        exploreLocations[0] = new MapLocation(minX, minY);
        exploreLocations[1] = new MapLocation(midX, minY);
        exploreLocations[2] = new MapLocation(maxX, minY);
        exploreLocations[3] = new MapLocation(minX, midY);
        exploreLocations[4] = new MapLocation(midX, midY);
        exploreLocations[5] = new MapLocation(maxX, midY);
        exploreLocations[6] = new MapLocation(minX, maxY);
        exploreLocations[7] = new MapLocation(midX, maxY);
        exploreLocations[8] = new MapLocation(maxX, maxY);

        lateExploreLocations=getSixFurthestExploreLocations(rc.getLocation());
        boolean centerIsIn=false;
        for (MapLocation loc : lateExploreLocations) {
            if (loc.equals(new MapLocation(midX, midY))) {centerIsIn=true;}
        }
        if (!centerIsIn) {
            lateExploreLocations = new MapLocation[]{
                    lateExploreLocations[0],
                    lateExploreLocations[1],
                    lateExploreLocations[2],
                    lateExploreLocations[3],
                    lateExploreLocations[4],
                    lateExploreLocations[5],
                    new MapLocation(midX, midY)
            };
        }

        friendlyTowerLocations = new FastSet();
        enemyTowerLocations = new FastSet();
        noRefillTowerLocations = new FastSet();
        flickerTowerLocations = new FastSet();
        ruinLocations = new FastSet();
        taintedRuins = new FastSet();
        vertCheckedLocations = new FastSet();
        horzCheckedLocations = new FastSet();
        diagCheckedLocations = new FastSet();
        first_phase=(mapHeight+mapWidth)/2;

        oppositeLocation=new MapLocation(mapWidth-rc.getLocation().x, mapHeight-rc.getLocation().y);
    }



    static MapLocation[] getSixFurthestExploreLocations(MapLocation from) {
        MapLocation[] result = new MapLocation[6];
        int[] bestDist = new int[6];

        // initialize distances to -1
        for (int i = 0; i < 6; i++) bestDist[i] = -1;

        for (int i = 0; i < exploreLocations.length; i++) {
            MapLocation loc = exploreLocations[i];
            if (loc == null) continue;

            int d = from.distanceSquaredTo(loc);

            // insert into top-6 if far enough
            for (int j = 0; j < 6; j++) {
                if (d > bestDist[j]) {
                    // shift down
                    for (int k = 5; k > j; k--) {
                        bestDist[k] = bestDist[k - 1];
                        result[k] = result[k - 1];
                    }
                    bestDist[j] = d;
                    result[j] = loc;
                    break;
                }
            }
        }
        return result;
    }

}