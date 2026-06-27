package micro_move_imitator;

import battlecode.common.*;

public class SpawnManager extends Globals {

    public static double[] weights = new double[8];

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

    public static void init(RobotController rc) throws GameActionException {
        MapLocation my_loc=rc.getLocation();
        int x=my_loc.x;
        int y=my_loc.y;

        weights[0]= Math.pow(Math.max(0,mapHeight-y-3),2.5);
        weights[1]= Math.pow(Math.max(mapWidth-x-3,0),2.5);
        weights[2]= Math.pow(Math.max(0,y-2),2.5);
        weights[3]= Math.pow(Math.max(x-2,0),2.5);
        int slide=Math.min(mapWidth-x-3,mapHeight-y-3);
        weights[4]= Math.pow(Math.max(0,Math.sqrt(2)*slide),2.5);
        slide=Math.min(mapWidth-x-3,y-2);
        weights[5]= Math.pow(Math.max(0,Math.sqrt(2)*slide),2.5);
        slide=Math.min(x-2,y-2);
        weights[6]= Math.pow(Math.max(0,Math.sqrt(2)*slide),2.5);
        slide=Math.min(x-2,mapHeight-y-3);
        weights[7]= Math.pow(Math.max(0,Math.sqrt(2)*slide),2.5);

    }

    public static Direction get_spawn_direction() throws GameActionException {
        double best_weight=Integer.MIN_VALUE;
        int best_index=0;
        if (weights[0]>best_weight) {
            best_weight=weights[0];
            best_index=0;
        }
        if (weights[1]>best_weight) {
            best_weight=weights[1];
            best_index=1;
        }
        if (weights[2]>best_weight) {
            best_weight=weights[2];
            best_index=2;
        }
        if (weights[3]>best_weight) {
            best_weight=weights[3];
            best_index=3;
        }
        if (weights[4]>best_weight) {
            best_weight=weights[4];
            best_index=4;
        }
        if (weights[5]>best_weight) {
            best_weight=weights[5];
            best_index=5;
        }if (weights[6]>best_weight) {
            best_weight=weights[6];
            best_index=6;
        }
        if (weights[7]>best_weight) {
            best_index=7;
        }
        weights[best_index]/=2;
        return adjacentDirections[best_index];


    }

}
