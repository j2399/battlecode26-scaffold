package micro_move_imitator;

import battlecode.common.*;
import micro_move_imitator.Globals;
import micro_move_imitator.WbugNav;
import java.util.ArrayList;

import java.util.ArrayList;
// import java.util.ArrayList;

public class NavigationRevised extends Globals {
    // variables
    private static MapLocation node;
    private static int[][] nodes;
    private static Direction[] directions;

// returns 1 if tile is passable or has dirt
    public static int coord6(int x, int y) throws GameActionException {
        MapLocation loc=new MapLocation(x,y);
        if(rc.canSenseLocation(loc)) {
            MapInfo tile = rc.senseMapInfo(loc);
        if ( tile.isWall() ) { return 0;}
        else {return 1;}
        }
        else {return 0;}
}

// returns 1 if tile has dirt
    public static int coord7(int x, int y) throws GameActionException {
        MapLocation loc=new MapLocation(x,y);
        if(rc.canSenseLocation(loc)) {
            MapInfo tile = rc.senseMapInfo(loc);
            if (tile.isDirt()) {
                return 1;
            } else {
                return 0;
            }
        }
        return 0;
    }


// nodes: make a list of visible nodes
// !!!!!!! only apply when at least 5 steps away from edge
    public static void get_nodes(int x, int y, Direction d) throws GameActionException {
//0:layer (4 parameter and 1 initial fringe), 1:start, 2:fringe, 3:explored,  4:x, 5:y,  6:is_wall or is_dirt, 7:path=is_dirt,
// 1,1,0    // ,c6(),c7()    //  0,0,0,
        switch (d) {
            case WEST:
                nodes = new int[][]{{1, 1, 1, 0, x - 2, y - 1,0,0}, {1, 2, 1, 1, 0, x - 2, y,0,0}, {1, 3, 1, 1, 0, x - 2, y + 1,0,0},
                        {4, 0, 0, 0, x - 4, y - 2,0,0}, {4, 0, 0, 0, x - 4, y - 1,0,0}, {4, 0, 0, 0, x - 4, y,0,0}, {4, 0, 0, 0, x - 4, y + 1,0,0}, {4, 0, 0, 0, x - 4, y + 2,0,0},
                        {3, 0, 0, 0, x - 3, y - 2,0,0}, {3, 0, 0, 0, x - 3, y - 1,0,0}, {3, 0, 0, 0, x - 3, y,0,0}, {3, 0, 0, 0, x - 3, y + 1,0,0}, {3, 0, 0, 0, x - 3, y + 2,0,0},
                        {2, 0, 0, 0, x - 2, y - 2,0,0}, {2, 0, 0, 0, x - 2, y - 1,0,0}, {2, 0, 0, 0, x - 2, y,0,0}, {2, 0, 0, 0, x - 2, y + 1,0,0}, {2, 0, 0, 0, x - 2, y + 2,0,0}
                };
                break;
            case NORTH:
                nodes = new int[][]{{1, 1, 1,  0, x - 1, y + 1,0,0}, {1, 2, 1,  0, x, y + 1,0,0}, {1, 3, 1,  0, x + 1, y + 1,0,0},
                        {4, 0, 0, 0, x - 2, y + 4,0,0}, {4, 0, 0, 0, x - 1, y + 4,0,0}, {4, 0, 0, 0, x, y + 4,0,0}, {4, 0, 0, 0, x + 1, y + 4,0,0}, {4, 0, 0, 0, x + 2, y + 4,0,0},
                        {3, 0, 0, 0, x - 2, y + 3,0,0}, {3, 0, 0, 0, x - 1, y + 3,0,0}, {3, 0, 0, 0, x, y + 3,0,0}, {3, 0, 0, 0, x + 1, y + 3,0,0}, {3, 0, 0, 0, x + 2, y + 3,0,0},
                        {2, 0, 0, 0, x - 2, y + 2,0,0}, {2, 0, 0, 0, x - 1, y + 2,0,0}, {2, 0, 0, 0, x, y + 2,0,0}, {2, 0, 0, 0, x + 1, y + 2,0,0}, {2, 0, 0, 0, x + 2, y + 2,0,0}
                };
                break;
            case EAST:
                nodes = new int[][]{{1, 1, 1,  0, x + 1, y - 1,0,0}, {1, 2, 1,  0, x + 1, y,0,0}, {1, 3, 1,  0, x + 1, y + 1,0,0},
                        {4, 0, 0, 0, x + 4, y - 2,0,0}, {4, 0, 0, 0, x + 4, y - 1,0,0}, {4, 0, 0, 0, x + 4, y,0,0}, {4, 0, 0, 0, x + 4, y + 1,0,0}, {4, 0, 0, 0, x + 4, y + 2,0,0},
                        {3, 0, 0, 0, x + 3, y - 2,0,0}, {3, 0, 0, 0, x + 3, y - 1,0,0}, {3, 0, 0, 0, x + 3, y,0,0}, {3, 0, 0, 0, x + 3, y + 1,0,0}, {3, 0, 0, 0, x + 3, y + 2,0,0},
                        {2, 0, 0, 0, x + 2, y - 2,0,0}, {2, 0, 0, 0, x + 2, y - 1,0,0}, {2, 0, 0, 0, x + 2, y,0,0}, {2, 0, 0, 0, x + 2, y + 1,0,0}, {2, 0, 0, 0, x + 2, y + 2,0,0}
                };
                break;
            case SOUTH:
                nodes = new int[][]{{1, 1, 1,  0, x - 1, y - 1,0,0}, {1, 2, 1,  0, x, y - 1,0,0}, {1, 3, 1,  0, x + 1, y - 1,0,0},
                        {4, 0, 0, 0, x - 2, y - 4,0,0}, {4, 0, 0, 0, x - 1, y - 4,0,0}, {4, 0, 0, 0, x, y - 4,0,0}, {4, 0, 0, 0, x + 1, y - 4,0,0}, {4, 0, 0, 0, x + 2, y - 4,0,0},
                        {3, 0, 0, 0, x - 2, y - 3,0,0}, {3, 0, 0, 0, x - 1, y - 3,0,0}, {3, 0, 0, 0, x, y - 3,0,0}, {3, 0, 0, 0, x + 1, y - 3,0,0}, {3, 0, 0, 0, x + 2, y - 3,0,0},
                        {2, 0, 0, 0, x - 2, y - 2,0,0}, {2, 0, 0, 0, x - 1, y - 2,0,0}, {2, 0, 0, 0, x, y - 2,0,0}, {2, 0, 0, 0, x + 1, y - 2,0,0}, {2, 0, 0, 0, x + 2, y - 2,0,0}
                };
                break;

            case NORTHEAST:
                nodes = new int[][]{{1, 1, 1,  0, x, y + 1,0,0}, {1, 2, 1,  0, x + 1, y + 1,0,0}, {1, 3, 1,  0, x + 1, y,0,0},
                        {4, 0, 0, 0, x + 1, y + 4,0,0}, {4, 0, 0, 0, x + 2, y + 4,0,0}, {4, 0, 0, 0, x + 3, y + 3,0,0}, {4, 0, 0, 0, x + 4, y + 2,0,0}, {4, 0, 0, 0, x + 4, y + 1,0,0},
                        {3, 0, 0, 0, x, y + 3,0,0}, {3, 0, 0, 0, x + 1, y + 3,0,0}, {3, 0, 0, 0, x + 2, y + 3,0,0}, {3, 0, 0, 0, x + 3, y + 2,0,0}, {3, 0, 0, 0, x + 3, y + 1,0,0},
                        {2, 0, 0, 0, x, y + 2,0,0}, {2, 0, 0, 0, x + 1, y + 2,0,0}, {2, 0, 0, 0, x + 2, y + 2,0,0}, {2, 0, 0, 0, x + 2, y + 1,0,0}, {2, 0, 0, 0, x + 2, y,0,0}
                };
                break;
            case SOUTHEAST:
                nodes = new int[][]{{1, 1, 1,  0, x + 1, y,0,0}, {1, 2, 1,  0, x + 1, y - 1,0,0}, {1, 3, 1,  0, x, y - 1,0,0},
                        {4, 0, 0, 0, x + 4, y - 1,0,0}, {4, 0, 0, 0, x + 4, y - 2,0,0}, {4, 0, 0, 0, x + 3, y - 3,0,0}, {4, 0, 0, 0, x + 2, y - 4,0,0}, {4, 0, 0, 0, x + 1, y - 4,0,0},
                        {3, 0, 0, 0, x + 3, y,0,0}, {3, 0, 0, 0, x + 3, y - 1,0,0}, {3, 0, 0, 0, x + 3, y - 2,0,0}, {3, 0, 0, 0, x + 2, y - 3,0,0}, {3, 0, 0, 0, x + 1, y - 3,0,0},
                        {2, 0, 0, 0, x + 2, y,0,0}, {2, 0, 0, 0, x + 2, y - 1,0,0}, {2, 0, 0, 0, x + 2, y - 2,0,0}, {2, 0, 0, 0, x + 1, y - 2,0,0}, {2, 0, 0, 0, x, y - 2,0,0}
                };

                break;
            case SOUTHWEST:
                nodes = new int[][]{{1, 1, 1,  0, x, y - 1,0,0}, {1, 2, 1,  0, x - 1, y - 1,0,0}, {1, 3, 1,  0, x - 1, y,0,0},
                        {4, 0, 0, 0, x + 4, y - 1,0,0}, {4, 0, 0, 0, x + 4, y - 2,0,0}, {4, 0, 0, 0, x - 3, y - 3,0,0}, {4, 0, 0, 0, x - 2, y - 4,0,0}, {4, 0, 0, 0, x - 1, y - 4,0,0},
                        {3, 0, 0, 0, x, y - 3,0,0}, {3, 0, 0, 0, x - 1, y - 3,0,0}, {3, 0, 0, 0, x - 2, y - 3,0,0}, {3, 0, 0, 0, x - 3, y - 2,0,0}, {3, 0, 0, 0, x - 3, y - 1,0,0},
                        {2, 0, 0, 0, x, y - 2,0,0}, {2, 0, 0, 0, x - 1, y - 2,0,0}, {2, 0, 0, 0, x - 2, y - 2,0,0}, {2, 0, 0, 0, x - 2, y - 1,0,0}, {2, 0, 0, 0, x - 2, y,0,0}
                };
                break;
            case NORTHWEST:
                nodes = new int[][]{{1, 1, 1,  0, x - 1, y,0,0}, {1, 2, 1,  0, x - 1, y + 1,0,0}, {1, 3, 1,  0, x, y + 1,0,0},
                        {4, 0, 0, 0, x - 4, y - 1,0,0}, {4, 0, 0, 0, x - 4, y - 2,0,0}, {4, 0, 0, 0, x - 3, y + 3,0,0}, {4, 0, 0, 0, x - 2, y + 4,0,0}, {4, 0, 0, 0, x - 1, y + 4,0,0},
                        {3, 0, 0, 0, x - 3, y,0,0}, {3, 0, 0, 0, x - 3, y + 1,0,0}, {3, 0, 0, 0, x - 3, y + 2,0,0}, {3, 0, 0, 0, x - 2, y + 3,0,0}, {3, 0, 0, 0, x - 1, y + 3,0,0},
                        {2, 0, 0, 0, x - 2, y,0,0}, {2, 0, 0, 0, x - 2, y + 1,0,0}, {2, 0, 0, 0, x - 2, y + 2,0,0}, {2, 0, 0, 0, x - 1, y + 2,0,0}, {2, 0, 0, 0, x, y + 2,0,0}
                };
                break;
        }
        //return prenodes;
    }



    public static void update_nodes() throws GameActionException {
        for (int i = 0; i <= 17; i++) {
            nodes[i][6] = coord6(nodes[i][4], nodes[i][5]); // 0 if wall
            nodes[i][7] = 2*coord7(nodes[i][4], nodes[i][5]); // 2 if dirt
        }
        // immediate neighbors of base get path=1or2
        nodes[0][7] =coord7(nodes[0][4], nodes[0][5])+coord6(nodes[0][4], nodes[0][5]);
        nodes[1][7] = coord7(nodes[1][4], nodes[1][5])+coord6(nodes[1][4], nodes[1][5]);
        nodes[2][7] =  coord7(nodes[2][4], nodes[2][5])+coord6(nodes[2][4], nodes[2][5]);
        //return nodes;
    }

// from MapLocation make a list
    public static int[] unpack(MapLocation loc) throws GameActionException {
        int[] result={loc.x, loc.y};
        return result;
    }

// find index of min element in fringe
    public static int min_fringe(int[][] n) throws GameActionException {
        int min_index=100;
        int min_value=100;
        for (int i = 0; i <=17; i++) {
            if ( (  nodes[i][2]==1) & (nodes[i][7] < min_value)) {
                min_index=i;
                min_value=nodes[i][7];
            }
        }
        return min_index;
    }

// find fringe size
    public static int size_fringe(int[][] n) throws GameActionException {
        int size=0;
        for (int i = 0; i <=17; i++) {
            if  (  n[i][2]==1) { size=size+1;}
        }
        return size;
    }

    // find index of min element in parameter
    public static int min_parameter(int[][] n) throws GameActionException {
        int min_index=100;
        int min_value=100;
        for (int i = 3; i <=7; i++) {
            if ( (nodes[i][7]!=0) & (nodes[i][7] < min_value)) {
                min_index=i;
                min_value=nodes[i][7];
            }
        }
        return min_index;
    }

// are (x,y) and (a,b)  neighbors
    public static boolean are_neighbors(int x, int y, int a, int b) throws GameActionException {
        if ( ((x-a)*(x-a)+(y-b)*(y-b))<2) {return true;}
        else {return false;}
    }

// check for unexplored passable neighbor of (x,y)
    public static boolean is_neighbor(int x, int y, int[] element) throws GameActionException {
         if ( (element[3]==0) & (element[6]==1) & ( ( (element[4]-x)*(element[4]-x)+(element[5]-y)*(element[5]-y) )<=2) ){return true;}
         else {return false;}
    }

    public static MapLocation best_loc;

// bugnav move
public static void apply_bugnav(MapLocation target) throws GameActionException {
    Direction dir = WbugNav.moveTo(target);
    if (rc.canTurn()) {rc.turn(dir);}
    if (rc.canMove(dir)) {rc.move(dir);}
}

// move diagonally from current dir; if not possible, move perpendicularly
    public static void move_around(Direction d, MapLocation target) throws GameActionException {
        switch (d) {
            case NORTH:
                directions=new Direction[] {Direction.NORTHEAST, Direction.NORTHWEST, Direction.EAST, Direction.WEST,Direction.NORTH};
                break;
            case NORTHEAST:
                directions=new Direction[] {Direction.EAST, Direction.NORTH,Direction.NORTHWEST, Direction.SOUTHEAST,Direction.NORTHEAST };
                break;
            case EAST:
                directions=new Direction[] {Direction.NORTHEAST,Direction.SOUTHEAST,Direction.NORTH,Direction.SOUTH,Direction.EAST};
                break;
            case SOUTHEAST:
                directions=new Direction[]{Direction.EAST,Direction.SOUTH,Direction.NORTHEAST,Direction.SOUTHWEST,Direction.SOUTHEAST};
                break;
            case SOUTH:
                directions=new Direction[] {Direction.SOUTHWEST,Direction.SOUTHEAST,Direction.EAST,Direction.WEST,Direction.SOUTH};
                break;
            case SOUTHWEST:
                directions=new Direction[] {Direction.SOUTH,Direction.WEST,Direction.SOUTHEAST,Direction.NORTHWEST,Direction.SOUTHWEST};
                break;
            case WEST:
                directions=new Direction[] {Direction.NORTHWEST,Direction.SOUTHWEST,Direction.NORTH,Direction.SOUTH,Direction.WEST};
                break;
            case NORTHWEST:
                directions=new Direction[] {Direction.NORTH,Direction.WEST,Direction.NORTHEAST,Direction.SOUTHWEST,Direction.NORTHWEST};
                break;
            default:
                directions=new Direction[] {Direction.NORTH,Direction.WEST,Direction.EAST,Direction.SOUTH,Direction.NORTHWEST};
                break;
        }
        int moved=0;
        for (Direction dir:directions) {
            if (rc.canMove(dir)) {
                rc.turn(dir);
                rc.move(dir);
                moved = 1;
                break;
            }
            if (moved == 0) {
                apply_bugnav(target);
            }
        }
    }




// BFS; move to best tile going to target
// !!!!!!! only apply when at least 5 steps away from edge
    public static void bfs(MapLocation target, RobotController rc) throws GameActionException {
        // starting point
        Direction dn = rc.getLocation().directionTo(target);
        int xn = rc.getLocation().x;
        int yn = rc.getLocation().y;

        int byte1=Clock.getBytecodeNum();

        // construct nodes
        get_nodes(xn, yn, dn);
        //int byte2=Clock.getBytecodeNum();
        //System.out.println("byes used get param: " +(byte2-byte1));
        //int byte3=Clock.getBytecodeNum();
        //System.out.println("byes used get nodes: " +(byte3-byte2));

        // if walls next ahead, then bugnav
        if ( (nodes[0][6] == 0) & (nodes[1][6] == 0) & (nodes[2][6] == 0)) {
            apply_bugnav(target);
            return;
        }

        // if long walls ahead, then turn and move
        if (nodes[3][6]+nodes[4][6]+nodes[5][6]+nodes[6][6]+nodes[7][6]==0){ move_around(dn,target); return;}
        if (nodes[8][6]+nodes[9][6]+nodes[10][6]+nodes[11][6]+nodes[12][6]==0){ move_around(dn,target); return;}
        if (nodes[13][6]+nodes[14][6]+nodes[15][6]+nodes[16][6]+nodes[17][6]==0){ move_around(dn,target); return;}

        // if a single possible move next ahead, go there
        MapLocation here=rc.getLocation();
        if ( (nodes[0][6] == 0) & (nodes[1][6] == 0)) {
            MapLocation loc=new MapLocation(nodes[2][4], nodes[2][5 ]);
            Direction dir = here.directionTo(loc);
            if (rc.canMove(dir)) {rc.move(dir);return;}
        }
        if ( (nodes[1][6] == 0) & (nodes[2][6] == 0)) {
            MapLocation loc=new MapLocation(nodes[0][4], nodes[0][5]);
            Direction dir = here.directionTo(loc);
            if (rc.canMove(dir)) {rc.move(dir);return;}
       }
       if ( (nodes[0][6] == 0) & (nodes[2][6] == 0)) {
             MapLocation loc=new MapLocation(nodes[1][4], nodes[1][5]);
             Direction dir = here.directionTo(loc);
             if (rc.canMove(dir)) {rc.move(dir);return;}
      }

      // if can move forward, then bfs computes paths
      while (size_fringe(nodes) > 0) {
             int index = min_fringe(nodes); // pick element in fringe
             nodes[index][2] = 0; // remove from fringe
             nodes[index][3] = 1; // add to explored

      // add to fringe the unexplored neighbors, change their  l to l+1, change their starting point
             for (int i = 0; i <= 17; i++) {
                   if (is_neighbor(nodes[index][4], nodes[index][5], nodes[i])) {
                       nodes[i][2] = 1; // add to fringe
                       nodes[i][7] = nodes[i][7] + 1; // increase path
                       nodes[i][1] = nodes[index][1]; // set starting point
                   }
             }
      }

            //0  level; parameter is 4
            //1  1,2,3 starting point
            //2  1 in fringe
            //3  1 in explored
            //4  x
            //5  y
            //6  1 if passable
            //7  1 if dirt, path

                //int old_bytes=Clock.getBytecodeNum();
                //int new_bytes=Clock.getBytecodeNum();
                //byte1=Clock.getBytecodeNum();
                //byte2=Clock.getBytecodeNum();
                //System.out.println("byes used get neigh: " +(byte2-byte1));
                //System.out.println("neigh size: "+neighbors.size());

                // compare paths
                int best_index = min_parameter(nodes);
                if (best_index != 100) { // if best tile exists
                    int best_start = nodes[best_index][1];
                    best_loc = new MapLocation(nodes[best_index - 1][4], nodes[best_index - 1][5]);
                    // move to best location
                    Direction newdir = rc.getLocation().directionTo(best_loc);
                    if (rc.canMove(newdir) & rc.canTurn()) {
                        rc.turn(newdir);
                        rc.move(newdir);
                        return;
                    }
                }
               move_around(dn, target);

                    /*
                    // random turn
                    switch (dn) {
                        case NORTH: rc.turn(Direction.NORTHEAST);
                            break;
                        case NORTHEAST: rc.turn(Direction.EAST);
                            break;
                        case EAST: rc.turn(Direction.SOUTHEAST);
                            break;
                        case SOUTHEAST: rc.turn(Direction.SOUTH);
                            break;
                        case SOUTH: rc.turn(Direction.SOUTHWEST);
                            break;
                        case SOUTHWEST: rc.turn(Direction.WEST);
                            break;
                        case WEST: rc.turn(Direction.NORTHWEST);
                            break;
                        case NORTHWEST: rc.turn(Direction.NORTH);
                            break;
                        default:  if (rc.canTurn()) {rc.turn(Direction.NORTH);}
                            break;

                     */





        System.out.println("bytes ysed by bfs is " + (Clock.getBytecodeNum()-byte1));
    }
}



 /*
// neighbors: make a list of all neighbors of (x,y)
    public static void all_neighb(MapLocation here) throws GameActionException {
        int[][] neighbors = {
                unpack(here.add(Direction.NORTH)),
                unpack(here.add(Direction.NORTHEAST)),
                unpack(here.add(Direction.EAST)),
                unpack(here.add(Direction.SOUTHEAST)),
                unpack(here.add(Direction.SOUTH)),
                unpack(here.add(Direction.SOUTHWEST)),
                unpack(here.add(Direction.WEST)),
                unpack(here.add(Direction.NORTHWEST))
        };
    }

  */






