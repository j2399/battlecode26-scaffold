package econ5;

import battlecode.common.*;

/**
 * Class used to compute / keep track of the symmetry. Only used for the first soldiers spawned from a paint tower.
 */

public class SymmetryManager extends Globals {

    static final int H = 1, V = 2, R = 4;
    static final int HC = (~H)&7, VC = (~V)&7, RC = (~R)&7;
/* The following code for checking symmetry was modified from Geffner's code on
 https://github.com/IvanGeffner/BC25/blob/master/basic45/SymmetryManager.java           */
    /*
    static int discardedSyms = 0;

    static MapLocation L;
    static int symX, symY;

    static long w, r;

    static long[] mapWalls = new long[60];
    static long[] mapVision = new long[60];

    static void setSim(int sym){
        switch(sym){
            case H -> discardedSyms = HC; // discard V and R
            case V -> discardedSyms = VC; // discard H and R
            case R -> discardedSyms = RC; // discard H and V
        }
    }

    static void discardH(){
        discardedSyms |= H;
    }

    static void discardV(){
        discardedSyms |= V;
    }

    static void discardR(){
        discardedSyms |= R;
    }

    static int getSym(){
        return switch(discardedSyms){
            case HC -> H;
            case VC -> V;
            case RC -> R;
            default -> 0;
        };
    }

    static void checkSym(RobotController rc){
        if (Clock.getBytecodesLeft() < 400) return;
        if (getSym() != 0) return;
        MapInfo[] infos = rc.senseNearbyMapInfos();
        for (MapInfo m : infos){
            if(m.isDirt()){
                return;
            }
            if (Clock.getBytecodesLeft() < 400) return;
            L = m.getMapLocation();

            if (L.x < 0 || L.x >= mapWidth || L.y < 0 || L.y >= mapHeight) {
                continue;
            }
            mapVision[L.x] |= (1L << L.y);
            w = m.isWall() ? 1L : 0L;
            mapWalls[L.x] |= (w << L.y);
            symX = mapWidth- L.x - 1;
            symY = mapHeight - L.y - 1;
            if (((mapVision[symX] & (1L << L.y)) != 0) && (((mapWalls[symX] >>> L.y) & 1) != w)) discardH();
            if (((mapVision[L.x] & (1L << symY)) != 0) && (((mapWalls[L.x] >>> symY) & 1) != w)) discardV();
            if (((mapVision[symX] & (1L << symY)) != 0) && (((mapWalls[symX] >>> symY) & 1) != w )) discardR();
        }
    }

    static MapLocation getSymmetric(MapLocation loc){
        return switch(getSym()){
            case H-> new MapLocation(mapWidth - loc.x - 1, loc.y);
            case V-> new MapLocation(loc.x, mapHeight - loc.y - 1);
            case R-> new MapLocation(mapWidth- loc.x - 1, mapHeight - loc.y - 1);
            default -> null;
        };
    }*/

}