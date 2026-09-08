package weightedMicro;

import battlecode.common.*;

import java.util.Random;

public class ExploreState extends Unit {
    static final Random rng = new Random(6147);

    // TODO implement explore state 

    public static MapLocation targetTemp=new MapLocation(w-locMy.x, h-locMy.y);

    public static void run() throws GameActionException {
        Direction dir=locMy.directionTo(targetTemp);
        System.out.println("in explore");
        if (rc.canMove(dir)){
            if (rc.canTurn(dir)){
                rc.turn(dir);
            }
            rc.move(dir);
            update();
        }
        else { 
            dir=dir.rotateLeft();
            if (rc.canMove(dir)){
            if (rc.canTurn(dir)){
                rc.turn(dir);
            }
            rc.move(dir);
            update();
            }
            else { 
                dir=dir.rotateRight().rotateRight();
                if (rc.canMove(dir)){
                if (rc.canTurn(dir)){
                    rc.turn(dir);
                }
                rc.move(dir);
                update();
            }
        }
        }

    }
}
