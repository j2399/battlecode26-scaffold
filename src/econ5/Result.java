package econ5;

import battlecode.common.Direction;
import battlecode.common.MapLocation;

public class Result {




   public static MapLocation attack_loc;
    public   static MapLocation carry_loc;
    public static int attack_order;
    public static int move_order;
    public static int carry_order;
    public static int rotate_order;
    public static int cheese_value;
    public  static Direction turn_dir;
    public  static Direction move_dir;



    Result(MapLocation attack, MapLocation carry, int attack_or,
           int move_or, int carry_or, int rotate_or, int cheese, Direction direc, Direction move) {
            attack_loc=attack;

            carry_loc=carry;
            attack_order=attack_or;
            move_order=move_or;
            carry_order=carry_or;
            carry_order=cheese_value;
            rotate_order=rotate_or;
            cheese_value=cheese;
            turn_dir=direc;
            move_dir=move;
    }

    Result() {
        attack_loc=null;
        carry_loc=null;
        attack_order=-1;
        move_order=-1;
        carry_order=-1;
        rotate_order=-1;
        cheese_value=Integer.MIN_VALUE;
        turn_dir=null;
        move_dir=null;
    }

}
