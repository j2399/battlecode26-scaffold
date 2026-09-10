package weighted_micro_v14;


import battlecode.common.*;

/**
 * Entry point: initializes shared Unit state once, then dispatches each
 * turn to King or Rat depending on this robot's UnitType.
 */
public class RobotPlayer {

    public static void run(RobotController rc) throws GameActionException {
        try {
            Unit.init(rc);
        }
        catch (Exception e) {
            System.out.println("Exception");
            e.printStackTrace();
        }

        while (true) {
            try {
                Unit.update();
                switch (rc.getType()) {
                    case RAT_KING -> King.run();
                    case BABY_RAT -> Rat.run();
                    default -> {}
                }
            } catch (GameActionException e) {
                System.out.println("GameActionException");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("Exception");
                e.printStackTrace();
            } finally {
                Clock.yield();
            }
        }
    }
}
