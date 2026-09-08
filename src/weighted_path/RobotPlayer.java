package weighted_path;

import battlecode.common.*;

/**
 * Entry point: initializes shared Unit state once, then dispatches each
 * turn to King or Rat depending on this robot's UnitType.
 */
public class RobotPlayer {

    public static void run(RobotController rc) throws GameActionException {
        Unit.init(rc);

        while (true) {
            try {
                Unit.update();
                switch (rc.getType()) {
                    case RAT_KING -> King.run(rc);
                    case BABY_RAT -> Rat.run(rc);
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
