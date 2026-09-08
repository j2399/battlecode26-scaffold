import battlecode.server.Config;
import battlecode.server.GameInfo;
import battlecode.server.Main;
import battlecode.server.Server;
import battlecode.crossplay.CrossPlayLanguage;

import java.io.File;

/**
 * Runs many matches inside a single JVM process, instead of the normal
 * one-process-per-match approach (battlecode.server.Main + System.exit).
 *
 * The engine's own Server class already supports this: it holds a
 * BlockingQueue<GameInfo> and run() loops taking games off it until a
 * poison pill, then returns. Main.main() only ever queues one game before
 * calling run() and immediately exiting the process -- but nothing stops
 * queuing many games first. Doing that means JVM startup/bootstrap and JIT
 * warmup is paid once per process instead of once per match, and later
 * matches in the batch benefit from JIT work already done on earlier ones.
 *
 * Usage: java BatchServer <teamA> <teamB> <map> <numGames> <startIndex> <saveDir> <classURL>
 * All other settings (bc.server.*, bc.engine.*) are picked up from -D
 * system properties exactly as with battlecode.server.Main.
 */
public class BatchServer {
    public static void main(String[] args) throws Exception {
        String teamA = args[0];
        String teamB = args[1];
        String map = args[2];
        int numGames = Integer.parseInt(args[3]);
        int startIndex = Integer.parseInt(args[4]);
        String saveDir = args[5];
        String classURL = args[6];

        Config options = Main.setupConfig(new String[0]);
        Server server = new Server(options, false);

        // One run() call per match (instead of queuing all numGames up front
        // and calling run() once) so we can print real per-match timing --
        // Server.run() loops until it sees the poison pill added by
        // terminateNotification(), then returns cleanly, and nothing in its
        // state (gameQueue, etc.) is one-shot, so calling it again on the
        // same Server/process is safe and still pays JVM startup only once.
        for (int i = 0; i < numGames; i++) {
            int runIndex = startIndex + i;
            String label = teamA + "-vs-" + teamB + "-on-" + map + "-run" + runIndex;
            File saveFile = new File(saveDir, label + ".bc26");

            long t0 = System.nanoTime();
            server.addGameNotification(new GameInfo(
                    teamA, CrossPlayLanguage.JAVA, teamA, classURL,
                    teamB, CrossPlayLanguage.JAVA, teamB, classURL,
                    new String[]{map},
                    saveFile,
                    false
            ));
            server.terminateNotification();
            server.run();
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("[timing] run=" + runIndex + " ms=" + elapsedMs);
        }
    }
}
