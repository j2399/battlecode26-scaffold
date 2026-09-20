#!/usr/bin/env python3
import subprocess, re, os, sys, random, time
import torch
import torch.nn as nn
import torch.optim as optim

BOTS_DIR = "/Users/csproj/Desktop/rbattlecode"
MAPS = [
    "DefaultSmall", "DefaultMedium", "DefaultLarge",
    "micromap", "tiny",
    "5t4rv4t10n_1337", "arrows", "averyfineline", "averystrangespace",
    "canyoudig", "cheesebottles", "cheesefarm", "cheeseguardians",
    "closeup", "corridorofdoomanddespair", "dirtfulcat", "dirtpassageway",
    "EscapeTheNight", "evileye", "hatefullattice", "jail", "keepout",
    "knifefight", "mercifullattice", "Meow", "minimaze", "Nofreecheese",
    "peaceinourtime", "pipes", "popthecork", "rift", "safelycontained",
    "sittingducks", "starvation", "streetsofnewyork", "TheHeist",
    "thunderdome", "toomuchcheese", "trapped", "uneruesansfin",
    "wallsofparadis", "whatsthecatdoin", "whereisthecheese", "ZeroDay",
]
MAX_ROUNDS = 100
STATE_DIM = 64
HIDDEN_DIM = 32  # must match rl_train.py's HIDDEN_DIM
NUM_TILES = 9
TEACHER_EPOCHS = 1500
STUDENT_EPOCHS = 2250
BATCH_SIZE = 2048
LR = 1e-3
MODEL_PATH = os.path.join(BOTS_DIR, "qnet.pth")
TEACHER_MODEL_PATH = os.path.join(BOTS_DIR, "teacher.pth")  # IL-trained TeacherTileNet weights -- rl_train.py now warm-starts RL from THIS, not qnet.pth (see its build_warm_start_model)

def weights_java_path(package="econ5"):
    return os.path.join(BOTS_DIR, "src", package, "QNetWeights.java")

def neuralnet_java_path(package="econ5"):
    return os.path.join(BOTS_DIR, "src", package, "NeuralNet.java")


class TeacherTileNet(nn.Module):
    # Hadamard-tanh (two independent tanh branches per layer, Klein et
    # al. 2026) was tried here to fix a diagnosed E/W argmax collapse,
    # but reverted: MSE-regressing against econ5's raw tile scores
    # (magnitude in the hundreds) through tanh's bounded [-1,1] hidden
    # activations drives the final layer toward large weights to reach
    # that scale, which saturates the tanh branches -- confirmed directly
    # (88.7% of the last hidden layer's neurons both-saturated >0.99,
    # every one of the 9 outputs correlated at 1.000 with each other,
    # i.e. one shared state-dependent signal plus tiny per-action
    # offsets, not real differentiation) and Xavier/Glorot init didn't
    # prevent it either -- the pressure comes from training dynamics
    # (the large-magnitude regression target), not initialization. Back
    # to plain ReLU; the E/W collapse is instead addressed by
    # rl_train.py's ARGMAX_SHARE_COEF term, which targets the actual
    # failure mode (population-level argmax concentration) directly
    # rather than via an architecture change.
    # Also tried: scaling to 900/450/225 (~9.7x params, same 4:2:1 ratio)
    # to see if more capacity helped the E/W collapse. It trained fine
    # (best loss yet, 114185 vs 129660 at the original size), but the
    # teacher's own NeuralNet.java is what every robot calls every turn
    # DURING self-play matches (training runs under bc-overrides'
    # unlimited bytecode, so a bigger network can still execute, just
    # slower) -- confirmed directly, match-playing time alone went from
    # ~58s to ~243s per iteration (~4.2x), not just the PPO update step.
    # Reverted back to 256/128/64: the E/W collapse this was meant to
    # address is instead handled by rl_train.py's ARGMAX_SHARE_COEF term
    # (architecture-independent), so the 4x slowdown bought nothing this
    # run actually needed.
    def __init__(self, state_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.q_tilescore = nn.Linear(64, NUM_TILES)

    def forward(self, x):
        h = self.shared(x)
        return self.q_tilescore(h)


class StudentTileNet(nn.Module):
    def __init__(self, state_dim, hidden=24):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.q_tilescore = nn.Linear(hidden, NUM_TILES)

    def forward(self, x):
        h = self.shared(x)
        return self.q_tilescore(h)


def parse_obs(line):
    line = line.strip()
    if "] " in line:
        line = line.split("] ", 1)[1]
    parts = line.split("|")
    if len(parts) < 3:
        return None

    raw_str = parts[0].strip()
    capture_flag = raw_str.endswith(",C")
    if capture_flag:
        raw_str = raw_str[:-2]

    raw = [float(x) for x in raw_str.split(",") if x]
    if len(raw) < STATE_DIM + 1:
        return None
    robot_id = int(raw[0])
    raw = raw[1:]

    # e*/ally* second column is now bearing-in-degrees-from-north (/180.0,
    # matching learner_rl/RobotPlayer.java's atan2(dx,dy) convention), not a
    # raw dy offset -- the first column stays distance (/64.0, unchanged
    # scale). See rl_train.py's SYMMETRY_ANGLE_FIELDS for how this
    # transforms under the 4x symmetry augmentation used downstream.
    state = [
        raw[0] / 64.0,                # x
        raw[1] / 64.0,                # y
        raw[2] / 64.0,                # e1 distance
        raw[3] / 180.0,               # e1 bearing
        raw[4] / 100.0,               # e1 health
        raw[5] / 8.0,                 # e1 dir
        raw[6] / 64.0,                # e2 distance
        raw[7] / 180.0,               # e2 bearing
        raw[8] / 100.0,               # e2 health
        raw[9] / 8.0,                 # e2 dir
        raw[10] / 64.0,               # e3 distance
        raw[11] / 180.0,              # e3 bearing
        raw[12] / 100.0,              # e3 health
        raw[13] / 8.0,                # e3 dir
        raw[14] / 8.0,                # own dir
        min(raw[15], 20) / 20.0,      # move cd
        min(raw[16], 20) / 20.0,      # action cd
        raw[17],                       # carrying (0/1)
        # Mirrors learner_rl/RobotPlayer.java's buildState() -- see
        # econ5/RobotPlayer.java's matching state-logging block.
        raw[18] / 100.0,               # own health
        raw[19] / 64.0, raw[20] / 180.0, raw[21] / 100.0, raw[22] / 8.0,   # ally1
        raw[23] / 100.0,               # localHealthSum
        raw[24] / 64.0, raw[25] / 180.0, raw[26] / 100.0, raw[27] / 8.0,  # ally2
        raw[28] / 64.0, raw[29] / 180.0, raw[30] / 100.0, raw[31] / 8.0,  # ally3
        raw[32], raw[33], raw[34], raw[35], raw[36], raw[37], raw[38], raw[39],  # canMove x8
        # Appended enemy/ally slots 4-6 -- see collect_dataset.py's
        # identical duplicate of this function for the full rationale.
        raw[40] / 64.0, raw[41] / 180.0, raw[42] / 100.0, raw[43] / 8.0,  # enemy4
        raw[44] / 64.0, raw[45] / 180.0, raw[46] / 100.0, raw[47] / 8.0,  # enemy5
        raw[48] / 64.0, raw[49] / 180.0, raw[50] / 100.0, raw[51] / 8.0,  # enemy6
        raw[52] / 64.0, raw[53] / 180.0, raw[54] / 100.0, raw[55] / 8.0,  # ally4
        raw[56] / 64.0, raw[57] / 180.0, raw[58] / 100.0, raw[59] / 8.0,  # ally5
        raw[60] / 64.0, raw[61] / 180.0, raw[62] / 100.0, raw[63] / 8.0,  # ally6
    ]
    value = float(parts[1]) if parts[1].strip() else 0.0
    tilescores = [int(x) for x in parts[2].split(",") if x.strip()]
    while len(tilescores) < NUM_TILES:
        tilescores.append(0)
    return state, tilescores, value, robot_id, capture_flag


def run_match(map_name, team_a, team_b, max_rounds):
    props_path = os.path.join(BOTS_DIR, "gradle.properties")
    with open(props_path) as f:
        props = f.read()
    props = re.sub(r"^teamA=.*", f"teamA={team_a}", props, flags=re.MULTILINE)
    props = re.sub(r"^teamB=.*", f"teamB={team_b}", props, flags=re.MULTILINE)
    props = re.sub(r"^maps=.*", f"maps={map_name}", props, flags=re.MULTILINE)
    props = re.sub(r"^maxRounds=.*", f"maxRounds={max_rounds}", props, flags=re.MULTILINE)
    with open(props_path, "w") as f:
        f.write(props)

    result = subprocess.run(
        ["./gradlew", "runJavaLocal"],
        cwd=BOTS_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )

    samples = []
    prev_by_id = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line or line.startswith(">") or line.startswith("WAITING") or "server" in line:
            continue
        if "|" not in line:
            continue
        parsed = parse_obs(line)
        if parsed:
            state, tilescores, value, robot_id, capture_flag = parsed
            if capture_flag and robot_id in prev_by_id:
                prev_idx = prev_by_id[robot_id]
                old_state, old_ts, old_value = samples[prev_idx]
                samples[prev_idx] = (old_state, old_ts, old_value - 80)
            prev_by_id[robot_id] = len(samples)
            samples.append((state, tilescores, value))
    return samples


def train_teacher(replay_data, device, epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"Training teacher on {device}", flush=True)
    t0 = time.time()
    model = TeacherTileNet(STATE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Pre-tensorize data once
    states = torch.tensor([x[0] for x in replay_data], dtype=torch.float32).to(device)
    targets = torch.tensor([x[1] for x in replay_data], dtype=torch.float32).to(device)
    n = len(states)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            s, t = states[idx], targets[idx]

            predicted = model(s)
            loss = loss_fn(predicted, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg = total_loss / n
        print(f"  Teacher Epoch {epoch+1}/{epochs}  loss={avg:.6f}  [{time.time()-t0:.0f}s]", flush=True)

    return model, avg, time.time() - t0


def distill(teacher, replay_data, device, hidden_dim=24, epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"Distilling student (hidden={hidden_dim}) on {device}", flush=True)
    t0 = time.time()
    student = StudentTileNet(STATE_DIM, hidden_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Pre-tensorize states and pre-compute teacher targets
    states = torch.tensor([x[0] for x in replay_data], dtype=torch.float32).to(device)
    n = len(states)
    teacher.eval()
    with torch.no_grad():
        t_out = teacher(states)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            s = states[idx]

            s_out = student(s)
            loss = loss_fn(s_out, t_out[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg = total_loss / n
        print(f"  Student Epoch {epoch+1}/{epochs}  distill_loss={avg:.6f}  [{time.time()-t0:.0f}s]", flush=True)

    return student, avg, time.time() - t0


def export_weights_java(model, path, package="learner"):
    named = {n: p.data for n, p in model.named_parameters()}
    tensor_order = [
        "shared.0.weight", "shared.0.bias",
        "q_tilescore.weight",  "q_tilescore.bias",
    ]

    with open(path, "w") as f:
        f.write(f"package {package};\n\n")
        f.write("public class QNetWeights {\n")

        for key in tensor_order:
            t = named[key]
            flat = t.flatten().tolist()
            jname = key.replace(".", "_")
            f.write(f"  public static final float[] {jname} = {{\n")
            for i, v in enumerate(flat):
                f.write(f"    {v:.12e}f")
                if i < len(flat) - 1:
                    f.write(",")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n  };\n\n")

        f.write("}\n")
    print(f"Weights written to {path}", flush=True)


def export_neuralnet_java(model, path, package="learner", state_dim=18, hidden=24):
    named = {n: p.data for n, p in model.named_parameters()}
    w_sh = named["shared.0.weight"]
    b_sh = named["shared.0.bias"]
    w_out = named["q_tilescore.weight"]
    b_out = named["q_tilescore.bias"]

    fmt = lambda v: f"{v:.12e}f"

    with open(path, "w") as f:
        f.write(f"package {package};\n\n")
        f.write("public class NeuralNet {\n")
        f.write(f"    private static final int STATE_DIM = {state_dim};\n")
        f.write(f"    private static final int HIDDEN = {hidden};\n\n")
        f.write(f"    private static final float[] out0 = new float[{NUM_TILES}];\n\n")
        f.write("    public float[] forward(float[] in) {\n")

        for i in range(state_dim):
            f.write(f"        float in{i}  = in[{i}];\n")

        f.write("\n")

        # Single chained expression per neuron, not a sequence of "h{j} +=
        # ...;" statements -- each term in a compound-assignment statement
        # has to reload h{j} from its local variable slot and store the
        # new value back (fload/ldc_w/fload/fmul/fadd/fstore, 6 bytecode
        # ops), while a chained "+" expression keeps the running sum on the
        # JVM operand stack the whole time (ldc_w/fload/fmul/fadd, 4 ops) --
        # measured directly (javap -c on both forms, HIDDEN=32/STATE_DIM=64):
        # 14143 -> 9919 total instructions, a 30% cut, pushing this well
        # back under BABY_RAT's 17500 bytecode/turn budget. Left-to-right
        # evaluation order (and thus the exact floating-point result) is
        # identical to the old form -- this only changes the bytecode shape.
        for j in range(hidden):
            terms = " + ".join(f"{fmt(w_sh[j, i].item())} * in{i}" for i in range(state_dim))
            b = b_sh[j].item()
            f.write(f"        float h{j} = {terms} + {fmt(b)};\n")
            f.write(f"        if (h{j} < 0) h{j} = 0;\n\n")

        for j in range(NUM_TILES):
            f.write(f"        out0[{j}] = 0")
            for i in range(hidden):
                w = w_out[j, i].item()
                f.write(f" + {fmt(w)} * h{i}")
            b = b_out[j].item()
            f.write(f" + {fmt(b)};\n")

        f.write("\n        return out0;\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"NeuralNet written to {path}", flush=True)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset_path = os.path.join(BOTS_DIR, "dataset_enemy.pt")
    data = torch.load(dataset_path, weights_only=True)
    states = data["states"]
    tilescores = data["tilescores"]
    samples = list(zip(states.tolist(), tilescores.tolist()))
    print(f"Loaded {len(samples)} enemy-only samples", flush=True)

    random.shuffle(samples)
    print(f"Using all {len(samples)} samples", flush=True)

    print(f"\n[1/2] Training teacher on {len(samples)} samples ({TEACHER_EPOCHS} epochs)...", flush=True)
    teacher, teacher_loss, t_time = train_teacher(samples, device)
    print(f"  Teacher final loss: {teacher_loss:.6f}  [{t_time:.0f}s]", flush=True)
    torch.save(teacher.cpu().state_dict(), TEACHER_MODEL_PATH)
    print(f"  Saved {TEACHER_MODEL_PATH} ({STATE_DIM}-dim state, 256/128/64 hidden) -- rl_train.py's RL warm-start", flush=True)
    teacher.to(device)

    print(f"\n[2/2] Distilling student from teacher ({STUDENT_EPOCHS} epochs)...", flush=True)
    student, student_loss, s_time = distill(teacher, samples, device, hidden_dim=HIDDEN_DIM)
    print(f"  Student final distill loss: {student_loss:.6f}  [{s_time:.0f}s]", flush=True)

    torch.save(student.state_dict(), MODEL_PATH)
    print(f"  Saved qnet.pth ({STATE_DIM}-dim state, {HIDDEN_DIM} hidden)", flush=True)

    # Deliberately NOT re-exporting NeuralNet.java into econ5/ or learner/
    # this time: both packages' own buildState() still only produce the
    # original 18-dim state, so a 40-dim NeuralNet.java would silently
    # break them at runtime (array-length mismatch, not a compile error) --
    # this run's only job is producing a correctly-shaped qnet.pth for
    # rl_train.py's build_warm_start_model() to load.

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Teacher: {t_time:.0f}s, Student: {s_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
