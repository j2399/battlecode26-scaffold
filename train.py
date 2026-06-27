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
STATE_DIM = 18
NUM_TILES = 9
TEACHER_EPOCHS = 1500
STUDENT_EPOCHS = 2250
BATCH_SIZE = 2048
LR = 1e-3
MODEL_PATH = os.path.join(BOTS_DIR, "qnet.pth")

def weights_java_path(package="econ5"):
    return os.path.join(BOTS_DIR, "src", package, "QNetWeights.java")

def neuralnet_java_path(package="econ5"):
    return os.path.join(BOTS_DIR, "src", package, "NeuralNet.java")


class TeacherTileNet(nn.Module):
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

    state = [
        raw[0] / 64.0,                # x
        raw[1] / 64.0,                # y
        raw[2] / 64.0,                # e1 dx
        raw[3] / 64.0,                # e1 dy
        raw[4] / 100.0,               # e1 health
        raw[5] / 8.0,                 # e1 dir
        raw[6] / 64.0,                # e2 dx
        raw[7] / 64.0,                # e2 dy
        raw[8] / 100.0,               # e2 health
        raw[9] / 8.0,                 # e2 dir
        raw[10] / 64.0,               # e3 dx
        raw[11] / 64.0,               # e3 dy
        raw[12] / 100.0,              # e3 health
        raw[13] / 8.0,                # e3 dir
        raw[14] / 8.0,                # own dir
        min(raw[15], 20) / 20.0,      # move cd
        min(raw[16], 20) / 20.0,      # action cd
        raw[17],                       # carrying (0/1)
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

        for j in range(hidden):
            f.write(f"        float h{j} = 0;\n")
            for i in range(state_dim):
                w = w_sh[j, i].item()
                f.write(f"        h{j} += {fmt(w)} * in{i};\n")
            b = b_sh[j].item()
            f.write(f"        h{j} += {fmt(b)};\n")
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

    print(f"\n[2/2] Distilling student from teacher ({STUDENT_EPOCHS} epochs)...", flush=True)
    student, student_loss, s_time = distill(teacher, samples, device)
    print(f"  Student final distill loss: {student_loss:.6f}  [{s_time:.0f}s]", flush=True)

    torch.save(student.state_dict(), MODEL_PATH)
    print(f"  Saved qnet.pth", flush=True)

    export_neuralnet_java(student, neuralnet_java_path("econ5"), "econ5")
    print(f"  Exported NeuralNet.java to econ5/", flush=True)

    export_neuralnet_java(student, neuralnet_java_path("learner"), "learner")
    print(f"  Exported NeuralNet.java to learner/", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Teacher: {t_time:.0f}s, Student: {s_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
