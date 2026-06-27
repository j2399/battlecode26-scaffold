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
TEACHER_EPOCHS = 2000
STUDENT_EPOCHS = 3000
BATCH_SIZE = 2048
LR = 1e-3
MODEL_PATH = os.path.join(BOTS_DIR, "qnet.pth")

def weights_java_path(package="learner"):
    return os.path.join(BOTS_DIR, "src", package, "QNetWeights.java")

def neuralnet_java_path(package="learner"):
    return os.path.join(BOTS_DIR, "src", package, "NeuralNet.java")


class TeacherQNet(nn.Module):
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
        self.q_moved  = nn.Linear(64, 9)
        self.q_facing = nn.Linear(64, 9)

    def forward(self, x):
        h = self.shared(x)
        return self.q_moved(h), self.q_facing(h)


class StudentQNet(nn.Module):
    def __init__(self, state_dim, hidden=36):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.q_moved  = nn.Linear(hidden, 9)
        self.q_facing = nn.Linear(hidden, 9)

    def forward(self, x):
        h = self.shared(x)
        return self.q_moved(h), self.q_facing(h)


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
    act = [int(x) for x in parts[2].split(",") if x.strip()]
    while len(act) < 2:
        act.append(0)
    return state, tuple(act[:2]), value, robot_id, capture_flag


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
            state, action, value, robot_id, capture_flag = parsed
            if capture_flag and robot_id in prev_by_id:
                prev_idx = prev_by_id[robot_id]
                old_state, old_action, old_value = samples[prev_idx]
                samples[prev_idx] = (old_state, old_action, old_value - 80)
            prev_by_id[robot_id] = len(samples)
            samples.append((state, action, value))
    return samples


def train_teacher(replay_data, device, epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"Training teacher on {device}")
    t0 = time.time()
    model = TeacherQNet(STATE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        random.shuffle(replay_data)
        total_loss = 0.0
        for i in range(0, len(replay_data), batch_size):
            batch = replay_data[i : i + batch_size]
            states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(device)
            values = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device)

            qm, qf = model(states)

            actions = torch.tensor(
                [[x[1][0], x[1][1]] for x in batch],
                dtype=torch.long, device=device,
            )
            predicted = (qm.gather(1, actions[:, 0:1]) +
                         qf.gather(1, actions[:, 1:2])).squeeze(1)
            loss = loss_fn(predicted, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg = total_loss / len(replay_data)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Teacher Epoch {epoch+1}/{epochs}  loss={avg:.6f}  [{time.time()-t0:.0f}s]")

    return model, avg, time.time() - t0


def distill(teacher, replay_data, device, hidden_dim=36, epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"Distilling student (hidden={hidden_dim}) on {device}")
    t0 = time.time()
    student = StudentQNet(STATE_DIM, hidden_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    teacher.eval()
    for epoch in range(epochs):
        random.shuffle(replay_data)
        total_loss = 0.0
        for i in range(0, len(replay_data), batch_size):
            batch = replay_data[i : i + batch_size]
            states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(device)

            with torch.no_grad():
                t_qm, t_qf = teacher(states)

            s_qm, s_qf = student(states)
            loss = loss_fn(s_qm, t_qm) + loss_fn(s_qf, t_qf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg = total_loss / len(replay_data)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Student Epoch {epoch+1}/{epochs}  distill_loss={avg:.6f}  [{time.time()-t0:.0f}s]")

    return student, avg, time.time() - t0


def export_weights_java(model, path, package="learner"):
    named = {n: p.data for n, p in model.named_parameters()}
    tensor_order = [
        "shared.0.weight", "shared.0.bias",
        "q_moved.weight",  "q_moved.bias",
        "q_facing.weight", "q_facing.bias",
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
    print(f"Weights written to {path}")


def export_neuralnet_java(model, path, package="learner", state_dim=18, hidden=36):
    named = {n: p.data for n, p in model.named_parameters()}
    w_sh = named["shared.0.weight"]
    b_sh = named["shared.0.bias"]
    w_mv = named["q_moved.weight"]
    b_mv = named["q_moved.bias"]
    w_fc = named["q_facing.weight"]
    b_fc = named["q_facing.bias"]

    fmt = lambda v: f"{v:.12e}f"

    with open(path, "w") as f:
        f.write(f"package {package};\n\n")
        f.write("public class NeuralNet {\n")
        f.write(f"    private static final int STATE_DIM = {state_dim};\n")
        f.write(f"    private static final int HIDDEN = {hidden};\n\n")
        f.write("    private static final float[][] out = new float[2][9];\n\n")
        f.write("    public float[][] forward(float[] in) {\n")

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

        f.write("        float[] out0 = out[0];\n")
        for j in range(9):
            f.write(f"        out0[{j}] = 0")
            for i in range(hidden):
                w = w_mv[j, i].item()
                f.write(f" + {fmt(w)} * h{i}")
            b = b_mv[j].item()
            f.write(f" + {fmt(b)};\n")

        f.write("\n        float[] out1 = out[1];\n")
        for j in range(9):
            f.write(f"        out1[{j}] = 0")
            for i in range(hidden):
                w = w_fc[j, i].item()
                f.write(f" + {fmt(w)} * h{i}")
            b = b_fc[j].item()
            f.write(f" + {fmt(b)};\n")

        f.write("\n        return out;\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"NeuralNet written to {path}")


def copy_learner_package(version):
    src = os.path.join(BOTS_DIR, "src", "learner")
    dst = os.path.join(BOTS_DIR, "src", f"learner_v{version}")
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        if not fname.endswith(".java"):
            continue
        with open(os.path.join(src, fname)) as f:
            content = f.read()
        content = content.replace("package learner;", f"package learner_v{version};")
        content = content.replace("import learner.", f"import learner_v{version}.")
        with open(os.path.join(dst, fname), "w") as f:
            f.write(content)


STATE_PATH = os.path.join(BOTS_DIR, "training_state.pt")


def save_state(all_samples, iteration, target):
    torch.save({"all_samples": all_samples, "iteration": iteration, "target": target}, STATE_PATH)
    sys.stdout.flush()


def load_state():
    if os.path.exists(STATE_PATH):
        state = torch.load(STATE_PATH, weights_only=True)
        return state["all_samples"], state["iteration"], state["target"]
    return None, 0, 1


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    dataset_path = os.path.join(BOTS_DIR, "dataset_enemy.pt")
    data = torch.load(dataset_path, weights_only=True)
    states = data["states"]
    actions = data["actions"]
    values = data["values"]
    samples = list(zip(states.tolist(), [tuple(a.tolist()) for a in actions], values.tolist()))
    print(f"Loaded {len(samples)} enemy-only samples", flush=True)

    print(f"\n[1/2] Training teacher on {len(samples)} samples ({TEACHER_EPOCHS} epochs)...", flush=True)
    teacher, teacher_loss, t_time = train_teacher(samples, device)
    print(f"  Teacher final loss: {teacher_loss:.6f}  [{t_time:.0f}s]", flush=True)

    print(f"\n[2/2] Distilling student from teacher ({STUDENT_EPOCHS} epochs)...", flush=True)
    student, student_loss, s_time = distill(teacher, samples, device)
    print(f"  Student final distill loss: {student_loss:.6f}  [{s_time:.0f}s]", flush=True)

    torch.save(student.state_dict(), MODEL_PATH)
    print(f"  Saved qnet.pth", flush=True)

    export_neuralnet_java(student, neuralnet_java_path("micro_move_imitator"), "micro_move_imitator")
    print(f"  Exported NeuralNet.java to micro_move_imitator/", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Teacher: {t_time:.0f}s, Student: {s_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
