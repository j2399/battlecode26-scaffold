#!/usr/bin/env python3
import subprocess, re, os, sys, torch

BOTS_DIR = "/Users/csproj/Desktop/rbattlecode"
MAX_ROUNDS = 100
STATE_DIM = 18

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
        raw[0] / 64.0,
        raw[1] / 64.0,
        raw[2] / 64.0,
        raw[3] / 64.0,
        raw[4] / 100.0,
        raw[5] / 8.0,
        raw[6] / 64.0,
        raw[7] / 64.0,
        raw[8] / 100.0,
        raw[9] / 8.0,
        raw[10] / 64.0,
        raw[11] / 64.0,
        raw[12] / 100.0,
        raw[13] / 8.0,
        raw[14] / 8.0,
        min(raw[15], 20) / 20.0,
        min(raw[16], 20) / 20.0,
        raw[17],
    ]
    value = float(parts[1]) if parts[1].strip() else 0.0
    tilescores = [int(x) for x in parts[2].split(",") if x.strip()]
    while len(tilescores) < 9:
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
        ["./gradlew", "runJavaLocal"], cwd=BOTS_DIR,
        capture_output=True, text=True,         timeout=600,
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

CHECKPOINT = os.path.join(BOTS_DIR, "collect_checkpoint.pt")

all_samples = []
processed = set()
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, weights_only=True)
    all_samples = ckpt["samples"]
    processed = set(ckpt["maps"])
    print(f"Resuming from checkpoint: {len(all_samples)} samples, {len(processed)} maps done", flush=True)

for map_name in MAPS:
    if map_name in processed:
        print(f"\n--- {map_name}: SKIP (already done) ---", flush=True)
        continue
    print(f"\n--- {map_name}: econ5 vs econ5 ---", flush=True)
    samples = run_match(map_name, "econ5", "econ5", MAX_ROUNDS)
    print(f"  {len(samples)} samples", flush=True)
    all_samples.extend(samples)
    processed.add(map_name)
    torch.save({"samples": all_samples, "maps": list(processed)}, CHECKPOINT)
    print(f"  checkpoint saved", flush=True)

print(f"\nTotal: {len(all_samples)} samples", flush=True)

states = torch.tensor([x[0] for x in all_samples])
tilescores = torch.tensor([x[1] for x in all_samples])
values = torch.tensor([x[2] for x in all_samples])

torch.save({"states": states, "tilescores": tilescores, "values": values},
           os.path.join(BOTS_DIR, "dataset.pt"))
print(f"Dataset saved to {os.path.join(BOTS_DIR, 'dataset.pt')}", flush=True)

# Filter for enemy-only samples (at least one enemy health > 0)
enemy_mask = (states[:, 4] > 0) | (states[:, 8] > 0) | (states[:, 12] > 0)
enemy_states = states[enemy_mask]
enemy_tilescores = tilescores[enemy_mask]
enemy_values = values[enemy_mask]
print(f"Enemy-only: {enemy_mask.sum().item()} / {len(states)} samples", flush=True)
torch.save({"states": enemy_states, "tilescores": enemy_tilescores, "values": enemy_values},
           os.path.join(BOTS_DIR, "dataset_enemy.pt"))
print(f"Enemy dataset saved to {os.path.join(BOTS_DIR, 'dataset_enemy.pt')}", flush=True)

# Cleanup checkpoint
os.remove(CHECKPOINT)
