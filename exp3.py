# Experiment 3: W1, b1, W2, b2 analysis
import os
import numpy as np
import config as cfg
from implementation import forwardprop, initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

SEED = 1
LR = 1
REG_LAMBDA = 0.00
FIG_DIR = os.path.join("figures", "exp3")
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(FIG_DIR, "exp3_results.txt")

# Check points
checkpoints = [0, 1, 100, 500, 1500, 10000]

# Intialize weights
W1, b1, W2, b2 = initialize_weights(seed=SEED)

snapshots = {}

snapshots[0] = {
    "W1": W1.copy(), "b1": b1.copy(),
    "W2": W2.copy(), "b2": b2.copy()
}

current_epoch = 0
for target_epoch in checkpoints[1:]:
    steps = target_epoch - current_epoch
    W1, b1, W2, b2, _, _ = train(
        x = x,
        y = y,
        W1 = W1,
        b1 = b1,
        W2 = W2,
        b2 = b2,
        lr = LR,
        epochs = steps,
        reg_lambda = REG_LAMBDA,
        method = 'batch',
        batch_size = m )
    current_epoch = target_epoch
    snapshots[target_epoch] = {
        "W1": W1.copy(), "b1": b1.copy(),
        "W2": W2.copy(), "b2": b2.copy()
    }

delta_b2 = np.mean(np.abs(snapshots[1]["b2"] - snapshots[0]["b2"]))
delta_W2 = np.mean(np.abs(snapshots[1]["W2"] - snapshots[0]["W2"]))
delta_b1 = np.mean(np.abs(snapshots[1]["b1"] - snapshots[0]["b1"]))
delta_W1 = np.mean(np.abs(snapshots[1]["W1"] - snapshots[0]["W1"]))

with open(RESULTS_FILE, "w") as f:

    print(" 1. PARAMETERS SHIFT AFTER 1 EPOCH", file=f)
    print(f"b2 shift: {delta_b2:.8f}", file=f)
    print(f"W2 shift: {delta_W2:.8f}", file=f)
    print(f"b1 shift: {delta_b1:.8f}", file=f)
    print(f"W1 shift: {delta_W1:.8f}", file=f)
    print(file=f)

    print(" 2. PARAMETERS EVOLUTION", file=f)

    for param_name in ["b2", "W2", "b1", "W1"]:
        print(f"\n--- Evolution of {param_name} ---", file=f)
        for ep in checkpoints:
            arr_flat = np.round(snapshots[ep][param_name].flatten(), 3)
            print(f"{param_name} (Epoch {ep:<5}):\n{arr_flat}", file=f)

    print("\n 3. ACTIVATION EVOLUTION", file=f)

    for i in range(m):
        y_flat = y[i].flatten()
        print(f"\n---SAMPLE {i} (Target y: {y_flat}) ---", file=f)

        print("\nHidden Layer a2 (3 neurons):", file=f)
        for ep in checkpoints:
            snap = snapshots[ep]
            _, a2_i, _ = forwardprop(x[i], snap["W1"], snap["b1"], snap["W2"], snap["b2"])
            print(f"  Epoch {ep:<5} -> a2: {np.round(a2_i.flatten(), 3)}", file=f)

        print("\nOutput Layer a3 (8 neurons):", file=f)
        for ep in checkpoints:
            snap = snapshots[ep]
            _, _, a3_i = forwardprop(x[i], snap["W1"], snap["b1"], snap["W2"], snap["b2"])
            print(f"  Epoch {ep:<5} -> a3: {np.round(a3_i.flatten(), 3)}", file=f)

print(f"Results written to {RESULTS_FILE}")
