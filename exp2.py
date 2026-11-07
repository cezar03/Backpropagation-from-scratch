# Experiment 3: W1, b1, W2, b2 analysis
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from implementation import forwardprop, initialize_weights, train
import pandas as pd

x = cfg.data
y = cfg.Y
m = len(x)

# Check points
checkpoints = [0, 1, 100, 500, 1500, 10000]

# Intialize weights
W1, b1, W2, b2 = initialize_weights(seed=1)

snapshots = {}

snapshots[0] = {
    "W1": W1.copy(), "b1": b1.copy(),
    "W2": W2.copy(), "b2": b2.copy()
}

current_epoch = 0
for target_epoch in checkpoints[1:]:
    steps = target_epoch - current_epoch
    W1, b1, W2, b2, _ = train(
        x, y, W1, b1, W2, b2,
        lr=1.0, epochs=steps, reg_lambda=0.0,
        method="batch", batch_size=m
    )
    current_epoch = target_epoch
    snapshots[target_epoch] = {
        "W1": W1.copy(), "b1": b1.copy(),
        "W2": W2.copy(), "b2": b2.copy()
    }


cols = [f"{checkpoints[i]} -> {checkpoints[i+1]}" for i in range(len(checkpoints) - 1)]
shifts = {
    f"{checkpoints[i]} -> {checkpoints[i+1]}": [
        np.mean(np.abs(snapshots[checkpoints[i+1]][p] - snapshots[checkpoints[i]][p]))
        for p in ["b2", "W2", "b1", "W1"]
    ]
    for i in range(len(checkpoints) - 1)
}
df_shifts = pd.DataFrame(shifts, index=["b2", "W2", "b1", "W1"]).round(4)

delta_b2 = np.mean(np.abs(snapshots[1]["b2"] - snapshots[0]["b2"]))
delta_W2 = np.mean(np.abs(snapshots[1]["W2"] - snapshots[0]["W2"]))
delta_b1 = np.mean(np.abs(snapshots[1]["b1"] - snapshots[0]["b1"]))
delta_W1 = np.mean(np.abs(snapshots[1]["W1"] - snapshots[0]["W1"]))

with open("exp2_results.txt", "w") as f:

    print(" 1. PARAMETERS SHIFT AFTER 1 EPOCH", file=f)
    print(f"b2 shift: {delta_b2:.8f}", file=f)
    print(f"W2 shift: {delta_W2:.8f}", file=f)
    print(f"b1 shift: {delta_b1:.8f}", file=f)
    print(f"W1 shift: {delta_W1:.8f}", file=f)
    print(file=f)

    print(" 2. PARAMETERS EVOLUTION")

    for param_name in ["b2", "W2", "b1", "W1"]:
        print(f"\n--- Evolution of {param_name} ---", file=f)
        for ep in checkpoints:
            arr_flat = np.round(snapshots[ep][param_name].flatten(), 3)
            print(f"{param_name} (Epoch {ep:<5}):\n{arr_flat}", file=f)

    print(" 3. ACTIAVTION EVOLUTION")

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

    print("\n--- MEAN ABSOLUTE PARAMETER SHIFTS ---", file=f)
    print(df_shifts.to_string(), file=f)