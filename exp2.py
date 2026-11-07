import os
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from implementation import initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

# Exp 2 Hyperparaemeters
SEED = 1
LR = 1
EPOCHS = 10000
REG_LAMBDA = 0.00
TARGET_COST = 0.01
FIG_DIR = os.path.join("figures", "exp2")
os.makedirs(FIG_DIR, exist_ok=True)

methods_to_test = [
    ("Batch GD", "batch", m),
    ("SGD", "sgd", 1),
    ("Mini-batch GD (bs = 2)", "mini-batch", 2),
    ("Mini-batch GD (bs = 4)", "mini-batch", 4)
]
colors = plt.cm.tab10.colors

W1_init, b1_init, W2_init, b2_init = initialize_weights(seed=SEED)

cost_data = {}
updates_per_epoch = {}
converged_epochs = {}

for label, method_name, batch_size in methods_to_test:
    W1_trained, b1_trained, W2_trained, b2_trained, cost_history, _ = train(
        x = x,
        y = y,
        W1 = W1_init,
        b1 = b1_init,
        W2 = W2_init,
        b2 = b2_init,
        lr = LR,
        epochs = EPOCHS,
        reg_lambda = REG_LAMBDA,
        method = method_name,
        batch_size = batch_size )

    converged_epoch = None
    for epoch, loss in enumerate(cost_history):
        if loss <= TARGET_COST:
            converged_epoch = epoch + 1
            break

    cost_data[label] = cost_history
    updates_per_epoch[label] = m // batch_size
    converged_epochs[label] = converged_epoch
    if converged_epoch is not None:
        total_updates = converged_epoch * updates_per_epoch[label]
        print(f"{label:<24} -> converged at epoch: {converged_epoch}, weight updates: {total_updates}, "
              f"final cost: {cost_history[-1]:.6f}")
    else:
        print(f"{label:<24} -> converged at epoch: None, final cost: {cost_history[-1]:.6f}")


def mark_convergence(ax, conv_marks):
    y_bottom = ax.get_ylim()[0]
    row_last = []
    min_gap = 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for xv, c, color in sorted(conv_marks):
        ax.vlines(xv, y_bottom, c, color=color, ls="--", linewidth=1.0)
        row = next((r for r, last in enumerate(row_last) if xv - last >= min_gap), len(row_last))
        if row == len(row_last):
            row_last.append(xv)
        row_last[row] = xv
        ax.annotate(str(xv), xy=(xv, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -20 - 9 * row), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=color, fontweight="bold")
    ax.set_ylim(bottom=y_bottom)
    return len(row_last)


# Plot 1: Cost vs. Epochs
conv_marks = []

plt.figure(figsize=(8, 5))
for i, (label, _, _) in enumerate(methods_to_test):
    epochs_axis = np.arange(1, len(cost_data[label]) + 1)
    plt.plot(epochs_axis, cost_data[label], label=label, color=colors[i], linewidth=2.0)

    ep = converged_epochs[label]
    if ep is not None:
        plt.plot(ep, cost_data[label][ep - 1], marker="o", markersize=7, color=colors[i],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        conv_marks.append((ep, cost_data[label][ep - 1], colors[i]))

plt.axhline(y=TARGET_COST, color="black", ls=":", linewidth=1.5, label="Target cost")
plt.yscale("log")
n_rows = mark_convergence(plt.gca(), conv_marks)

plt.title("Cost vs. Epochs")
plt.xlabel("Epochs", labelpad=14 + 9 * n_rows)
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "methods_cost_vs_epochs.png"), dpi=300)

# Plot 2: Zoom In epochs 0 to 2000
plt.figure(figsize=(8, 5))
for i, (label, _, _) in enumerate(methods_to_test):
    epochs_axis = np.arange(1, len(cost_data[label]) + 1)
    plt.plot(epochs_axis, cost_data[label], label=label, color=colors[i], linewidth=2.0)

plt.axhline(y=TARGET_COST, color="black", ls=":", linewidth=1.5, label="Target cost")
plt.xlim(0, 2000)  # Zooms into the saddle point transition
plt.ylim(TARGET_COST / 2, 1.0)
plt.yscale("log")
plt.title("Plateau Breakout Detail (Epochs 0 - 2000)")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "methods_plateau_zoom.png"), dpi=300)

# Plot 3: Cost vs. Updates
conv_marks = []

plt.figure(figsize=(8, 5))
for i, (label, _, _) in enumerate(methods_to_test):
    updates_axis = np.arange(1, len(cost_data[label]) + 1) * updates_per_epoch[label]
    plt.plot(updates_axis, cost_data[label], label=label, color=colors[i], linewidth=2.0)

    ep = converged_epochs[label]
    if ep is not None:
        n_updates = ep * updates_per_epoch[label]
        plt.plot(n_updates, cost_data[label][ep - 1], marker="o", markersize=7, color=colors[i],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        conv_marks.append((n_updates, cost_data[label][ep - 1], colors[i]))

plt.axhline(y=TARGET_COST, color="black", ls=":", linewidth=1.5, label="Target cost")
plt.yscale("log")
n_rows = mark_convergence(plt.gca(), conv_marks)

plt.title("Cost vs. Weight Updates")
plt.xlabel("Weight updates", labelpad=14 + 9 * n_rows)
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "methods_cost_vs_updates.png"), dpi=300)
plt.show()
