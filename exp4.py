# Experiment 4 Learning rate evolution
import os
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from implementation import initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

# Exp 4 Hyperparameters
SEED = 1
EPOCHS = 10000
REG_LAMBDA = 0.00
TARGET_COST = 0.01
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)  # create the figures folder if it doesn't exist yet
lr_values = sorted(set(cfg.lr))  # remove duplicates and sort from small to large

W1_init, b1_init, W2_init, b2_init = initialize_weights(seed=SEED)

cost_data = {}
converged_epochs = {}

for lr in lr_values:
    W1_trained, b1_trained, W2_trained, b2_trained, cost_history, _ = train(
        x = x,
        y = y,
        W1 = W1_init,
        b1 = b1_init,
        W2 = W2_init,
        b2 = b2_init,
        lr = lr,
        epochs = EPOCHS,
        reg_lambda = REG_LAMBDA,
        method = 'batch',
        batch_size = m )

    converged_epoch = None
    for epoch, loss in enumerate(cost_history):
        if loss <= TARGET_COST:
            converged_epoch = epoch + 1
            break

    cost_data[lr] = cost_history
    converged_epochs[lr] = converged_epoch
    print(f"lr = {lr:<5} -> converged at epoch: {converged_epoch}, final cost: {cost_history[-1]:.6f}")

# Plot 1: Cost vs. Epochs for every learning rate (one distinct color per lr)
colors = plt.cm.tab10.colors  # 10 clearly different colors: blue, orange, green, red, purple, ...

conv_marks = []  # (epoch, cost, color) of every convergence dot

plt.figure(figsize=(8, 5))
for i, lr in enumerate(lr_values):
    epochs_axis = np.arange(1, len(cost_data[lr]) + 1)
    linestyle = "-" if i < len(colors) else "--"  # after 10 lrs the colors repeat, so switch to dashed
    plt.plot(epochs_axis, cost_data[lr], label=f"lr = {lr}", color=colors[i % len(colors)],
             ls=linestyle, linewidth=2.0)

    # dot where the line first reaches the target cost
    if converged_epochs[lr] is not None:
        ep = converged_epochs[lr]
        plt.plot(ep, cost_data[lr][ep - 1], marker="o", markersize=7, color=colors[i % len(colors)],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        conv_marks.append((ep, cost_data[lr][ep - 1], colors[i % len(colors)]))

plt.axhline(y=TARGET_COST, color="black", ls=":", linewidth=1.5, label="Target cost")
plt.yscale("log")

# vertical line from each dot down to the x axis, with the exact epoch written under the axis
ax = plt.gca()
y_bottom = ax.get_ylim()[0]
row_last_ep = []  # last epoch written on each row of labels
for ep, c, color in sorted(conv_marks):
    ax.vlines(ep, y_bottom, c, color=color, ls="--", linewidth=1.0)
    # epochs that are close together would overlap, so put the label on the first row with enough space
    row = next((r for r, last in enumerate(row_last_ep) if ep - last >= 400), len(row_last_ep))
    if row == len(row_last_ep):
        row_last_ep.append(ep)
    row_last_ep[row] = ep
    ax.annotate(str(ep), xy=(ep, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -20 - 9 * row), textcoords="offset points",
                ha="center", va="top", fontsize=7, color=color, fontweight="bold")
ax.set_ylim(bottom=y_bottom)

plt.title("Cost vs. Epochs for different learning rates")
plt.xlabel("Epochs", labelpad=14 + 9 * len(row_last_ep))
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "lr_cost_curves.png"), dpi=300)

# Plot 2: Epochs needed to reach the target cost vs. learning rate
conv_lrs = [lr for lr in lr_values if converged_epochs[lr] is not None]
conv_eps = [converged_epochs[lr] for lr in conv_lrs]
fail_lrs = [lr for lr in lr_values if converged_epochs[lr] is None]

plt.figure(figsize=(8, 5))
plt.plot(conv_lrs, conv_eps, marker="o", markersize=8, color="tab:blue", linewidth=2.0, label="Converged")
if fail_lrs:
    plt.scatter(fail_lrs, [EPOCHS] * len(fail_lrs), marker="x", s=80, color="tab:red",
                label=f"Not converged within {EPOCHS} epochs")

plt.xscale("log")
plt.title("Epochs to reach target cost vs. Learning rate")
plt.xlabel("Learning rate (Log Scale)")
plt.ylabel("Epochs")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "lr_convergence.png"), dpi=300)
plt.show()
