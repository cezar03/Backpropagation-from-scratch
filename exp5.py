import os
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from implementation import initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

SEED = 1
EPOCHS = 10000
REG_LAMBDA = 0.00
TARGET_COST = 0.01
FIG_DIR = os.path.join("figures", "exp5")
LR_VALUE = 1
os.makedirs(FIG_DIR, exist_ok=True)


weights_t = [
    ("Random", initialize_weights(seed=SEED)),
    ("Zero", (np.zeros((3, 8)), np.zeros((3, 1)), np.zeros((8, 3)), np.zeros((8, 1)))),
    ("Constant 0.5", (np.full((3, 8), 0.5), np.full((3, 1), 0.5), np.full((8, 3), 0.5), np.full((8, 1), 0.5))),

]

cost_data = {}
converged_epochs = {}

for name, (W1_init, b1_init, W2_init, b2_init) in weights_t:
    W1_trained, b1_trained, W2_trained, b2_trained, cost_history, _ = train(
        x = x,
        y = y,
        W1 = W1_init,
        b1 = b1_init,
        W2 = W2_init,
        b2 = b2_init,
        lr = LR_VALUE,
        epochs = EPOCHS,
        reg_lambda = REG_LAMBDA,
        method = 'batch',
        batch_size = m )

    converged_epoch = None
    for epoch, loss in enumerate(cost_history):
        if loss <= TARGET_COST:
            converged_epoch = epoch + 1
            break

    cost_data[name] = cost_history
    converged_epochs[name] = converged_epoch
    print(f"Weight Initialization: {name:<15} -> converged at epoch: {converged_epoch}, final cost: {cost_history[-1]:.6f}")


colors = plt.cm.tab10.colors
conv_marks = []  
plt.figure(figsize=(8, 5))
for i, (name, _) in enumerate(weights_t):
    epochs_axis = np.arange(1, len(cost_data[name]) + 1)
    plt.plot(epochs_axis, cost_data[name], label=name, color=colors[i % len(colors)], linewidth=2.0)

    if converged_epochs[name] is not None:
        ep = converged_epochs[name]
        plt.plot(ep, cost_data[name][ep - 1], marker="o", markersize=7, color=colors[i % len(colors)],
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        conv_marks.append((ep, cost_data[name][ep - 1], colors[i % len(colors)]))

plt.axhline(y=TARGET_COST, color="black", ls=":", linewidth=1.5, label="Target cost")
plt.yscale("log")


y_bottom = ax.get_ylim()[0]
row_last_ep = []  
for ep, c, color in sorted(conv_marks):
    ax.vlines(ep, y_bottom, c, color=color, ls="--", linewidth=1.0)
    
    row = next((r for r, last in enumerate(row_last_ep) if ep - last >= 400), len(row_last_ep))
    if row == len(row_last_ep):
        row_last_ep.append(ep)
    row_last_ep[row] = ep
    ax.annotate(str(ep), xy=(ep, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -20 - 9 * row), textcoords="offset points",
                ha="center", va="top", fontsize=7, color=color, fontweight="bold")
ax.set_ylim(bottom=y_bottom)

plt.title("Cost vs. Epochs for different weight initializations")
plt.xlabel("Epochs", labelpad=14 + 9 * len(row_last_ep))
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "init_cost_curves.png"), dpi=300)
plt.show()
