import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import os
from implementation import initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

# Exp 1 Hyperparaemeters
SEED = 1
LR = 1
EPOCHS = 10000
REG_LAMBDA = 0.00
FIG_DIR = os.path.join("figures", "exp1")
os.makedirs(FIG_DIR, exist_ok=True)
W1_init, b1_init, W2_init, b2_init = initialize_weights(seed=SEED)



W1_trained, b1_trained, W2_trained, b2_trained, cost_history, lr_history = train(
    x = x, 
    y = y, 
    W1 = W1_init,
    b1 = b1_init,
    W2 = W2_init,
    b2 = b2_init,
    lr = LR,
    epochs = EPOCHS,
    reg_lambda = REG_LAMBDA,
    method = 'batch',
    batch_size = m )

epochs_axis = np.arange(1, len(cost_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_axis, cost_history, label="Batch GD", color="tab:blue", linewidth=2.0)

plt.yscale("log")
plt.title("Cost vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Scale)")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()


plt.savefig(os.path.join(FIG_DIR, "batch_gd_cost_curve.png"), dpi=300)
plt.show()