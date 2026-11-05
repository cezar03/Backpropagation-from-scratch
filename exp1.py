import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from implementation import initialize_weights, train

x = cfg.data
y = cfg.Y
m = len(x)

# Exp 1 Hyperparaemeters
SEED = 1
LR = 1
EPOCHS = 10000
REG_LAMBDA = 0.00
TARGET_COST = 0.005

methods_to_test = [ 
    ("Batch GD", "batch", m),
    ("SGD", "sgd", 1),
    ("Mini-batch GD (bs = 2)", "mini-batch", 2),
    ("Mini-batch GD (bs = 4)", "mini-batch", 4)
]

W1_init, b1_init, W2_init, b2_init = initialize_weights(seed=SEED)

results = []
cost_data = {}
updates_data = {}


for label, method_name, batch_size in methods_to_test:
    updates_per_epoch = m // batch_size


    W1_trained, b1_trained, W2_trained, b2_trained, cost_history = train(
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

    if converged_epoch is not None:
        total_updates = converged_epoch * updates_per_epoch
    else:
        total_updates = EPOCHS * updates_per_epoch

# Plots
    epochs_axis = np.arange(1, len(cost_history) + 1)
    cost_data[label] = (epochs_axis, cost_history)
    updates_data[label] = (epochs_axis, epochs_axis * updates_per_epoch)




# Plot 1: Cost vs. Epochs
plt.figure(figsize=(8, 5))
for label in cost_data:
    plt.plot(cost_data[label][0], cost_data[label][1], label=label)

plt.axhline(y=TARGET_COST)
plt.title("Cost vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Scale)")
plt.yscale("log")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend()

# Plot 2: Zoom In epochs 0 to 2000
plt.figure(figsize=(8, 5))
for label in cost_data:
  plt.plot(cost_data[label][0], cost_data[label][1], label=label, linewidth=1.8)

plt.xlim(0, 2000)  # Zooms into the saddle point transition
plt.ylim(0.01, 1.0)
plt.title("Plateau Breakout Detail (Epochs 0 – 2,000)", fontsize=13)
plt.xlabel("Epochs", fontsize=11)  #
plt.ylabel("Cost (Log Scale)", fontsize=11)  #
plt.yscale("log")  #
plt.grid(True, which="both", ls=":", alpha=0.5)  #
plt.legend(fontsize=10)  #
plt.tight_layout()

# Plot 3: Cost vs. Updates
plt.figure(figsize=(8, 5))
for label in updates_data:
    plt.plot(updates_data[label][0], updates_data[label][1], label=label)

plt.axhline(y=TARGET_COST)
plt.title("updates_x vs. epochs_x")
plt.xlabel("epochs_x")
plt.ylabel("updates_x")
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
