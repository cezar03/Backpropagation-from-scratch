import config as cfg
from implementation import cost, forwardprop, train
import numpy as np

LR = 1
REG_LAMBDA = 0.00

W1, b1, W2, b2, cost_history, _ = train(
    x = cfg.data,
    y = cfg.Y,
    W1 = cfg.W1,
    b1 = cfg.b1,
    W2 = cfg.W2,
    b2 = cfg.b2,
    lr = LR,
    epochs = cfg.n_iter,
    reg_lambda = REG_LAMBDA,
    method = 'batch',
    batch_size = cfg.m )

final_cost = cost(cfg.data, cfg.Y, cfg.m, W1, b1, W2, b2, REG_LAMBDA)
print(f"Final cost: {final_cost:.6f}\n")


print("Trained Network Evaluation:")
for i in range(cfg.m):
  a1, a2, a3 = forwardprop(cfg.data[i], W1, b1, W2, b2)
  print(
      f"Input: {cfg.data[i].astype(int)} | "
      f"Hidden: {np.round(a2.flatten(), 2)} | "
      f"Output: {np.round(a3.flatten(), 2)}"
  )
