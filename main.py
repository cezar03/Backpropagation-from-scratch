import config as cfg
from implementation import cost, forwardprop, gradient_descent
import numpy as np

W1, b1, W2, b2 = gradient_descent(
    cfg.data, 
    cfg.Y, 
    cfg.m, 
    cfg.lr, 
    cfg.n_iter, 
    cfg.W1, 
    cfg.b1, 
    cfg.W2, 
    cfg.b2, 
    cfg.reg_lambda)

final_cost = cost(cfg.data, cfg.Y, cfg.m, W1, b1, W2, b2, cfg.reg_lambda)
print(f"Final cost: {final_cost:.6f}\n")


print("Trained Network Evaluation:")
for i in range(cfg.m):
  a1, a2, a3 = forwardprop(cfg.data[i], W1, b1, W2, b2)
  print(
      f"Input: {cfg.data[i].astype(int)} | "
      f"Hidden: {np.round(a2.flatten(), 2)} | "
      f"Output: {np.round(a3.flatten(), 2)}"
  )
