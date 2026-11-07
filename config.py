
import numpy as np
data = np.array([[0,0,0,0,0,0,0,1]
                 ,[0,0,0,0,0,0,1,0]
                 ,[0,0,0,0,0,1,0,0]
                 ,[0,0,0,0,1,0,0,0]
                 ,[0,0,0,1,0,0,0,0]
                 ,[0,0,1,0,0,0,0,0]
                 ,[0,1,0,0,0,0,0,0]
                 ,[1,0,0,0,0,0,0,0]])

Y = data
m = 8
epochs = 10000
eps = 0.01
np.random.seed(0)
W1 = np.random.randn(3, 8) * eps    # row i = the 8 incoming weights of hidden unit i
b1 = np.random.randn(3, 1) * eps
W2 = np.random.randn(8, 3) * eps    # row i = the 3 incoming weights of output unit i
b2 = np.random.randn(8, 1) * eps
lr=[1, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]  #learning rate
n_iter=30000  #number of epochs
reg_lambda=[0.00, 0.01, 0.1, 1, 2, 5, 10, 20, 50]