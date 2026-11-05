
import numpy as np


def initialize_weights(input_dim=8, hidden_dim=3, output_dim=8, eps= 0.01, seed=0):
    np.random.seed(seed)
    W1 = np.random.randn(hidden_dim, input_dim) * eps
    b1 = np.random.randn(hidden_dim, 1) * eps
    W2 = np.random.randn(output_dim, hidden_dim) * eps
    b2 = np.random.randn(output_dim, 1) * eps
    return W1, b1, W2, b2

def sigmoid(x):
    
 return  1/(1+np.exp(-x))

def forwardprop(x, W1, b1, W2, b2):
    a1 = x.reshape(-1,1)  #reshape the input to (8,1)
    a2 = sigmoid(W1 @ a1 + b1)  #values of the 3 hidden nodes after activation, a2=matrix of them
    a3 = sigmoid(W2 @ a2 + b2) #values of the 8 output nodes after acivation, a3=matrix of them
    return a1, a2, a3

def backprop(x, y, W1, b1, W2, b2):
    a1, a2, a3 = forwardprop(x, W1, b1, W2, b2)
    y = y.reshape(-1,1)  #reshape the output to (8,1)

    d3 = (a3 - y) * (a3 * (1-a3)) #output layer error term, error * last term is the derivative of the sigmoid
    d2 = (W2.T @ d3) * (a2 * (1-a2)) #hidden layer error term, same as above

    nabla_W1 = d2 @ a1.T
    nabla_W2 = d3 @ a2.T
    nabla_b1 = d2
    nabla_b2 = d3
    return nabla_W1, nabla_b1, nabla_W2, nabla_b2

def gradient_descent(x_batch, y_batch, lr, W1, b1, W2, b2, reg_lambda):

    batch_size = len(x_batch)
    
    dW1 = np.zeros_like(W1)     #set the deltas (gradient accumulators) to 0 when starting a new epoch.
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    for i in range(batch_size):
        gW1, gb1, gW2, gb2 = backprop(x_batch[i],y_batch[i], W1, b1, W2, b2)
        dW1 += gW1
        db1 += gb1
        dW2 += gW2
        db2 += gb2
    W1 = W1 - lr * (dW1 / batch_size + reg_lambda * W1)
    b1 = b1 - lr * db1 / batch_size
    W2 = W2 - lr * (dW2 / batch_size + reg_lambda * W2)
    b2 = b2 - lr * db2 / batch_size

    return W1, b1, W2, b2


def cost(x, y, m, W1, b1, W2, b2, reg_lambda):      #only used for tests, no use is the training pipeline
    total_cost=0
    for i in range(m):
        a1, a2 ,a3=forwardprop(x[i], W1, b1, W2, b2)
        y_col = y[i].reshape(-1,1)  #reshape the output to (8,1)
        total_cost = total_cost + 1/2*np.sum((a3-y_col)**2) 

    reg_penalty = 0.5 * reg_lambda * (np.sum((W1**2)) + np.sum((W2**2)))  #regularization penalty

    return total_cost/m + reg_penalty


def train(x, y, W1, b1, W2, b2, lr, epochs, reg_lambda=0, method='batch',
          batch_size=2):

    m = len(x) # Total number of training samples

    if method == 'batch':
        bs = m
    elif method == "sgd":
        bs = 1
    elif method == "mini-batch":
        bs = batch_size
    else:
        raise ValueError(f"Wrong method {method}")

    W1_curr, b1_curr = W1.copy(), b1.copy()
    W2_curr, b2_curr = W2.copy(), b2.copy()
    cost_history = [] # Average loss throught the training process

    for epoch in range(epochs):
        for i in range(0, m, bs):
            x_batch = x[i: i + bs]
            y_batch = y[i: i + bs]

            W1_curr, b1_curr, W2_curr, b2_curr = gradient_descent(
                x_batch, y_batch, lr, W1_curr, b1_curr, W2_curr, b2_curr, reg_lambda)

        epoch_cost = cost(x, y, m, W1_curr, b1_curr, W2_curr, b2_curr, reg_lambda)
        cost_history.append(epoch_cost)


    return  W1_curr, b1_curr, W2_curr, b2_curr, cost_history

