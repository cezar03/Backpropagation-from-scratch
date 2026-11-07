# Backpropagation

## Requirements

- Python 3.10 or newer (tested with Python 3.13)
- NumPy, Matplotlib and pandas

```
pip install numpy matplotlib pandas
```

## Files

| File | Contents |
|---|---|
| `implementation.py` | The network: `initialize_weights`, `sigmoid`, `forwardprop`, `backprop`, `gradient_descent`, `cost` and `train` |
| `config.py` | The dataset (8 one-hot vectors), the list of learning rates used in Experiment 4, and default values (number of examples, epochs, a preset initialisation with seed 0, and a list of weight decay values) |
| `exp1.py` – `exp5.py` | One script per experiment in the report (see below) |
| `main.py` | Quick demo: trains the network once and prints the result |
| `figures/expN/` | Saved plots and result files of experiment N |

## How to run

Run every script from the project folder, so the imports and output paths work.

For a quick demo, run:

```
python main.py
```

It trains the network with the preset initialisation from `config.py` and prints the final cost, followed by the hidden code and the output for
each of the 8 inputs.

To reproduce an experiment from the report, run its script, for example:

```
python exp1.py
```

## Experiments

All experiments use seed 1, learning rate 1, no weight decay and batch gradient descent,
unless the experiment changes that setting. The settings are defined at the top of each script.

| Script | Report section | What it does | Output |
|---|---|---|---|
| `exp1.py` | Experiment 1 | Trains the network once and plots the cost per epoch | `figures/exp1/batch_gd_cost_curve.png` |
| `exp2.py` | Experiment 2 | Saves the weights, biases and activations at epochs 0, 1, 100, 500, 1500 and 10 000, and the mean parameter change between them | `figures/exp2/exp2_results.txt` |
| `exp3.py` | Experiment 3 | Compares batch, mini-batch (batch size 2 and 4) and stochastic gradient descent, per epoch and per weight update | `figures/exp3/methods_cost_vs_epochs.png`, `figures/exp3/methods_plateau_zoom.png`, `figures/exp3/methods_cost_vs_updates.png` |
| `exp4.py` | Experiment 4 | Trains with learning rates from 0.01 to 50 and records when each run reaches a cost of 0.01 | `figures/exp4/lr_cost_curves.png`, `figures/exp4/lr_convergence.png` |
| `exp5.py` | Experiment 5 | Compares random initialisation with all-zero and all-0.5 initialisation | `figures/exp5/init_cost_curves.png` |

## Using the network directly

```python
import config as cfg
from implementation import initialize_weights, train, forwardprop

W1, b1, W2, b2 = initialize_weights(seed=1)
W1, b1, W2, b2, cost_history, _ = train(cfg.data, cfg.Y, W1, b1, W2, b2,
                                        lr=1, epochs=10000, method="batch", batch_size=8)

a1, a2, a3 = forwardprop(cfg.data[0], W1, b1, W2, b2)
print("hidden code:", a2.flatten())
print("output:     ", a3.flatten())
```

`method` can be `"batch"`, `"mini-batch"` (with `batch_size`) or `"sgd"`. `train` returns the
trained parameters, the cost after every epoch and the learning rate used at every epoch.
