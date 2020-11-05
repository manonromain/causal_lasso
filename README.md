# Causal Lasso

This repository implements paper "A Bregman Method for Structure Learning on Sparse Directed Acyclic Graphs"

This package requires `numpy`, `scipy`, `tqdm` and `networkx`, installation can be done with `pip install requirements.txt`.

Minimal testing code is:
```
import numpy as np
from bregman_solver import BregmanSolver
X = np.random.random((1000, 30)) # Replace with your data
lasso = BregmanSolver()
W_est = lasso.fit(X)
nx.draw(nx.DiGraph(W_est))
```