# Causal Lasso

This repository implements paper "A Bregman Method for Structure Learning on Sparse Directed Acyclic Graphs".

### Requirements

This package requires `numpy`, `scipy`, `tqdm`, `networkx` and `mosek`, installation can be done with `pip install requirements.txt`.


### Solver 
For now, the default solver used at each iteration is Mosek. We plan to provide an open source implementation in the near future. 

MOSEK's license is free for academic use, first obtain your license [here](https://www.mosek.com/products/academic-licenses/) using institutional email and place the obtained file `mosek.lic` in a file called:
```
%USERPROFILE%\mosek\mosek.lic           (Windows)
$HOME/mosek/mosek.lic                   (Linux, MacOS)
``` 


Further info [here](https://docs.mosek.com/9.2/install/installation.html#setting-up-the-license).


### Use
Minimal testing code is:
```
import numpy as np
import networkx as nx
from bregman_solver import BregmanSolver
X = np.random.random((1000, 30)) # Replace with your data
lasso = BregmanSolver()
W_est = lasso.fit(X)
nx.draw(nx.DiGraph(W_est))
```

A more detailed tutorial is available in `examples/tutorial.ipynb`
