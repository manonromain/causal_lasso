# Causal Lasso

This repository implements paper "A Bregman Method for Structure Learning on Sparse Directed Acyclic Graphs" by Manon Romain and Alexandre d'Aspremont.

### Requirements

This package requires `numpy`, `scipy`, `tqdm`, `networkx` and `mosek`, installation can be done with `pip install -r docs/requirements.txt`.


### Solver 
For now, the default solver used at each iteration is Mosek. We plan to provide an open source implementation in the near future. 

MOSEK's license is free for academic use, first obtain your license [here](https://www.mosek.com/products/academic-licenses/) using institutional email and place the obtained file `mosek.lic` in a file called:
```
%USERPROFILE%\mosek\mosek.lic           (Windows)
$HOME/mosek/mosek.lic                   (Linux, MacOS)
``` 


Further information available [here](https://docs.mosek.com/9.2/install/installation.html#setting-up-the-license).


### Use
Minimal testing code is:
```
import numpy as np
import networkx as nx
from causal_lasso.solver import CLSolver
X = np.random.random((1000, 30)) # Replace with your data
lasso = CLSolver()
W_est = lasso.fit(X)
nx.draw(nx.DiGraph(W_est))
```

A more detailed tutorial is available in `examples/tutorial.ipynb`.
