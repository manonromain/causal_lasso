import numpy as np
from causal_lasso.general_weights_algo import dyn_no_lips_gen
from causal_lasso.positive_weights_algo import dyn_no_lips_pos


class CLSolver:
    """
        Solver class

        Arguments:
            version: "pos" or "gen"
            dagness_pen: mu in paper, coefficient for the dag penalty
            dagness_exp: alpha in paper, coefficient inside the dag penalty
            l1_pen: lambda in paper, l1 norm regularization parameter
            eps: tolerance for early stopping
            max_iter: maximum number of iterations
            mosek: if True, uses mosek instead of cvxpy
            verbose: if True, prints objective every 10 iterations
            logging: if true, solver will logged an important number of metrics at each iteration in self.log_dict
    """
    def __init__(self, version="gen", dagness_pen=100, dagness_exp=1e-2, l1_pen=1e-6,
                 eps=1e-7, max_iter=200, mosek=True,
                 verbose=False, logging=False):
        """Inits solver"""
        self.version = version
        self.dagness_pen = dagness_pen
        self.dagness_exp = dagness_exp
        self.l1_pen = l1_pen
        self.eps = eps
        self.mosek = mosek
        self.max_iter = max_iter
        self.verbose = verbose
        self.logging = logging

    def fit(self, X):
        """Returns output adjacency matrix from Causal Lasso"""
        n = X.shape[1]
        if self.version == "gen":
            W0_plus, W0_minus = np.random.random((n, n)), np.random.random((n, n))
            W_out, log_dict = dyn_no_lips_gen(X, W0_plus, W0_minus, self.dagness_exp, self.dagness_pen, self.l1_pen,
                                              eps=self.eps, mosek=self.mosek, max_iter=self.max_iter,
                                              verbose=self.verbose, logging=self.logging)
        else:
            W0 = np.random.random((n, n))
            W_out, log_dict = dyn_no_lips_pos(X, W0, self.dagness_exp, self.dagness_pen, self.l1_pen,
                                              eps=self.eps, mosek=self.mosek, max_iter=self.max_iter,
                                              verbose=self.verbose, logging=self.logging)
        # Final thresholding
        W_out = np.where(np.abs(W_out) > 0.5, W_out, 0)
        self.sol = W_out
        self.log_dict = log_dict
        return W_out
