import numpy as np
from bregman_dag_general import dyn_no_lips_gen
from bregman_dag_pos import dyn_no_lips_pos

class BregmanSolver:
    def __init__(self, version="pos", dagness_pen=100, dagness_exp=1e-2, l1_pen=1e-6,
                 eps=1e-7, mosek=True, max_iter=200,
                 verbose=False, logging=False):
        """

            Args:
                version:
                dagness_pen:
                dagness_exp:
                l1_pen:
                eps:
                mosek:
                max_iter:
                verbose:
                logging:
        """
        self.version = version
        self.dagness_pen = dagness_pen
        self.dagness_exp = dagness_exp
        self.l1_pen = l1_pen
        self.eps = eps
        self.mosek = mosek
        self.max_iter = max_iter
        self.verbose = verbose
        self.logging = logging

    def run(self, X):
        n = X.shape[1]
        if self.version == "gen":
            W0_plus, W0_minus = 4 * np.random.random((n, n)), 4 * np.random.random((n, n))
            W_out, log_dict = dyn_no_lips_gen(X, W0_plus, W0_minus, self.dagness_exp, self.dagness_pen, self.l1_pen,
                                       eps=self.eps, mosek=self.mosek, max_iter=self.max_iter,
                                       verbose=self.verbose, logging=self.logging)
        else:
            W0 = 4 * np.random.random((n, n))
            W_out, log_dict = dyn_no_lips_pos(X, W0, self.dagness_exp, self.dagness_pen, self.l1_pen,
                                       eps=self.eps, mosek=self.mosek, max_iter=self.max_iter,
                                       verbose=self.verbose, logging=self.logging)
        self.sol = W_out
        self.log_dict = log_dict
        return W_out
