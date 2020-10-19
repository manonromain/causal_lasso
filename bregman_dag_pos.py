import cvxpy as cp
import numpy as np
import time
from tqdm.autonotebook import tqdm
from scipy.linalg import sqrtm
import mosek.fusion as msk


def dyn_no_lips_pos(X, W0, dagness_exp, dagness_pen, l1_pen, eps=1e-4, mosek=True, max_iter=500,
                    verbose=False, logging=False):
    """ Main algorithm described in our paper: Ours^+

    Args:
        X   (np.array): sample matrix
        W0  (np.array): initial adj matrix
        dagness_exp (float): alpha in paper
        dagness_pen (float): mu in paper
        l1_pen (float): lambda in paper
        eps    (float): sensivity for early stopping
        mosek   (bool): solver to use (false is cvxpy)
        max_iter (int): maximum number of iterations
        verbose (bool): prints objective value
        logging (bool): if enabled, returns all objective values + additional info (useful for analysis)

    Returns:
        Wk     (np.array): current iterate of weighted adj matrix
        logging_np (dict): if logging is True, dict of metrics/info
    """

    m, n = X.shape
    if m > n:
        s_mat = 1 / m * np.real(sqrtm(X.T @ X))
    else:
        s_mat = 1 / np.sqrt(m) * X
    prev_support = W0 > 0.5

    # Functions
    def dag_penalty(W):
        return dagness_pen * np.trace(np.linalg.matrix_power(np.eye(n) + dagness_exp * W, n))

    def grad_f_scalar_H(W, H):
        """Returns <∇f(W), D>"""
        powW = np.linalg.matrix_power(np.eye(n) + dagness_exp * W, n - 1)
        return n * dagness_pen * dagness_exp * np.trace(powW @ H)

    def distance_kernel(Wx, Wy):
        """the kernel used is mu(n - 1)(1+beta||W||_F)^n"""
        norm_y = np.linalg.norm(Wy, "fro")
        Wy_normalized = Wy / norm_y

        norm_x = np.linalg.norm(Wx, "fro")
        product = np.trace(Wy_normalized.T @ (Wx - Wy))

        hWx = (1 + dagness_exp * norm_x) ** n
        hWy = (1 + dagness_exp * norm_y) ** n
        grad_hy_scalar_x_minus_y = n * dagness_exp * (1 + dagness_exp * norm_y) ** (n - 1) * product
        return dagness_pen * (n - 1) * (hWx - hWy - grad_hy_scalar_x_minus_y)

    if logging:
        logging_dict = {"dagness_exp": dagness_exp, "dagness_pen": dagness_pen, "l1_pen": l1_pen,  # constants
                        "time": [], "ll_trace": [], "l1_val": [], "dag_constraint": [],
                        "nb_change_support": [], "support": []}

    # Constants
    lamb = 500

    # Init
    Wk = W0
    nnl_curr, nnl_prev = 0, 0
    it_nolips = 0
    start = time.time()
    pbar = tqdm(desc="NoLips", total=max_iter)
    while (it_nolips < 2 or (np.abs((nnl_prev - nnl_curr)/nnl_prev) >= eps)) and it_nolips < max_iter:
        # Implementing dynamic version of Dragomir et al. (2019)
        while True:
            # Solving Bregman iteration map
            if not mosek:
                next_W = solve_subproblem_cvxpy(s_mat, Wk,
                                                lamb, l1_pen, dagness_pen, dagness_exp)
            else:
                next_W = solve_subproblem_mosek(s_mat, Wk,
                                                lamb, l1_pen, dagness_pen, dagness_exp)

            # Sufficient decrease condition
            if np.abs(dag_penalty(next_W) - dag_penalty(Wk) - grad_f_scalar_H(Wk, next_W - Wk))  \
               > 1 / lamb * distance_kernel(next_W, Wk):
                lamb = lamb / 2
            else:
                break

        lamb = min(2 * lamb, 5000)

        Wk = next_W
        
        nnl_prev = nnl_curr
        nnl_curr = 1/m * np.linalg.norm(X @ (np.eye(n) - Wk), "fro")**2
        dag_penalty_k = dag_penalty(Wk)
        # Logging
        if logging:
            support = Wk > 0.5
            logging_dict["time"].append(time.time() - start)
            logging_dict["ll_trace"].append(nnl_curr)
            logging_dict["dag_constraint"].append(dag_penalty_k/dagness_pen)
            logging_dict["l1_val"].append(np.sum(Wk))
            logging_dict["nb_change_support"].append(np.sum(prev_support ^ support))
            logging_dict["support"].append(support.flatten())
            prev_support = support
        
        if verbose:
            print("Objective value at iteration {}".format(it_nolips))
            print(nnl_curr + dag_penalty_k + l1_pen * np.sum(Wk))
        it_nolips += 1
        pbar.update(1)
    print("Done in", time.time() - start, "s and", it_nolips, "iterations")
    if logging:
        logging_np = {k: np.array(v) for k, v in logging_dict.items()}
        return Wk, logging_np
    else:
        return Wk


def compute_C(n, Wk_value, dagness_pen, dagness_exp, lamb):
    # Compute C = grad f - 1/lamb grad h
    Wk_norm = np.linalg.norm(Wk_value, "fro")
    Wk_normalized = Wk_value / Wk_norm
    C = dagness_pen * n * dagness_exp * np.linalg.matrix_power(np.eye(n) + dagness_exp * Wk_value, n - 1)
    C -= 1/lamb * dagness_pen * (n - 1) * n * dagness_exp * (1 + dagness_exp * Wk_norm) ** (n - 1) * Wk_normalized.T
    return C


def solve_subproblem_mosek(s_mat, Wk_value,
                           lamb, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk>
                        + 1/lamb * Dh(W, Wk)

        this is only implemented for a specific penalty and kernel
    """

    n = s_mat.shape[1]

    C = compute_C(n, Wk_value, dagness_pen, dagness_exp, lamb)

    with msk.Model('model') as M:
        W = M.variable('W', [n, n], msk.Domain.greaterThan(0.))
        W.setLevel(Wk_value.flatten())
        t = M.variable('t')
        s1 = M.variable("s1")
        s = M.variable("s")

        # beta ||W|| <= s1 - 1
        z1 = msk.Expr.vstack([msk.Expr.sub(s1, 1.), msk.Expr.mul(dagness_exp, msk.Var.flatten(W))])
        M.constraint("qc1", z1, msk.Domain.inQCone())

        # s1 <= s^{1/n}
        M.constraint(msk.Expr.vstack(s, 1.0, s1), msk.Domain.inPPowerCone(1 / n))

        # t >= ||S(I-W)||^2
        z2 = msk.Expr.mul(s_mat, msk.Expr.sub(msk.Matrix.eye(n), W))
        M.constraint("rqc1", msk.Expr.vstack(t, .5, msk.Expr.flatten(z2)), msk.Domain.inRotatedQCone())

        # Set the objective function
        obj_spars = msk.Expr.sum(W)
        obj_tr = msk.Expr.dot(C.T, W)
        obj_vec = msk.Expr.vstack([t, obj_tr, s, obj_spars])
        obj = msk.Expr.dot([1., 1., dagness_pen * (n - 1) / lamb, l1_pen], obj_vec)

        M.objective(msk.ObjectiveSense.Minimize, obj)
        M.solve()
        M.selectedSolution(msk.SolutionType.Interior)

        next_W = M.getVariable('W').level().reshape(n, n)

        # Correcting mosek errors
        next_W = np.maximum(next_W, 0.0)
    return next_W


def solve_subproblem_cvxpy(s_mat, Wk_value,
                           lamb, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk>
                        + 1/lamb * Dh(W, Wk)

        this is only implemented for a specific penalty and kernel
    """

    n = s_mat.shape[1]

    # Compute C = grad g - 1/lamb grad h
    C = compute_C(n, Wk_value, dagness_pen, dagness_exp, lamb)

    W = cp.Variable([n, n], nonneg=True)
    W.value = Wk_value

    # Set the objective function
    obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro")**2
    obj_spars = l1_pen * msk.Expr.sum(W)
    obj_tr = cp.trace(C @ W)
    obj_kernel = dagness_pen * (n - 1) / lamb * (1 + dagness_exp * cp.norm(W, "fro")) ** n
    obj = obj_ll + obj_spars + obj_tr + obj_kernel

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    next_W = W.value

    # Correcting round-off errors
    next_W = np.maximum(next_W, 0.0)
    return next_W

