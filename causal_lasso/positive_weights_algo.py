import numpy as np
import time
from tqdm.autonotebook import tqdm
from scipy.linalg import sqrtm, expm
import logging
import cvxpy as cp
try:
    import mosek.fusion as msk
except ImportError:
    pass

try:
    import torch
    from cvxpylayers.torch import CvxpyLayer
except ImportError:
    logging.info("If you want to use PyTorch CVXPY layers, you should install it first")
    logging.info("using pip install torch cvxpylayers --user")
    pass


def dyn_no_lips_pos(X, W0, dagness_exp, dagness_pen, l1_pen, eps=1e-4, solver="mosek", max_iter=500,
                    logging_dict=False, device=None):
    """ Main algorithm described in our paper: $Ours^+$

    Args:
        X   (np.array): sample matrix
        W0  (np.array): initial adj matrix
        dagness_exp (float): alpha in paper
        dagness_pen (float): mu in paper
        l1_pen (float): lambda in paper
        eps    (float): sensivity for early stopping
        solver   (string): solver to use (choices are cvxpy, cvxpylayers or mosek)
        max_iter (int): maximum number of iterations
        logging_dict (bool): if enabled, returns all objective values + additional info (useful for analysis)

    Returns:
        Wk     (np.array): current iterate of weighted adj matrix
        logging_np (dict): if logging_dict is True, dict of metrics/info
    """

    m, n = X.shape
    if m > n:
        # s_mat = 1 / np.sqrt(m) * np.real(sqrtm(X.T  @ X))
        s_mat = np.real(sqrtm(np.cov(X.T, bias=True)))
    else:
        s_mat = 1 / np.sqrt(m) * X

    # Functions
    def dag_penalty(W):
        if solver == "cvxpylayers":
            return dagness_pen * torch.trace(torch.matrix_exp(dagness_exp * W))
        else:
            return dagness_pen * np.trace(expm(dagness_exp * W))

    def grad_f_scalar_H(W, H):
        """Returns <∇f(W), D>"""
        if solver == "cvxpylayers":
            expW = torch.matrix_exp(dagness_exp * W)
            return dagness_pen * dagness_exp * torch.trace(torch.matmul(expW, H))
        else:
            expW = expm(dagness_exp * W)
            return dagness_pen * dagness_exp * np.trace(expW @ H)

    def distance_kernel(Wx, Wy):
        """the kernel used is mu/2 exp(||W||^2)"""
        if solver == "cvxpylayers":
            norm_y_sq = torch.norm(Wy, "fro")**2
            norm_x_sq = torch.norm(Wx, "fro")**2
            hWx = torch.exp(dagness_exp**2 * norm_x_sq)
            hWy = torch.exp(dagness_exp**2 * norm_y_sq)
            product = dagness_exp * torch.trace(torch.matmul(Wy.T, Wx - Wy))
        else:
            norm_y_sq = np.linalg.norm(Wy, "fro")**2
            norm_x_sq = np.linalg.norm(Wx, "fro")**2
            hWx = np.exp(dagness_exp**2 * norm_x_sq)
            hWy = np.exp(dagness_exp**2 * norm_y_sq)

            product = dagness_exp * np.trace(Wy.T @ (Wx - Wy))

        grad_hy_scalar_x_minus_y = 2 * hWy * product
        return .5 * dagness_pen * (hWx - hWy - grad_hy_scalar_x_minus_y)

    if logging_dict:
        log_dict = {"dagness_exp": dagness_exp, "dagness_pen": dagness_pen, "l1_pen": l1_pen,  # constants
                        "time": [], "l2_error": [], "l1_val": [], "dag_constraint": [],
                        "nb_change_support": [], "support": [], "gammas": []}

    # Constants
    gamma = 500

    # Init
    if solver == "cvxpylayers":
        Wk = torch.tensor(W0, device=device)
    else:
        Wk = W0

    previous_dag_penalty = dag_penalty(Wk)
    prev_support = Wk > 0.5
    l2_error_curr, l2_error_prev = 0, 0

    it_nolips = 0
    start = time.time()
    pbar = tqdm(desc="Causal Lasso for positive weights", total=max_iter)

    if solver == "cvxpylayers":
        layer = create_cvxpylayer(s_mat, dagness_pen, dagness_exp)



    while (it_nolips < 2 or (np.abs((l2_error_prev - l2_error_curr)/l2_error_prev) >= eps)) and it_nolips < max_iter:
        # Implementing dynamic version of Dragomir et al. (2019)
        it = 0
        assert np.sum(Wk**2) >= 1/dagness_exp
        while it < 1000:
            # Solving Bregman iteration map
            try:
                if solver == "cvxpylayers":
                    next_W = solve_subproblem_cvxpylayer(layer, Wk, l1_pen, dagness_pen, dagness_exp, gamma)
                elif solver == "mosek":
                    next_W = solve_subproblem_mosek(s_mat, Wk,
                                                    gamma, l1_pen, dagness_pen, dagness_exp)
                else:
                    next_W = solve_subproblem_cvxpy(s_mat, Wk,
                                                    gamma, l1_pen, dagness_pen, dagness_exp)

            except msk.SolutionError as e:
                logging.warning(e)
                gamma = gamma / 2
                logging.warning("Trying gamma smaller", it, gamma, current_dag_penalty - previous_dag_penalty - grad_f_scalar_H(Wk, next_W - Wk)
                                - 1 / gamma * distance_kernel(next_W, Wk))
                it += 1
                continue

            # Sufficient decrease condition
            current_dag_penalty = dag_penalty(next_W)
            if current_dag_penalty - previous_dag_penalty - grad_f_scalar_H(Wk, next_W - Wk)  \
               > 1 / gamma * distance_kernel(next_W, Wk):
                gamma = gamma / 2
            else:
                break
        gamma_k = gamma
        gamma = min(2 * gamma, 1e10)

        Wk = next_W
        previous_dag_penalty = current_dag_penalty
        
        l2_error_prev = l2_error_curr
        l2_error_curr = 1/m * np.linalg.norm(X @ (np.eye(n) - Wk), "fro")**2
        # Logging
        if logging_dict:
            support = Wk > 0.5
            log_dict["time"].append(time.time() - start)
            log_dict["l2_error"].append(l2_error_curr)
            log_dict["dag_constraint"].append(current_dag_penalty/dagness_pen)
            log_dict["l1_val"].append(np.sum(Wk))
            log_dict["nb_change_support"].append(np.sum(prev_support ^ support))
            log_dict["support"].append(support.flatten())
            log_dict["gammas"].append(gamma_k)
            prev_support = support
        

        logging.info("Objective value at iteration {}".format(it_nolips))
        logging.info(l2_error_curr + current_dag_penalty + l1_pen * np.sum(Wk))

        # if it_nolips > 50 and it_nolips % 10 == 0:
        #     dagness_pen *= 10

        it_nolips += 1
        pbar.update(1)
    logging.info("Done in", time.time() - start, "s and", it_nolips, "iterations")
    if logging_dict:
        logging_np = {k: np.array(v) for k, v in log_dict.items()}
        return Wk, logging_np
    else:
        return Wk, {}


def compute_C(n, Wk_value, dagness_pen, dagness_exp, gamma):
    # Compute C = grad f - 1/gamma grad h
    Wk_normsq = np.linalg.norm(Wk_value, "fro")**2
    C = expm(dagness_exp * Wk_value) - 1/gamma * np.exp(dagness_exp**2 * Wk_normsq) * Wk_value
    return dagness_pen * dagness_exp * C.T


def solve_subproblem_mosek(s_mat, Wk_value,
                           gamma, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with MOSEK
        this is only implemented for a specific penalty and kernel

        Args:
            s_mat (np.array): data matrix
            Wk_value (np.array): current iterate value
            gamma (float): Bregman iteration map param
            l1_pen (float): lambda in paper
            dagness_pen (float): mu in paper
            dagness_exp (float): alpha in paper
    """

    n = s_mat.shape[1]

    C = compute_C(n, Wk_value, dagness_pen, dagness_exp, gamma)

    with msk.Model('model') as M:
        W = M.variable('W', [n, n], msk.Domain.greaterThan(0.))
        W.setLevel(Wk_value.flatten())
        t = M.variable('t')
        y = M.variable("y")
        s = M.variable("s")

        # y >= ||dagness_exp * W||^2
        M.constraint("qc1", msk.Expr.vstack(y, 0.5, msk.Expr.mul(dagness_exp, msk.Var.flatten(W))), msk.Domain.inRotatedQCone())

        # s >= e^y
        M.constraint(msk.Expr.vstack(s, 1.0, y), msk.Domain.inPExpCone())

        # t >= ||S(I-W)||^2
        z2 = msk.Expr.mul(s_mat, msk.Expr.sub(msk.Matrix.eye(n), W))
        M.constraint("rqc1", msk.Expr.vstack(t, .5, msk.Expr.flatten(z2)), msk.Domain.inRotatedQCone())

        # Constrain diag to be zero
        M.constraint(W.diag(), msk.Domain.equalsTo(0.0))

        # Set the objective function
        obj_spars = msk.Expr.sum(W)
        obj_tr = msk.Expr.dot(C.T, W)
        obj_vec = msk.Expr.vstack([t, obj_tr, s, obj_spars])
        obj = msk.Expr.dot([1., 1., dagness_pen * .5 / gamma, l1_pen], obj_vec)

        M.objective(msk.ObjectiveSense.Minimize, obj)
        M.solve()
        M.selectedSolution(msk.SolutionType.Interior)

        next_W = M.getVariable('W').level().reshape(n, n)

        # Correcting mosek errors
        # next_W = np.maximum(next_W, 0.0)
    return next_W


def solve_subproblem_cvxpy(s_mat, Wk_value,
                           gamma, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with CVXPY
        this is only implemented for a specific penalty and kernel

        Args:
            s_mat (np.array): data matrix
            Wk_value (np.array): current iterate value
            gamma (float): Bregman iteration map param
            l1_pen (float): lambda in paper
            dagness_pen (float): mu in paper
            dagness_exp (float): alpha in paper
        """

    n = s_mat.shape[1]

    # Compute C = grad g - 1/gamma grad h
    C_f, C_h = compute_C(n, Wk_value, dagness_pen, dagness_exp, gamma)

    W = cp.Variable([n, n], nonneg=True)
    W.value = Wk_value

    # Set the objective function
    obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro")**2
    obj_spars = l1_pen * cp.sum(W)
    obj_tr_f = cp.trace(C_f @ W)
    obj_tr_h = cp.trace(C_h @ W)
    obj_kernel = dagness_pen / gamma * cp.exp(dagness_exp * cp.norm(W, "fro")**2)
    obj = obj_ll + obj_spars + obj_tr_f + obj_tr_h + obj_kernel
    print("value_before_obj", obj.value)

    prob = cp.Problem(cp.Minimize(obj), [cp.diag(W) == np.zeros(n)])
    prob.solve()
    print("value_after_obj", obj.value)
    if prob.status != "optimal":
        logging.warning(prob.status)
    next_W = W.value

    # Correcting round-off errors
    next_W = np.maximum(next_W, 0.0)
    return next_W


def create_cvxpylayer(s_mat, dagness_pen, dagness_exp):
    return NotImplementedError
# def create_cvxpylayer(s_mat, dagness_pen, dagness_exp):
#     """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
#         with CVXPY
#         this is only implemented for a specific penalty and kernel
#
#         Args:
#             s_mat (np.array): data matrix
#             Wk_value (np.array): current iterate value
#             gamma (float): Bregman iteration map param
#             l1_pen (float): lambda in paper
#             dagness_pen (float): mu in paper
#             dagness_exp (float): alpha in paper
#         """
#
#     n = s_mat.shape[1]
#
#     # Compute C = grad g - 1/gamma grad h
#     C_param = cp.Parameter((n,n))
#     inv_gamma_param = cp.Parameter(nonneg=True)
#     l1_pen_param = cp.Parameter(nonneg=True)
#
#     W = cp.Variable([n, n], nonneg=True)
#
#     # Set the objective function
#     obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro")**2
#     obj_spars = l1_pen_param * cp.sum(W)
#     obj_tr = cp.trace(C_param @ W)
#     obj_kernel = dagness_pen * (n - 1) * inv_gamma_param * (1 + dagness_exp * cp.norm(W, "fro")) ** n
#     obj = obj_ll + obj_spars + obj_tr + obj_kernel
#
#     prob = cp.Problem(cp.Minimize(obj), [cp.diag(W)==np.zeros(n)])
#     assert prob.is_dpp(), "".format()
#
#     # set_trace()
#
#     layer = CvxpyLayer(prob, parameters=[C_param, inv_gamma_param, l1_pen_param],
#                        variables=[W])
#     return layer


def solve_subproblem_cvxpylayer(layer, Wk_value, l1_pen, dagness_pen, dagness_exp, gamma):
    raise NotImplementedError
# def solve_subproblem_cvxpylayer(layer, Wk_value, l1_pen, dagness_pen, dagness_exp, gamma):
#     n = Wk_value.shape[0]
#     C = compute_C(n, Wk_value, dagness_pen, dagness_exp, gamma)
#     torch_C = torch.tensor(C, dtype=torch.float64)
#     torch_inv_gamma = torch.tensor(1/gamma, dtype=torch.float64)
#     torch_l1_pen = torch.tensor(l1_pen, dtype=torch.float64)
#
#     next_W, = layer(torch_C, torch_inv_gamma, torch_l1_pen)
#     return next_W.numpy()


def init_no_lips(s_mat, l1_pen):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with CVXPY
        this is only implemented for a specific penalty and kernel

        Args:
            s_mat (np.array): data matrix
            l1_pen (float): lambda in paper
        """

    n = s_mat.shape[1]

    W = cp.Variable([n, n], nonneg=True)

    # Set the objective function
    obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro") ** 2
    obj_spars = l1_pen * cp.sum(W)
    obj = obj_ll + obj_spars

    prob = cp.Problem(cp.Minimize(obj), [cp.diag(W) == np.zeros(n)])
    prob.solve()
    if prob.status != "optimal":
        logging.warning(prob.status)
    next_W = W.value

    # Correcting round-off errors
    next_W = np.maximum(next_W, 0.0)
    return next_W




