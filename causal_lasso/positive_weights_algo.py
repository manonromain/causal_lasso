import numpy as np
import time
from tqdm.autonotebook import tqdm
from scipy.linalg import sqrtm
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


def dyn_no_lips_pos(X, W0, degree_poly, dagness_exp, dagness_pen, l1_pen, eps=1e-4, solver="mosek", max_iter=500,
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
            return dagness_pen * torch.trace(torch.matrix_power(torch.eye(n) + dagness_exp * W, degree_poly))
        else:
            return dagness_pen * np.trace(np.linalg.matrix_power(np.eye(n) + dagness_exp * W, degree_poly))

    def grad_f_scalar_H(W, H):
        """Returns <∇f(W), D>"""
        if solver == "cvxpylayers":
            powW = torch.matrix_power(torch.eye(n) + dagness_exp * W, degree_poly - 1)
            return n * dagness_pen * dagness_exp * torch.trace(torch.matmul(powW, H))
        else:
            powW = np.linalg.matrix_power(np.eye(n) + dagness_exp * W, degree_poly - 1)
            return n * dagness_pen * dagness_exp * np.trace(powW @ H)

    def distance_kernel(Wx, Wy):
        """the kernel used is mu(n - 1)(1+beta||W||_F)^n"""
        if solver == "cvxpylayers":
            norm_y = torch.norm(Wy, "fro")
            norm_x = torch.norm(Wx, "fro")

            Wy_normalized = Wy / norm_y
            product = torch.trace(torch.matmul(Wy_normalized.T, Wx - Wy))

        else:
            norm_y = np.linalg.norm(Wy, "fro")
            norm_x = np.linalg.norm(Wx, "fro")

            Wy_normalized = Wy / norm_y
            product = np.trace(Wy_normalized.T @ (Wx - Wy))
        hWx = (1 + dagness_exp * norm_x) ** degree_poly
        hWy = (1 + dagness_exp * norm_y) ** degree_poly
        grad_hy_scalar_x_minus_y = degree_poly * dagness_exp * (1 + dagness_exp * norm_y) ** (degree_poly - 1) * product
        return dagness_pen * (degree_poly - 1) * (hWx - hWy - grad_hy_scalar_x_minus_y)

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

    # DEBUG
    def phi(W):
        g = np.linalg.norm(s_mat @ (np.eye(n) - W), "fro")**2 + l1_pen * np.sum(W)
        f = dag_penalty(W)
        return f, g


    #####
    while (it_nolips < 2 or (np.abs((l2_error_prev - l2_error_curr)/l2_error_prev) >= eps)) and it_nolips < max_iter:
        # Implementing dynamic version of Dragomir et al. (2019)
        it = 0
        while it < 1000:
            # Solving Bregman iteration map
            try:
                if solver == "cvxpylayers":
                    next_W = solve_subproblem_cvxpylayer(layer, Wk, degree_poly,
                                                         l1_pen, dagness_pen, dagness_exp, gamma)
                elif solver == "mosek":
                    next_W = solve_subproblem_mosek(s_mat, Wk, degree_poly,
                                                    gamma, l1_pen, dagness_pen, dagness_exp)
                else:
                    next_W = solve_subproblem_cvxpy(s_mat, Wk, degree_poly,
                                                    gamma, l1_pen, dagness_pen, dagness_exp)

            except msk.SolutionError as e:
                logging.warning(e)
                gamma = gamma / 2
                logging.warning("Trying gamma smaller", it, gamma)
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
        gamma = min(2 * gamma, 10000)

        Wk = next_W
        previous_dag_penalty = current_dag_penalty
        
        l2_error_prev = l2_error_curr
        l2_error_curr = 1/m * np.linalg.norm(X @ (np.eye(n) - Wk), "fro")**2
        fk, gk = phi(Wk)
        # Logging
        if logging_dict:
            support = Wk > 0.5
            log_dict["time"].append(time.time() - start)
            log_dict["l2_error"].append(l2_error_curr)
            log_dict["dag_constraint"].append(current_dag_penalty/dagness_pen)
            print("Current phi:", fk + gk, fk, gk)
            log_dict["l1_val"].append(np.sum(Wk))
            log_dict["nb_change_support"].append(np.sum(prev_support ^ support))
            log_dict["support"].append(support.flatten())
            log_dict["gammas"].append(gamma_k)
            prev_support = support
        

        logging.info("Objective value at iteration {}".format(it_nolips))
        logging.info(fk + gk)
        it_nolips += 1
        pbar.update(1)
    logging.info("Done in", time.time() - start, "s and", it_nolips, "iterations")
    if logging_dict:
        logging_np = {k: np.array(v) for k, v in log_dict.items()}
        return Wk, logging_np
    else:
        return Wk, {}


def compute_C(n, Wk_value, degree_poly, dagness_pen, dagness_exp, gamma):
    # Compute C = grad f - 1/gamma grad h
    Wk_norm = np.linalg.norm(Wk_value, "fro")
    Wk_normalized = Wk_value / Wk_norm
    C = dagness_pen * degree_poly * dagness_exp * np.linalg.matrix_power(np.eye(n) + dagness_exp * Wk_value, degree_poly - 1)
    C -= 1/gamma * dagness_pen * (degree_poly - 1) * degree_poly * dagness_exp * (1 + dagness_exp * Wk_norm) ** (degree_poly - 1) * Wk_normalized.T
    return C


def solve_subproblem_mosek(s_mat, Wk_value, degree_poly,
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

    C = compute_C(n, Wk_value, degree_poly, dagness_pen, dagness_exp, gamma)

    with msk.Model('model') as M:
        W = M.variable('W', [n, n], msk.Domain.greaterThan(0.))
        W.setLevel(Wk_value.flatten())
        t = M.variable('t')
        s1 = M.variable("s1")
        s = M.variable("s")

        # beta ||W|| <= s1 - 1
        z1 = msk.Expr.vstack([msk.Expr.sub(s1, 1.), msk.Expr.mul(dagness_exp, msk.Var.flatten(W))])
        M.constraint("qc1", z1, msk.Domain.inQCone())

        # s1 <= s^{1/degree_poly}
        M.constraint(msk.Expr.vstack(s, 1.0, s1), msk.Domain.inPPowerCone(1 / degree_poly))

        # t >= ||S(I-W)||^2
        z2 = msk.Expr.mul(s_mat, msk.Expr.sub(msk.Matrix.eye(n), W))
        M.constraint("rqc1", msk.Expr.vstack(t, .5, msk.Expr.flatten(z2)), msk.Domain.inRotatedQCone())

        # Constrain diag to be zero
        M.constraint(W.diag(), msk.Domain.equalsTo(0.0))

        # Set the objective function
        obj_spars = msk.Expr.sum(W)
        obj_tr = msk.Expr.dot(C.T, W)
        obj_vec = msk.Expr.vstack([t, obj_tr, s, obj_spars])
        obj = msk.Expr.dot([1., 1., dagness_pen * (degree_poly - 1) / gamma, l1_pen], obj_vec)

        M.objective(msk.ObjectiveSense.Minimize, obj)
        M.solve()
        M.selectedSolution(msk.SolutionType.Interior)

        next_W = M.getVariable('W').level().reshape(n, n)

        # Correcting mosek errors
        # next_W = np.maximum(next_W, 0.0)
    return next_W


def solve_subproblem_cvxpy(s_mat, Wk_value, degree_poly,
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
    C = compute_C(n, Wk_value, degree_poly, dagness_pen, dagness_exp, gamma)

    W = cp.Variable([n, n], nonneg=True)
    W.value = Wk_value

    # Set the objective function
    obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro")**2
    obj_spars = l1_pen * cp.sum(W)
    obj_tr = cp.trace(C @ W)
    obj_kernel = dagness_pen * (degree_poly - 1) / gamma * (1 + dagness_exp * cp.norm(W, "fro")) ** degree_poly
    obj = obj_ll + obj_spars + obj_tr + obj_kernel

    prob = cp.Problem(cp.Minimize(obj), [cp.diag(W) == np.zeros(n)])
    prob.solve()
    if prob.status != "optimal":
        logging.warning(prob.status)
    next_W = W.value

    # Correcting round-off errors
    next_W = np.maximum(next_W, 0.0)
    return next_W


def create_cvxpylayer(s_mat, dagness_pen, dagness_exp):
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
    C_param = cp.Parameter((n,n))
    inv_gamma_param = cp.Parameter(nonneg=True)
    l1_pen_param = cp.Parameter(nonneg=True)

    W = cp.Variable([n, n], nonneg=True)

    # Set the objective function
    obj_ll = cp.norm(s_mat @ (np.eye(n) - W), "fro")**2
    obj_spars = l1_pen_param * cp.sum(W)
    obj_tr = cp.trace(C_param @ W)
    obj_kernel = dagness_pen * (n - 1) * inv_gamma_param * (1 + dagness_exp * cp.norm(W, "fro")) ** n
    obj = obj_ll + obj_spars + obj_tr + obj_kernel

    prob = cp.Problem(cp.Minimize(obj), [cp.diag(W) == np.zeros(n)])
    assert prob.is_dpp(), "".format()

    # set_trace()

    layer = CvxpyLayer(prob, parameters=[C_param, inv_gamma_param, l1_pen_param],
                       variables=[W])
    return layer



def solve_subproblem_cvxpylayer(layer, Wk_value, degree_poly, l1_pen, dagness_pen, dagness_exp, gamma):
    n = Wk_value.shape[0]
    C = compute_C(n, Wk_value, degree_poly, dagness_pen, dagness_exp, gamma)
    torch_C = torch.tensor(C, dtype=torch.float64)
    torch_inv_gamma = torch.tensor(1/gamma, dtype=torch.float64)
    torch_l1_pen = torch.tensor(l1_pen, dtype=torch.float64)

    next_W, = layer(torch_C, torch_inv_gamma, torch_l1_pen)
    return next_W.numpy()


def init_nolips(s_mat, l1_pen):
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




