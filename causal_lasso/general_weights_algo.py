
import numpy as np
import time
from tqdm.autonotebook import tqdm
from scipy.linalg import sqrtm
import cvxpy as cp
try:
    import mosek.fusion as msk
except ImportError:
    pass



def dyn_no_lips_gen(X, W0_plus, W0_minus, dagness_exp, dagness_pen, l1_pen, eps=1e-4, mosek=False, max_iter=200,
                    verbose=False, logging=False):
    """ Main algorithm described in our paper

    Args:
        X   (np.array): sample matrix
        W0_plus  (np.array): initial positive part adj matrix
        W0_minus  (np.array): initial negative part adj matrix
        dagness_exp  (float): alpha in paper
        dagness_pen  (float): mu in paper
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
    prev_support = np.abs(W0_plus - W0_minus) > 0.5
    if m > n:
        s_mat = 1 / m * np.real(sqrtm(X.T @ X))
    else:
        s_mat = 1 / np.sqrt(m) * X

    # Functions
    def dag_penalty(W_plus, W_minus):
        sum_W = W_plus + W_minus
        return dagness_pen * np.trace(np.linalg.matrix_power(np.eye(n) + dagness_exp * sum_W, n))  # pow version

    def grad_f_scalar_H(W_plus, W_minus, H_plus, H_minus):
        """Returns <∇f(W), D>"""
        sum_W = W_plus + W_minus
        sum_H = H_plus + H_minus
        powW = np.linalg.matrix_power(np.eye(n) + dagness_exp * sum_W, n - 1)
        return n * dagness_pen * dagness_exp * np.trace(powW @ sum_H)

    def distance_kernel(Wx_plus, Wx_minus, Wy_plus, Wy_minus):
        """
        the kernel used is C (1+beta||W||_F)^n where C = mu * (n - 1)
        """
        sum_x = Wx_plus + Wx_minus
        sum_y = Wy_plus + Wy_minus
        norm_y = np.linalg.norm(sum_y, "fro")
        sum_Wy_normalized = sum_y / norm_y
        norm_x = np.linalg.norm(sum_x, "fro")
        product = np.trace(sum_Wy_normalized.T @ (sum_x - sum_y))

        hWx = (1 + dagness_exp * norm_x) ** n
        hWy = (1 + dagness_exp * norm_y) ** n
        grad_hy_scalar_x_minus_y = n * dagness_exp * (1 + dagness_exp * norm_y) ** (n - 1) * product
        return dagness_pen * (n - 1) * (hWx - hWy - grad_hy_scalar_x_minus_y)

    if logging:
        log_dict = {"dagness_exp": dagness_exp, "dagness_pen": dagness_pen, "l1_pen": l1_pen, # constants
                   "time": [], "l2_error": [], "l1_val": [], "dag_constraint": [],
                   "nb_change_support":[], "support": [], "support_pos": [], "support_neg": [],
                   "gammas":[]
                    }

    # Constants
    gamma = 500

    # Init
    Wk_plus, Wk_minus = W0_plus, W0_minus
    l2_error_curr, l2_error_prev = 0, 0
    start = time.time()
    it_nolips = 0
    pbar = tqdm(desc="Causal Lasso", total=max_iter)
    while (it_nolips < 2 or (np.abs((l2_error_prev - l2_error_curr)/l2_error_prev) >= eps)) and (it_nolips < max_iter):
        it = 0
        while True:
            if gamma < 1:
                print("violated L-smad")

            try:  # TODO more solvers
                if mosek:
                    next_W_plus, next_W_minus = bregman_map_mosek(s_mat, Wk_plus, Wk_minus,
                                                                  gamma, l1_pen, dagness_pen, dagness_exp)
                else:
                    next_W_plus, next_W_minus = bregman_map_cvx(s_mat, Wk_plus, Wk_minus,
                                                                gamma, l1_pen, dagness_pen, dagness_exp)
            except msk.SolutionError:
                gamma = gamma / 2
                it += 1
                continue

            # Sufficient decrease condition
            if dag_penalty(next_W_plus, next_W_minus) - dag_penalty(Wk_plus, Wk_minus) \
               - grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus, next_W_minus - Wk_minus)  \
               > 1 / gamma * distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus):
                gamma = gamma / 2
                it += 1
            else:
                break
        gamma_k = gamma
        # Trying increasing step size
        gamma = min(2 * gamma, 10000)

        # TODO delete?
        if np.sum(next_W_minus + next_W_plus) < n/((n-2)*dagness_exp):
             print("assertion false because of thresholding: iteration map may not be stable")

        # Compute current iterate
        Wk = next_W_plus - next_W_minus
        Wk_plus, Wk_minus = next_W_plus, next_W_minus
        l2_error_prev = l2_error_curr
        l2_error_curr = np.linalg.norm(s_mat @ (np.eye(n) - next_W_plus + next_W_minus), "fro")**2
        dag_penalty_k = dag_penalty(Wk_plus, Wk_minus)

        if logging:
            support = np.abs(Wk) > 0.5
            # Logging
            log_dict["time"].append(time.time() - start)
            log_dict["l2_error"].append(l2_error_curr)
            log_dict["dag_constraint"].append(dag_penalty_k/dagness_pen)
            log_dict["l1_val"].append(np.sum(next_W_plus) + np.sum(next_W_minus))
            log_dict["nb_change_support"].append(np.sum(support ^ prev_support))
            log_dict["support"].append(support.flatten())
            log_dict["support_pos"].append((Wk > 0.5).flatten())
            log_dict["support_neg"].append((Wk < -0.5).flatten())
            log_dict["gammas"].append(gamma_k)
            prev_support = support

        if it_nolips%10 == 0 and verbose:
            print("Objective value at iteration {}".format(it_nolips))
            print(l2_error_curr + dag_penalty_k + l1_pen * np.sum(Wk_plus + Wk_minus))

        it_nolips += 1
        pbar.update(1)
    print("Done in", time.time() - start, "s and", it_nolips, "iterations")
    if logging:
        logging_np = {k: np.array(v) for k, v in log_dict.items()}
        return Wk, logging_np
    else:
        return Wk, {}


def compute_C(n, sum_Wk, dagness_pen, dagness_exp, gamma):
    sum_Wk_norm = np.linalg.norm(sum_Wk, "fro")
    sum_Wk_normalized = sum_Wk / sum_Wk_norm
    # C = grad f - 1/gamma grad h
    C = dagness_pen * n * dagness_exp * np.linalg.matrix_power(np.eye(n) + dagness_exp * sum_Wk, n - 1)
    C -= 1 / gamma * dagness_pen * (n - 1) * n * dagness_exp * \
        (1 + dagness_exp * sum_Wk_norm) ** (n - 1) * sum_Wk_normalized.T
    return C

def bregman_map_cvx(s_mat, Wk_plus_value, Wk_minus_value,
                    gamma, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with CVX
        this is only implemented for a specific penalty and kernel

        Args:
            s_mat (np.array): data matrix
            Wk_plus_value (np.array): current iterate value for W+
            Wk_minus_value (np.array): current iterate value for W-
            gamma (float): Bregman iteration map param
            l1_pen (float): lambda in paper
            dagness_pen (float): mu in paper
            dagness_exp (float): alpha in paper
    """


    n = s_mat.shape[1]

    W_plus = cp.Variable((n, n), nonneg=True)
    W_plus.value = Wk_plus_value
    W_minus = cp.Variable((n, n), nonneg=True)
    W_minus.value = Wk_minus_value
    sum_W = W_plus + W_minus  # sum variable

    obj_ll = cp.norm(s_mat @ (np.eye(n) - W_plus + W_minus), "fro") ** 2
    obj_spars = l1_pen * cp.sum(W_plus + W_minus)

    # Compute C
    sum_Wk = Wk_plus_value + Wk_minus_value
    C = compute_C(n, sum_Wk, dagness_pen, dagness_exp, gamma)

    obj_trace = cp.trace(C @ sum_W)
    obj_kernel = 1 / gamma * dagness_pen * (n - 1) * (1 + dagness_exp * cp.norm(sum_W, "fro"))**n

    obj = obj_ll + obj_spars + obj_trace + obj_kernel
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(W_plus) + cp.sum(W_minus) >= n/((n-2)*dagness_exp)])
    prob.solve()

    if prob.status != "optimal":
        prob.solve(verbose=True)

    next_W_plus, next_W_minus = W_plus.value, W_minus.value

    # FIXME
    tilde_W_plus = np.maximum(next_W_plus - next_W_minus, 0.0)
    tilde_W_minus = np.maximum(next_W_minus - next_W_plus, 0.0)
    tilde_sum = tilde_W_plus + tilde_W_minus
    #
    if np.sum(tilde_sum) >= n / ((n - 2) * dagness_exp):
        return tilde_W_plus, tilde_W_minus
    else:
        return next_W_plus, next_W_minus


def bregman_map_mosek(s_mat, Wk_plus_value, Wk_minus_value,
                      gamma, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with MOSEK
        this is only implemented for a specific penalty and kernel

        Args:
            s_mat (np.array): data matrix
            Wk_plus_value (np.array): current iterate value for W+
            Wk_minus_value (np.array): current iterate value for W-
            gamma (float): Bregman iteration map param
            l1_pen (float): lambda in paper
            dagness_pen (float): mu in paper
            dagness_exp (float): alpha in paper
    """

    n = s_mat.shape[1]
    # Compute C
    sum_Wk = Wk_plus_value + Wk_minus_value
    C = compute_C(n, sum_Wk, dagness_pen, dagness_exp, gamma)
    #

    with msk.Model('model2') as M:
        W_plus = M.variable('W_plus', [n, n], msk.Domain.greaterThan(0.))
        W_minus = M.variable('W_minus', [n, n], msk.Domain.greaterThan(0.))
        W_plus.setLevel(Wk_plus_value.flatten())
        W_minus.setLevel(Wk_minus_value.flatten())
        sum_W = msk.Expr.add(W_plus, W_minus)
        diff_W = msk.Expr.sub(W_plus, W_minus)
        t = M.variable('T')
        s1 = M.variable("s1")
        s = M.variable("s")

        # beta ||W+ + W-|| <= s1 - 1
        sum_W_flat = msk.Expr.add(msk.Var.flatten(W_plus), msk.Var.flatten(W_minus))
        z1 = msk.Expr.vstack([msk.Expr.sub(s1, 1.), msk.Expr.mul(dagness_exp, sum_W_flat)])
        M.constraint("qc1", z1, msk.Domain.inQCone())

        # s1 <= s^{1/n}
        M.constraint(msk.Expr.vstack(s, 1.0, s1), msk.Domain.inPPowerCone(1 / n))

        # t >= ||S(I-W)||^2
        z2 = msk.Expr.mul(s_mat, msk.Expr.sub(msk.Matrix.eye(n), diff_W))
        M.constraint("rqc1", msk.Expr.vstack(t, .5, msk.Expr.flatten(z2)), msk.Domain.inRotatedQCone())

        # sum(W) >= n/(n-2)dagness_exp
        normW1 = msk.Expr.sum(sum_W)
        M.constraint("lin1", normW1, msk.Domain.greaterThan(n/((n-2)*dagness_exp)))

        # Set the objective function
        obj_tr = msk.Expr.dot(C.T, sum_W)
        obj_vec = msk.Expr.vstack([t, obj_tr, s, normW1])
        obj = msk.Expr.dot([1., 1., dagness_pen * (n - 1) / gamma, l1_pen], obj_vec)

        M.objective(msk.ObjectiveSense.Minimize, obj)

        M.solve()
        M.selectedSolution(msk.SolutionType.Interior)

        next_W_plus = M.getVariable('W_plus').level().reshape(n, n)
        next_W_minus = M.getVariable('W_minus').level().reshape(n, n)

    # compute w_tilde: getting rid of ambiguous edges
    tilde_W_plus = np.maximum(next_W_plus - next_W_minus, 0.0)
    tilde_W_minus = np.maximum(next_W_minus - next_W_plus, 0.0)
    tilde_sum = tilde_W_plus + tilde_W_minus
    # If we stay in the right space
    if np.sum(tilde_sum) >= n/((n-2)*dagness_exp):
        # Thresholding
        tilde_W_plus[tilde_W_plus < 0.4] = 0
        tilde_W_minus[tilde_W_minus < 0.4] = 0
        return tilde_W_plus, tilde_W_minus
    else:
        # Thresholding
        next_W_plus[next_W_plus < 0.4] = 0
        next_W_minus[next_W_minus < 0.4] = 0
        return next_W_plus, next_W_minus
