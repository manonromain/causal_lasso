import sys
import numpy as np
import time
from tqdm.autonotebook import tqdm
from scipy.linalg import sqrtm, expm
import cvxpy as cp
import logging
from IPython.core.debugger import set_trace
try:
    import mosek.fusion as msk
except ImportError:
    print("Couldn't import Mosek")
    pass
try:
    import torch
    from cvxpylayers.torch import CvxpyLayer
except ImportError:
    logging.info("If you want to use PyTorch CVXPY layers, you should install it first")
    logging.info("using pip install torch cvxpylayers --user")
    pass


CONSTANT = .5 * (1 + np.sqrt(1 + 4 / np.exp(1)))
DEBUG = 1

def dyn_no_lips_gen(X, W0_plus, W0_minus, dagness_exp, dagness_pen, l1_pen, eps=1e-4, solver="mosek", max_iter=200,
                    logging_dict=False, device=None):
    """ Main algorithm described in our paper

    Args:
        X   (np.array): sample matrix
        W0_plus  (np.array): initial positive part adj matrix
        W0_minus  (np.array): initial negative part adj matrix
        dagness_exp  (float): alpha in paper
        dagness_pen  (float): mu in paper
        l1_pen (float): lambda in paper
        eps    (float): sensivity for early stopping
        solver   (str): solver to use
        max_iter (int): maximum number of iterations
        logging_dict (bool): if enabled, returns all objective values + additional info (useful for analysis)

    Returns:
        Wk     (np.array): current iterate of weighted adj matrix
        logging_np (dict): if logging is True, dict of metrics/info
    """

    m, n = X.shape
    prev_support = np.abs(W0_plus - W0_minus) > 0.5
    #if m > n:
    #    s_mat = 1 / m * np.real(sqrtm(X.T @ X))
    #else:
    s_mat = 1 / np.sqrt(m) * X

    # Functions
    def dag_penalty(W_plus, W_minus):
        sum_W = W_plus + W_minus
        return dagness_pen * np.trace(expm(dagness_exp * sum_W))

    def grad_f_scalar_H(W_plus, W_minus, H_plus, H_minus):
        """ Returns <∇f(W), D> """
        sum_W = W_plus + W_minus
        sum_H = H_plus + H_minus
        expW = expm(dagness_exp * sum_W)
        return dagness_pen * dagness_exp * np.trace(expW @ sum_H)

    def distance_kernel(Wx_plus, Wx_minus, Wy_plus, Wy_minus):
        """
        the kernel used is Ce^(||W||_F**2) where C = dagness_pen/2
        """

        # sum_x = Wx_plus + Wx_minus
        # sum_y = Wy_plus + Wy_minus
        # norm_y_sq = np.sum(sum_y ** 2)
        # norm_x_sq = np.sum(sum_x ** 2)
        # product = np.trace(sum_y.T @ (sum_x - sum_y))
        #
        # hWx = np.exp(dagness_exp ** 2 * norm_x_sq)
        # hWy = np.exp(dagness_exp ** 2 * norm_y_sq)
        # grad_hy_scalar_x_minus_y = 2 * dagness_exp * hWy * product
        # assert hWx >= hWy + grad_hy_scalar_x_minus_y, (hWx, hWy + grad_hy_scalar_x_minus_y)

        sum_x = Wx_plus.flatten() + Wx_minus.flatten()
        sum_y = Wy_plus.flatten() + Wy_minus.flatten()
        norm_y_sq = np.sum(sum_y ** 2)
        norm_x_sq = np.sum(sum_x ** 2)
        product = np.sum(sum_y * (sum_x - sum_y))
        #
        hWx = np.exp(dagness_exp * norm_x_sq)
        hWy = np.exp(dagness_exp * norm_y_sq)
        grad_hy_scalar_x_minus_y = 2 * dagness_exp * product
        if DEBUG:
            assert hWx/hWy >= 1 + grad_hy_scalar_x_minus_y, (hWx/hWy, 1 + grad_hy_scalar_x_minus_y)
        return .5 * dagness_pen * (hWx - hWy * (1 + grad_hy_scalar_x_minus_y))

    if logging_dict:
        log_dict = {"dagness_exp": dagness_exp, "dagness_pen": dagness_pen, "l1_pen": l1_pen, # constants
                   "time": [], "l2_error": [], "l1_val": [], "dag_constraint": [],
                   "nb_change_support":[], "support": [], "support_pos": [], "support_neg": [],
                   "gammas":[]
                    }

    # Constants
    gamma = 0.9

    # Init
    Wk_plus, Wk_minus = W0_plus, W0_minus
    l2_error_curr, l2_error_prev = 0, 0
    start = time.time()
    it_nolips = 0
    pbar = tqdm(desc="Causal Lasso", total=max_iter)
    if solver == "cvxpylayers":
        layer = layer_cvxtorch(s_mat, dagness_pen, dagness_exp)
    while (it_nolips < 2 or (np.abs((l2_error_prev - l2_error_curr)/l2_error_prev) >= eps)) and (it_nolips < max_iter):
        # it = 0
        #while True:
        # if gamma < 1/2:
        #     np.save("23_02_21_wrong_w_plus", next_W_plus)
        #     np.save("23_02_21_wrong_w_minus", next_W_minus)
        #     np.save("23_02_21_wrong_wk_plus", Wk_plus)
        #     np.save("23_02_21_wrong_wk_minus", Wk_minus)
        #     print("===========================")
        #     logging.warning("violated L-smad")
        #     break

        # try:  # TODO more solvers
        if solver == "mosek":
            next_W_plus, next_W_minus = bregman_map_mosek(s_mat, Wk_plus, Wk_minus,
                                                          gamma, l1_pen, dagness_pen, dagness_exp)
        elif solver == "cvxpylayers":
            next_W_plus, next_W_minus = apply_bregman_map_cvxtorch(layer, Wk_plus, Wk_minus,
                                                        gamma, l1_pen, dagness_pen, dagness_exp)
        else:
            next_W_plus, next_W_minus = bregman_map_cvx(s_mat, Wk_plus, Wk_minus,
                                                        gamma, l1_pen, dagness_pen, dagness_exp)
        # Sufficient decrease condition
        # if dag_penalty(next_W_plus, next_W_minus) - dag_penalty(Wk_plus, Wk_minus) \
        #    - grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus, next_W_minus - Wk_minus)  \
        #    > 1 / gamma * distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus):
        #     gamma = gamma / 2
        #     it += 1
        # else:
        #     break
        gamma_k = gamma
        # Trying increasing step size
        # gamma = min(2 * gamma, 10000)

        # TODO delete
        if np.sum(next_W_minus + next_W_plus) < np.sqrt(n)*CONSTANT:
            logging.warning("assertion false: iteration map may not be stable")


        ##### DEBUG
        def phi(W_plus, W_minus):
            g = np.linalg.norm(s_mat @ (np.eye(n) - W_plus + W_minus), "fro") ** 2 + l1_pen * np.sum(W_plus + W_minus)
            f = dag_penalty(W_plus, W_minus)
            return f + g, f, g


        # verify all lemmas and prop
        # descent lemma
        if DEBUG:
            assert np.abs(phi(next_W_plus, next_W_minus)[1] - phi(Wk_plus, Wk_minus)[1] - grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus,
                                                                       next_W_minus - Wk_minus)) \
                <= distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus),  \
                (np.abs(phi(next_W_plus, next_W_minus)[1] - phi(Wk_plus, Wk_minus)[1] - grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus,
                                                                       next_W_minus - Wk_minus)), \
                distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus))

            # Lemma 4.1 (eq 4.3) - is the bregman solved?
            assert phi(next_W_plus, next_W_minus)[2] + grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus,
                                                                       next_W_minus - Wk_minus) \
                + 1 / gamma * distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus) <= phi(Wk_plus, Wk_minus)[2], \
                (phi(next_W_plus, next_W_minus)[2], grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus,
                                                                       next_W_minus - Wk_minus), 1 / gamma * distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus),
                 phi(next_W_plus, next_W_minus)[2] + grad_f_scalar_H(Wk_plus, Wk_minus, next_W_plus - Wk_plus,
                                                                       next_W_minus - Wk_minus)
                + 1 / gamma * distance_kernel(next_W_plus, next_W_minus, Wk_plus, Wk_minus), "<=",
                 phi(Wk_plus, Wk_minus)[2])

            # (eq 4.2)
            assert phi(next_W_plus, next_W_minus)[0] <= phi(Wk_plus, Wk_minus)[0] \
                - (1/gamma - 1) * distance_kernel(next_W_plus, next_W_minus,  Wk_plus, Wk_minus)




        # Compute current iterate
        Wk = next_W_plus - next_W_minus
        Wk_plus, Wk_minus = next_W_plus, next_W_minus
        l2_error_prev = l2_error_curr
        l2_error_curr = np.linalg.norm(s_mat @ (np.eye(n) - next_W_plus + next_W_minus), "fro")**2
        dag_penalty_k = dag_penalty(Wk_plus, Wk_minus)


        if logging_dict:
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

        if it_nolips % 10 == 0:
            if it_nolips >= 50:
                dagness_pen *= 10
            logging.info("Objective value at iteration {}".format(it_nolips))
            logging.info(l2_error_curr + dag_penalty_k + l1_pen * np.sum(Wk_plus + Wk_minus))

        it_nolips += 1
        pbar.update(1)
    logging.info("Done in", time.time() - start, "s and", it_nolips, "iterations")

    # Thresholding
    # Wk[np.abs(Wk) < 0.3] = 0
    if logging_dict:
        logging_np = {k: np.array(v) for k, v in log_dict.items()}
        return Wk, logging_np
    else:
        return Wk, {}


def compute_C(n, sum_Wk, dagness_pen, dagness_exp, inv_gamma):
    sum_Wk_norm_sq = np.sum(sum_Wk ** 2)
    # C = grad f - 1/gamma grad h
    C_1 = dagness_exp * expm(dagness_exp * sum_Wk)
    C_2 = - inv_gamma * dagness_exp * \
        np.exp(dagness_exp * sum_Wk_norm_sq) * sum_Wk.T
    return dagness_pen * C_1, dagness_pen * C_2


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
    #W_plus.value = Wk_plus_value
    W_minus = cp.Variable((n, n), nonneg=True)
    #W_minus.value = Wk_minus_value
    sum_W = W_plus + W_minus  # sum variable

    obj_ll = cp.norm(s_mat @ (np.eye(n) - W_plus + W_minus), "fro") ** 2
    obj_spars = l1_pen * cp.sum(W_plus + W_minus)

    # Compute C
    sum_Wk = Wk_plus_value + Wk_minus_value
    C_f, C_h = compute_C(n, sum_Wk, dagness_pen, dagness_exp, 1 / gamma)

    if DEBUG:
        print("Previous solution satisfies all constraints", np.sum(sum_Wk) >= np.sqrt(n) * CONSTANT, np.all(Wk_minus_value >= 0),
              np.all(Wk_plus_value >= 0))
    obj_trace_f = cp.trace(C_f @ (sum_W - sum_Wk))
    obj_trace_h = cp.trace(C_h @ (sum_W - sum_Wk))
    norm_sq_W = cp.sum_squares(sum_W.flatten())
    obj_kernel = 1 / gamma * dagness_pen * .5 * cp.exp(dagness_exp * norm_sq_W)
    obj_kernel_k = 1 / gamma * dagness_pen * .5 * np.exp(dagness_exp * np.sum(sum_Wk ** 2))

    obj = obj_ll + obj_spars + obj_trace_f + obj_trace_h + obj_kernel - obj_kernel_k
    # value_before_opt = obj.value
    # if DEBUG:
    #     print("value before", value_before_opt)
    #     print("\t ll", obj_ll.value)
    #     print("\t sp", obj_spars.value)
    #     print("\t tr", obj_trace_f.value)
    #     print("\t kern", (obj_kernel - obj_kernel_k + obj_trace_h).value)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(sum_W) >= np.sqrt(n) * CONSTANT])
    try:
        prob.solve()
        value_after_opt = obj.value
        if DEBUG:
            print("value after", value_after_opt)
            print("\t ll", obj_ll.value)
            print("\t sp", obj_spars.value)
            print("\t tr", obj_trace_f.value)
            print("\t kern", (obj_kernel - obj_kernel_k + obj_trace_h).value, "=", obj_kernel.value, "-",  obj_kernel_k, "+",
                  obj_trace_h.value)
            print(prob.status)
    except cp.SolverError:
        out = prob.solve(verbose=True)
        if DEBUG:
            print(out, obj.value)
    if prob.status != "optimal":
        W_plus.value = Wk_plus_value
        W_minus.value = Wk_minus_value
        out = prob.solve(verbose=True)
        if DEBUG:
            print(out, obj.value)

    next_W_plus, next_W_minus = np.maximum(W_plus.value, 0), np.maximum(W_minus.value, 0)

    # tilde_W_plus = np.maximum(next_W_plus - next_W_minus, 0.0)
    # tilde_W_minus = np.maximum(next_W_minus - next_W_plus, 0.0)
    # #
    # if np.sum(tilde_W_plus + tilde_W_minus) >= np.sqrt(n) * CONSTANT:
    #     return tilde_W_plus, tilde_W_minus

    return next_W_plus, next_W_minus


def layer_cvxtorch(s_mat, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with new CVXPY layers and PyTorch
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
    # FIXME
    raise NotImplementedError
    n = s_mat.shape[1]

    # Variables
    W_plus = cp.Variable((n, n), nonneg=True)
    W_minus = cp.Variable((n, n), nonneg=True)

    # Parameters
    inv_gamma_param = cp.Parameter(nonneg=True)
    l1_pen_param = cp.Parameter(nonneg=True)
    C_param = cp.Parameter((n,n))
    sum_W = W_plus + W_minus  # sum variable

    obj_ll = cp.norm(s_mat @ (np.eye(n) - W_plus + W_minus), "fro") ** 2
    obj_spars = l1_pen_param * cp.sum(W_plus + W_minus)

    obj_trace = cp.trace(C_param @ sum_W)
    obj_kernel = inv_gamma_param * (dagness_pen * (n - 1) * (1 + dagness_exp * cp.norm(sum_W, "fro"))**n)

    obj = obj_ll + obj_spars + obj_trace + obj_kernel
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(W_plus) + cp.sum(W_minus) >= n/((n-2)*dagness_exp)])
    assert prob.is_dpp(), "{}{}{}{}".format()

    #set_trace()

    layer = CvxpyLayer(prob, parameters = [C_param, inv_gamma_param, l1_pen_param],
                       variables = [W_plus, W_minus])

    return layer


def apply_bregman_map_cvxtorch(layer, Wk_plus_value, Wk_minus_value,
                               gamma, l1_pen, dagness_pen, dagness_exp):
    """ Solves argmin g(W) + <grad f (Wk), W-Wk> + 1/gamma * Dh(W, Wk)
        with new CVXPY layers and PyTorch
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

    raise NotImplementedError
    n = Wk_plus_value.shape[0]

    #TODO allow GPU
    torch_gamma = torch.tensor(1 / gamma, dtype=torch.float32)
    torch_l1_pen = torch.tensor(l1_pen, dtype=torch.float32)
    # Compute C
    sum_Wk = Wk_plus_value + Wk_minus_value
    C = compute_C(n, sum_Wk, dagness_pen, dagness_exp, 1 / gamma)
    torch_C = torch.tensor(C, dtype=torch.float32)

    x_star = layer(torch_C, torch_gamma, torch_l1_pen)
    #set_trace()
    next_W_plus, next_W_minus = x_star[0].numpy(), x_star[1].numpy()

    tilde_W_plus = np.maximum(next_W_plus - next_W_minus, 0.0)
    tilde_W_minus = np.maximum(next_W_minus - next_W_plus, 0.0)
    tilde_sum = tilde_W_plus + tilde_W_minus
    #
    if np.sum(tilde_sum) >= n / ((n - 2) * dagness_exp):
        # Thresholding
        # tilde_W_plus[tilde_W_plus < 0.4] = 0
        # tilde_W_minus[tilde_W_minus < 0.4] = 0
        return tilde_W_plus, tilde_W_minus
    else:
        # Thresholding
        # next_W_plus[next_W_plus < 0.4] = 0
        # next_W_minus[next_W_minus < 0.4] = 0
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
    C_f, C_h = compute_C(n, sum_Wk, dagness_pen, dagness_exp, 1 / gamma)

    #
    obj_kernel_k = 1 / gamma * dagness_pen * .5 * np.exp(dagness_exp * np.sum(sum_Wk ** 2))

    if DEBUG:
        print("Previous solution satifies assump", np.sum(sum_Wk) >= np.sqrt(n) * CONSTANT, np.all(Wk_plus_value >= 0),
              np.all(Wk_minus_value >= 0), np.all(np.diag(Wk_minus_value + Wk_plus_value) == np.zeros(n)))

    with msk.Model('model') as M:
        W_plus = M.variable('W_plus', [n, n], msk.Domain.greaterThan(0.))
        W_minus = M.variable('W_minus', [n, n], msk.Domain.greaterThan(0.))
        #W_plus.setLevel(Wk_plus_value.flatten())
        #W_minus.setLevel(Wk_minus_value.flatten())
        sum_W = msk.Expr.add(W_plus, W_minus)
        diff_W = msk.Expr.sub(W_plus, W_minus)
        t = M.variable('t')
        y = M.variable("y")
        s = M.variable("s")

        #  y >= dagness_exp||(W+ + W-)||^2
        sum_W_flat = msk.Expr.add(msk.Var.flatten(W_plus), msk.Var.flatten(W_minus))
        M.constraint("qc1", msk.Expr.vstack(y, .5/dagness_exp, sum_W_flat), msk.Domain.inRotatedQCone())

        #  s >= e^y
        M.constraint(msk.Expr.vstack(s, 1.0, y), msk.Domain.inPExpCone())

        # t >= ||S(I-W)||^2
        z2 = msk.Expr.mul(s_mat, msk.Expr.sub(msk.Matrix.eye(n), diff_W))
        M.constraint("rqc1", msk.Expr.vstack(t, .5, msk.Expr.flatten(z2)), msk.Domain.inRotatedQCone())

        # # ||W|| >= CONSTANT # C_alpha
        normW1 = msk.Expr.sum(sum_W)
        M.constraint("lin1", normW1, msk.Domain.greaterThan(np.sqrt(n)*CONSTANT))

        # # Constrain diag to be zero
        M.constraint(W_plus.diag(), msk.Domain.equalsTo(0.0))
        #
        # # Constrain diag to be zero
        M.constraint(W_minus.diag(), msk.Domain.equalsTo(0.0))


        # Set the objective function
        obj_tr_f = msk.Expr.dot(C_f.T, msk.Expr.sub(sum_W, sum_Wk))
        obj_tr_h = msk.Expr.dot(C_h.T, msk.Expr.sub(sum_W, sum_Wk))
        obj_vec = msk.Expr.vstack([t, obj_tr_f, obj_tr_h, s, normW1])
        obj = msk.Expr.dot([1., 1., 1., dagness_pen * 0.5 / gamma, l1_pen], obj_vec)

        obj = msk.Expr.sub(obj, obj_kernel_k)

        M.objective(msk.ObjectiveSense.Minimize, obj)
        #print("MSK - Value before:", M.primalObjValue())
        try:
            # M.setLogHandler(sys.stdout)
            M.solve()
            M.selectedSolution(msk.SolutionType.Interior)
            next_W_plus = np.maximum(M.getVariable('W_plus').level().reshape(n, n), 0)
            next_W_minus = np.maximum(M.getVariable('W_minus').level().reshape(n, n), 0)

            if DEBUG:
                print("MSK - Value after:",  M.primalObjValue())

                print("Value before=", np.sum((s_mat @ (np.eye(n) - Wk_plus_value + Wk_minus_value)) ** 2) + l1_pen * np.sum(sum_Wk))
                new_sum = next_W_plus + next_W_minus
                print("g=", np.sum((s_mat @ (np.eye(n) - next_W_plus + next_W_minus))**2) + l1_pen*np.sum(new_sum))
                print("trf=", np.trace(C_f @ (new_sum - sum_Wk)))
                print("kern=", .5 * dagness_pen * np.exp(dagness_exp * np.sum(new_sum ** 2)) - obj_kernel_k
                      + np.trace(C_h @ (new_sum - sum_Wk)))

                # print("No loose inequalities = ")
                # print(M.getVariable('y').level(), ">=", dagness_exp* np.sum(new_sum**2))
                # print(.5 * dagness_pen * M.getVariable('s').level(), ">=", .5 * dagness_pen * np.exp(M.getVariable('y').level()), .5 * dagness_pen * np.exp(dagness_exp * np.sum(new_sum ** 2)))
                # print(M.getVariable('t').level(), ">=", np.sum((s_mat @ (np.eye(n) - next_W_plus + next_W_minus))**2))
                # print("except:", np.sum(new_sum), ">=", np.sqrt(n) * CONSTANT)
        except msk.SolutionError:
            print("stopped because of Solution Error")
            return Wk_plus_value, Wk_minus_value




    # compute w_tilde: getting rid of ambiguous edges
    tilde_W_plus = np.maximum(next_W_plus - next_W_minus, 0.0)
    tilde_W_minus = np.maximum(next_W_minus - next_W_plus, 0.0)
    tilde_sum = tilde_W_plus + tilde_W_minus
    # If we stay in the right space
    # set_trace()
    #if np.sum(tilde_sum) >= np.sqrt(n)*CONSTANT:
        # Thresholding
        # tilde_W_plus[tilde_W_plus < 0.4] = 0
        # tilde_W_minus[tilde_W_minus < 0.4] = 0
    #    return tilde_W_plus, tilde_W_minus
    #else:
        # Thresholding
        # next_W_plus[next_W_plus < 0.4] = 0
        # next_W_minus[next_W_minus < 0.4] = 0
    return next_W_plus, next_W_minus


def init_no_lips(s_mat, l1_pen):
    """
        Solves argmin g(W)
        with CVX

        Args:
        s_mat(np.array): data  matrix
        l1_pen(float): lambda in paper
    """


    n = s_mat.shape[1]

    W_plus = cp.Variable((n, n), nonneg=True)
    W_minus = cp.Variable((n, n), nonneg=True)
    sum_W = W_plus + W_minus  # sum variable

    obj_ll = cp.norm(s_mat @ (np.eye(n) - W_plus + W_minus), "fro") ** 2
    obj_spars = l1_pen * cp.sum(W_plus + W_minus)

    obj = obj_ll + obj_spars
    prob = cp.Problem(cp.Minimize(obj), [cp.diag(W_plus) == np.zeros(n), cp.diag(W_minus) == np.zeros(n)])
    prob.solve()

    if prob.status != "optimal":
        prob.solve(verbose=True)

    next_W_plus, next_W_minus = np.maximum(W_plus.value, 0), np.maximum(W_minus.value, 0)

    return next_W_plus, next_W_minus
