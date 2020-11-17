import numpy as np

def compare_graphs(w_gt, w, verbose=True):
    """Compares two graphs and returns several metrics"""
    thres_w = np.abs(w) > 0.5
    undirected_w = thres_w + thres_w.T
    if verbose:
        print("Number of edges in ground truth graph", np.sum(w_gt))
        print("Number of edges in estimation", np.sum(thres_w))
    correct = np.sum(np.multiply(w_gt, thres_w))
    rev = np.sum(np.multiply(w_gt, np.multiply((1 - thres_w), thres_w.T)))
    missing = np.sum(w_gt) - np.sum(np.multiply(w_gt, undirected_w))
    extra = np.sum(thres_w) - np.sum(np.multiply(thres_w, (w_gt + w_gt.T)))
    shd = extra + rev + missing
    if verbose:
        print("Correct edges", correct)
        print("Reversed edges", rev)
        print("Missing edges", missing)
        print("Extra edges", extra)
        print("SHD", shd)
    return correct, extra, rev, missing, shd
