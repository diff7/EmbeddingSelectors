import numpy as np
from tqdm import tqdm
from scipy.spatial import distance


def k_center_greedy(X, s, b):
    """
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X that have already been selected
    - b: int, new selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    """
    n = X.shape[0]
    p = np.setdiff1d(np.arange(n), s, assume_unique=True)  # pool indices
    sel = np.empty(b, dtype=np.int64)

    sl = len(s)
    D = np.zeros([sl + b, len(p)], dtype=np.float32)
    D[:sl] = distance.cdist(X[s], X[p], metric="euclidean")  # shape (|s|,|p|)
    mins = np.min(D[:sl], axis=0)  # vector of length |p|
    cols = np.ones(len(p), dtype=bool)  # columns still in use

    # for i in tqdm(range(b), desc="Greedy k-Centers"):
    print("MAKING k-center")
    for i in tqdm(range(b)):
        j = np.argmax(mins)
        u = p[j]
        sel[i] = u

        if i == b - 1:
            break

        mins[j] = -1
        cols[j] = False

        # compute dist between selected point and remaining pool points
        r = sl + i + 1
        D[r, cols] = distance.cdist(X[u : u + 1], X[p[cols]])[0]
        mins = np.minimum(mins, D[r])
    print("MAKING k-center DONE")
    return sel