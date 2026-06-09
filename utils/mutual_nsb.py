import numpy as np
import infomeasure as im

from pyitlib import discrete_random_variable as drv

def joint_encode(cols):
    """
    Encode one or more discrete columns into a single discrete label.
    cols: list/tuple of 1D arrays, all same length
    returns: 1D integer array
    """
    A = np.column_stack(cols)
    _, inv = np.unique(A, axis=0, return_inverse=True)
    return inv

def cmi_target_vs_rest(
    A,
    X,
    cond_cols=None,
    approach="nsb",
    base=2,
    K = 5
):
    """
    Estimate I(X_target ; X_rest | X_cond) for a discrete dataset.

    X: array of shape (N, d)
    target_col: int
    cond_cols: None, int, or iterable of ints
        Column(s) to condition on.
        If None, this reduces to ordinary MI between target and rest.
    """
    X = np.asarray(X)
    N, d = X.shape

    if isinstance(cond_cols, int):
        cond_cols = [cond_cols]
    elif cond_cols is None:
        cond_cols = []
    else:
        cond_cols = list(cond_cols)
    
    a_ = max(0, cond_cols[0] - K//2)
    b_ = min(d, cond_cols[0] + K//2 + max(0, K//2 - cond_cols[0]))
    # print("a_, b_ ", a_, b_)
    rest_cols = [j for j in range(a_, b_) if j not in cond_cols]
    # rest_cols = rest_cols[:1]
    if len(rest_cols) == 0:
        raise ValueError("No 'rest' variables left after removing target and cond_cols")

    x = A
    y = joint_encode([X[:, j] for j in rest_cols])

    # if len(cond_cols) == 0:
    #     return im.mutual_information(x, y, approach=approach, base=base)

    z = joint_encode([X[:, j] for j in cond_cols])
    cmi_bits = drv.information_mutual_conditional(
                    x, y, z,
                    estimator= "JAMES-STEIN", # "JAMES-STEIN",   # or "GOOD-TURING", "PERKS", "MINIMAX", "ML"
                    base=2
                    )
    return cmi_bits

def mutual_nsb_cal(xn, Yn, xn_index = 0, k = 5):
    # print("xn ", xn[:10])
    # print("X ", Yn[:10])
    # print("xn_index ", xn_index)
    CIL = cmi_target_vs_rest(A= xn, X= Yn, cond_cols= xn_index, K = k)

    # print("CIL nsb", CIL)

    return CIL

# import numpy as np
# from pyitlib import discrete_random_variable as drv


# def _factorize_1d(a):
#     a = np.asarray(a).ravel()
#     _, inv = np.unique(a, return_inverse=True)
#     codes = inv.astype(np.int64)
#     K = int(codes.max()) + 1 if codes.size else 0
#     return codes, K


# def _factorize_2d(Y):
#     Y = np.asarray(Y)
#     if Y.ndim == 1:
#         Y = Y.reshape(-1, 1)
#     if Y.ndim != 2:
#         raise ValueError("y must be 1D or 2D.")

#     N, m = Y.shape
#     Y_codes = np.empty((N, m), dtype=np.int64)
#     Y_cards = np.empty(m, dtype=np.int64)

#     for j in range(m):
#         _, inv = np.unique(Y[:, j], return_inverse=True)
#         Y_codes[:, j] = inv.astype(np.int64)
#         Y_cards[j] = int(Y_codes[:, j].max()) + 1 if N > 0 else 0

#     return Y_codes, Y_cards


# def _joint_encode_columns(cols_codes, cols_cards):
#     """
#     cols_codes: list of 1D integer-coded arrays, same length N
#     cols_cards: list of cardinalities for those arrays
#     Returns:
#         joint_codes: shape (N,), dtype=int64
#         K_joint: int
#     """
#     if len(cols_codes) == 0:
#         return None, 1

#     N = len(cols_codes[0])
#     joint = np.zeros(N, dtype=np.int64)
#     mult = 1

#     for arr, K in zip(reversed(cols_codes), reversed(cols_cards)):
#         arr = np.asarray(arr, dtype=np.int64).ravel()
#         joint += arr * mult
#         mult *= int(K)

#     return joint, int(mult)


# def _cmi_single(x_codes, Kx, y_codes, Ky, cond_codes, Kcond, estimator, base):
#     """
#     Compute I(X ; Y | Cond) for single coded X, Y, and joint-coded Cond.
#     If no conditioning, falls back to I(X ; Y).
#     """
#     if cond_codes is None:
#         val = drv.information_mutual(
#             x_codes,
#             y_codes,
#             estimator=estimator,
#             base=base,
#             Alphabet_X=np.arange(Kx, dtype=np.int64),
#             Alphabet_Y=np.arange(Ky, dtype=np.int64),
#         )
#     else:
#         val = drv.information_mutual_conditional(
#             x_codes,
#             y_codes,
#             cond_codes,
#             estimator=estimator,
#             base=base,
#             Alphabet_X=np.arange(Kx, dtype=np.int64),
#             Alphabet_Y=np.arange(Ky, dtype=np.int64),
#             Alphabet_Z=np.arange(Kcond, dtype=np.int64),
#         )
#     return float(val)


# def cmi_chain_rule_sparse(
#     x,
#     y,
#     z=None,
#     k=1,
#     estimator="JAMES-STEIN",
#     base=2,
#     order="greedy",
#     return_details=False,
# ):
#     """
#     Sparse chain-rule approximation to I(X ; Y_block | Z), where:
#       - x: shape (N,)
#       - y: shape (N, m)
#       - z: shape (N,) or None

#     Approximation:
#       sum_j I(X ; Y_{pi_j} | Z, S_j)
#     where S_j contains at most k previously selected Y columns.

#     Parameters
#     ----------
#     x : array-like, shape (N,)
#     y : array-like, shape (N,) or (N, m)
#     z : array-like, shape (N,), optional
#     k : int
#         Max number of previous Y columns to keep in each conditioning set.
#         Typical values: 0, 1, 2.
#     estimator : str
#         pyitlib discrete estimator: "JAMES-STEIN", "GOOD-TURING", "ML", ...
#     base : int or float
#         2 for bits.
#     order : str
#         "greedy" = order Y columns by descending I(X ; Y_j | Z)
#         "natural" = use original column order
#     return_details : bool
#         If True, return per-term diagnostics too.

#     Returns
#     -------
#     total : float
#         Approximate I(X ; Y_block | Z)
#     """
#     x = np.asarray(x).ravel()
#     y = np.asarray(y)
#     if y.ndim == 1:
#         y = y.reshape(-1, 1)
#     if y.ndim != 2:
#         raise ValueError("y must be 1D or 2D.")

#     if z is not None:
#         z = np.asarray(z).ravel()

#     N = x.shape[0]
#     if y.shape[0] != N:
#         raise ValueError("x and y must have the same number of samples.")
#     if z is not None and z.shape[0] != N:
#         raise ValueError("z must have the same number of samples as x and y.")

#     k = max(0, int(k))

#     # Factorize inputs
#     x_codes, Kx = _factorize_1d(x)
#     y_codes, y_cards = _factorize_2d(y)

#     if z is None:
#         z_codes, Kz = None, 1
#     else:
#         z_codes, Kz = _factorize_1d(z)

#     m = y_codes.shape[1]

#     # Base score for ordering: I(X ; Y_j | Z)
#     base_scores = []
#     for j in range(m):
#         score_j = _cmi_single(
#             x_codes, Kx,
#             y_codes[:, j], int(y_cards[j]),
#             z_codes, Kz if z_codes is not None else 1,
#             estimator, base
#         )
#         base_scores.append(score_j)
#     base_scores = np.asarray(base_scores, dtype=float)

#     if order == "greedy":
#         order_idx = list(np.argsort(-base_scores))
#     elif order == "natural":
#         order_idx = list(range(m))
#     else:
#         raise ValueError("order must be 'greedy' or 'natural'.")

#     total = 0.0
#     details = []

#     # We'll keep the strongest previously selected columns (by base score)
#     prev_selected = []

#     for pos, j in enumerate(order_idx):
#         # choose up to k previous Y columns for conditioning
#         if len(prev_selected) == 0 or k == 0:
#             kept_prev = []
#         else:
#             prev_sorted = sorted(prev_selected, key=lambda idx: -base_scores[idx])
#             kept_prev = prev_sorted[:k]

#         cond_cols_codes = []
#         cond_cols_cards = []

#         # Always include Z first if provided
#         if z_codes is not None:
#             cond_cols_codes.append(z_codes)
#             cond_cols_cards.append(Kz)

#         # Add truncated previous Y columns
#         for idx in kept_prev:
#             cond_cols_codes.append(y_codes[:, idx])
#             cond_cols_cards.append(int(y_cards[idx]))

#         cond_joint, Kcond = _joint_encode_columns(cond_cols_codes, cond_cols_cards)

#         term = _cmi_single(
#             x_codes, Kx,
#             y_codes[:, j], int(y_cards[j]),
#             cond_joint, Kcond,
#             estimator, base
#         )

#         total += term
#         prev_selected.append(j)

#         details.append({
#             "term_pos": pos,
#             "y_col": int(j),
#             "base_score_I_XYj_given_Z": float(base_scores[j]),
#             "kept_prev_cols": [int(t) for t in kept_prev],
#             "term_value": float(term),
#         })

#     if return_details:
#         return float(total), {
#             "order": [int(j) for j in order_idx],
#             "base_scores": base_scores,
#             "terms": details,
#         }

#     return float(total)


# def mutual_nsb_cal(xn, Yn, xn_index = 0, k = 1):

    

#     rest_cols = [j for j in range(np.shape(Yn)[1]) if j != xn_index]

#     CIL, info = cmi_chain_rule_sparse(
#         xn,
#         y=Yn[:,rest_cols],
#         z=Yn[:,xn_index],
#         k=k,                    # keep only 1 previous Y in each term
#         estimator="JAMES-STEIN",
#         base=2,
#         order="greedy",
#         return_details=True,
#     )
#     CIL = max(0, CIL)
#     print("CIL nsb", CIL)

#     return CIL
