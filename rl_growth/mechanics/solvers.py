import numpy as np


def bisect_vec(f, lo, hi, n_iter=60, check_bracket=True, on_fail="nan"):
    """Vectorised bisection: find x in [lo, hi] with f(x) = 0 for every batch entry.

    ``f`` must accept and return arrays of the same shape.  ``lo``/``hi`` may be
    scalars or arrays.  The bracket must contain a sign change for every entry.
    """
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    lo, hi = np.broadcast_arrays(lo, hi)
    lo = lo.copy(); hi = hi.copy()
    flo = f(lo)
    fhi = f(hi)
    bad = None
    if check_bracket:
        bad = ~(np.sign(flo) != np.sign(fhi))
        if np.any(bad) and on_fail == "raise":
            raise RuntimeError(f"bisect_vec: no sign change in bracket for {int(bad.sum())} entries")
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    x = 0.5 * (lo + hi)
    if bad is not None and np.any(bad):
        x = np.where(bad, np.nan, x)      # caller decides how to treat unsolvable entries
    return x
