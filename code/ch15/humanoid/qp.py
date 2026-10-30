"""
The strictly convex QP of the repository's load sharing,

    minimize 0.5 x'Hx + f'x  subject to  blower <= [x[:ms]; A x] <= bupper

(ms = len(bupper) - rows of A, as daqp reads it), solved with daqp's
interface: solve(H, f, A, bupper, blower) returns (x, fval, exitflag, info).

Physical AI, Together, Chapter 15 (Section 15.3).

The QP is written, after a diagonal scaling of x and a Cholesky factor of
H, as a least-distance problem, which SciPy's NNLS solves exactly (Lawson
and Hanson, Solving Least Squares Problems, 1974, chapter 23). It is the
same problem that daqp solves, so the solution is the same to the solvers'
tolerances; humanoid_carry.py uses it where daqp is not installed (--qp
nnls) and says so in its record. A failure returns exitflag -1, which the
repository's share.py treats as daqp's: it keeps its last solution.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

INF = 1e29


def solve(H, f, A, bupper, blower=None, sense=None, **settings):
    H = np.asarray(H, float)
    n = H.shape[0]
    f = np.asarray(f, float)
    A = np.asarray(A, float).reshape(-1, n)
    bu = np.asarray(bupper, float)
    m = bu.size
    bl = np.full(m, -1e30) if blower is None else np.asarray(blower, float)
    if sense is not None and np.any(np.asarray(sense) != 0):
        raise NotImplementedError("only inequality constraints")
    ms = m - A.shape[0]
    C = np.vstack([np.eye(n)[:ms], A])
    upper, lower = bu < INF, bl > -INF
    Cm = np.vstack([C[upper], -C[lower]])
    dm = np.concatenate([bu[upper], -bl[lower]])
    fail = (np.full(n, np.nan), np.inf, -1, {"iterations": 0})
    try:
        s = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-300))
        R = np.linalg.cholesky(H * s[:, None] * s[None, :]).T       # D H D = R'R
        Ri = solve_triangular(R, np.eye(n))
        w = solve_triangular(R, f * s, trans="T")
        Cs = Cm * s[None, :]
        # z = R y + w, x = D y:  Cm x <= dm  becomes  G z >= h
        G = -Cs @ Ri
        h = -(dm + Cs @ (Ri @ w))
        if G.shape[0]:
            norm = np.linalg.norm(G, axis=1)
            norm[norm == 0.0] = 1.0
            E = np.vstack([(G / norm[:, None]).T, (h / norm)[None, :]])
            u, _ = nnls(E, np.r_[np.zeros(n), 1.0], maxiter=50 * E.shape[1])
            r = E @ u - np.r_[np.zeros(n), 1.0]
            if abs(r[n]) < 1e-14:
                return fail                                         # infeasible
            z = -r[:n] / r[n]
        else:
            z = np.zeros(n)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return fail
    x = s * (Ri @ (z - w))
    if not np.all(np.isfinite(x)):
        return fail
    viol = float(np.max(Cm @ x - dm)) if len(dm) else 0.0
    flag = 1 if viol <= 1e-6 * (1.0 + float(np.max(np.abs(dm), initial=0.0))) else 2
    return x, float(0.5 * x @ H @ x + f @ x), flag, {"iterations": 0, "violation": viol}
