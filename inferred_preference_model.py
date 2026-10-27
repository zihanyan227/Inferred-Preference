"""
Inferring A's own preference from A's decision, given A's knowledge of B and A's care for B.
Adapted from Davis, Carlson, Dunham & Jara-Ettinger (2023), Cognition 240, 105580.

    P(R_A | d, K, W)  ∝  P(d | R_A, K, W)  ·  P(R_A | K, W)
       posterior          likelihood           prior
                       (generative model)

Paper inverts the same generative model for W (Exp 1) and K (Exp 2);
here we invert it for R_A (A's own reward function).

Setup: a box with n1 of item 1 (brownie, "B") and n2 of item 2 (cupcake, "C").
A picks k items first; B gets whatever is left.

W (A's care for B) is only ever given to the observer as one of three conditions,
mirroring K:
    'A cares more about self'  (w_A > w_B,  care c < 0.5)
    'A cares equally'          (w_A = w_B,  care c = 0.5)
    'A cares more about B'     (w_A < w_B,  care c > 0.5)
Inside each condition the observer puts a uniform distribution over the care values
on CARE_GRID that satisfy it, and the likelihood is marginalised over them.
"""
import itertools
import numpy as np
from scipy import stats

ITEM = ("brownie", "cupcake")

# --------------------------------------------------------------------------
# Hypothesis spaces
# --------------------------------------------------------------------------
R_GRID = np.round(np.linspace(0, 1, 11), 2)              # per-item reward levels
R_SPACE = np.array(list(itertools.product(R_GRID, R_GRID)))   # (121, 2): (r1, r2)


def belief_K(kind):
    """A's belief K over B's reward function, as a distribution over R_SPACE.
    kind: 'unknown' | 'B prefers brownie' | 'B prefers cupcake' | 'B likes both'
          or a tuple (r1, r2) for exact knowledge."""
    r1, r2 = R_SPACE[:, 0], R_SPACE[:, 1]
    if isinstance(kind, tuple):
        m = np.all(np.isclose(R_SPACE, kind), axis=1).astype(float)
    elif kind == "unknown":
        m = np.ones(len(R_SPACE))
    elif kind == "B prefers brownie":
        m = (r1 > r2).astype(float)
    elif kind == "B prefers cupcake":
        m = (r2 > r1).astype(float)
    elif kind == "B likes both":
        m = np.isclose(r1, r2).astype(float)
    else:
        raise ValueError(kind)
    return m / m.sum()


CARE_GRID = np.round(np.linspace(0, 1, 21), 3)            # care c = w_B / (w_A + w_B)


def W_space(norm="sum1"):
    """Hypothesis space over W = (w_A, w_B), shape (n, 2).
    norm='sum1' : w_A + w_B = 1, i.e. W = (1 - c, c) for c in CARE_GRID   (default)
    norm='fixA' : w_A = 1, w_B = c / (1 - c)  (same care ratios, A never cares less about self)"""
    c = CARE_GRID
    if norm == "sum1":
        return np.stack([1 - c, c], axis=1)
    if norm == "fixA":
        c = np.clip(c, 0, 0.95)                               # avoid w_B -> infinity at c = 1
        return np.stack([np.ones_like(c), c / (1 - c)], axis=1)
    raise ValueError(norm)


W_CONDITIONS = ("A cares more about self", "A cares equally", "A cares more about B")


def belief_W(kind):
    """Observer's belief about A's social weights: a uniform distribution over the
    CARE_GRID values allowed by one of the three W conditions (analogous to belief_K).
      'A cares more about self' -> uniform over c in {0, .05, ..., .45}
      'A cares equally'         -> c = .5
      'A cares more about B'    -> uniform over c in {.55, ..., 1}"""
    c = CARE_GRID
    if kind == "A cares more about self":
        m = (c < 0.5).astype(float)
    elif kind == "A cares equally":
        m = np.isclose(c, 0.5).astype(float)
    elif kind == "A cares more about B":
        m = (c > 0.5).astype(float)
    else:
        raise ValueError(f"W must be one of {W_CONDITIONS}, got {kind!r}")
    return m / m.sum()


def _W_points(W, norm="sum1"):
    """W condition -> (array of (w_A, w_B) points inside it, their uniform probabilities)."""
    pW = belief_W(W)
    keep = pW > 0
    return W_space(norm)[keep], pW[keep]


# --------------------------------------------------------------------------
# Reward with diminishing returns (paper eq. 2.3)
# --------------------------------------------------------------------------
def D(x, rho):
    """Discounted count: 1 + rho + rho^2 + ... (x terms)."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isclose(rho, 1.0), x, (1 - rho ** x) / (1 - rho + 1e-12))


def decisions(n1, n2, k):
    return [(x, k - x) for x in range(k + 1) if x <= n1 and k - x <= n2]


def rho_prior(a, b, n=25):
    """Discretised Beta(a, b) prior over discount rate rho (integrated out, as in paper).
    a=None -> no discounting (rho = 1)."""
    """ the prior will be a distribution over discount rates"""
    if a is None:
        return np.array([1.0]), np.array([1.0])
    edges = np.linspace(0, 1, n + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w = np.diff(stats.beta.cdf(edges, a, b))
    return mids, w / w.sum()


# --------------------------------------------------------------------------
# Likelihood  P(d | R_A, K, W)   (paper eq. 2.2 + softmax)
# --------------------------------------------------------------------------
def likelihood(box, k, K, W, beta=10.0, rho=(4, 2), norm="sum1"):
    """Returns array (n_RA, n_decisions).
    W: one of the three W conditions. The likelihood is marginalised over the care
       values inside that condition (uniform), just as rho is:
       P(d | R_A, K, W) = sum_c P(c | W) P(d | R_A, K, c)."""
    n1, n2 = box
    ds = decisions(n1, n2, k)
    Er1, Er2 = K @ R_SPACE[:, 0], K @ R_SPACE[:, 1]       # vector K * R_space <- expected value of B's reward
    rhos, pw = rho_prior(*rho) if rho is not None else rho_prior(None, None)
    Wpts, pWs = _W_points(W, norm)
    L = np.zeros((len(R_SPACE), len(ds)))
    for (wA, wB), pW in zip(Wpts, pWs):                    # over possible social weights
        for rh, p in zip(rhos, pw):                        # over possible discount rates
            U = np.zeros_like(L)
            for j, (x, y) in enumerate(ds):
                RA = R_SPACE[:, 0] * D(x, rh) + R_SPACE[:, 1] * D(y, rh)
                ERB = Er1 * D(n1 - x, rh) + Er2 * D(n2 - y, rh)
                U[:, j] = wA * RA + wB * ERB
            U = beta * U
            U -= U.max(axis=1, keepdims=True)
            P = np.exp(U)
            L += pW * p * P / P.sum(axis=1, keepdims=True)
    return L, ds


# --------------------------------------------------------------------------
# Prior  P(R_A | K, W)
# --------------------------------------------------------------------------
def prior_RA(K, W, projection=0.0, norm="sum1"):
    """Default: uniform (A's taste independent of K and W).
    projection > 0: 'birds of a feather' — the more A cares about B, the more A is
    assumed to share the taste A attributes to B. projection < 0: contrast."""
    dA = R_SPACE[:, 0] - R_SPACE[:, 1]
    dB = K @ (R_SPACE[:, 0] - R_SPACE[:, 1])
    Wpts, pWs = _W_points(W, norm)
    care = pWs @ (Wpts[:, 1] / Wpts.sum(axis=1))           # expected care within the W condition
    logp = projection * care * dA * dB * 10
    p = np.exp(logp - logp.max())
    return p / p.sum()


# --------------------------------------------------------------------------
# Posterior  P(R_A | d, K, W)
# --------------------------------------------------------------------------
def posterior(d, box=(2, 2), k=2, K="unknown", W="A cares equally",
              beta=10.0, rho=(4, 2), projection=0.0, norm="sum1"):
    """W: 'A cares more about self' | 'A cares equally' | 'A cares more about B'."""
    Kv = belief_K(K) if not isinstance(K, np.ndarray) else K
    L, ds = likelihood(box, k, Kv, W, beta, rho, norm)
    post = L[:, ds.index(tuple(d))] * prior_RA(Kv, W, projection, norm)
    return post / post.sum()


def summarize(post):
    r1, r2 = R_SPACE[:, 0], R_SPACE[:, 1]
    pB = post[r1 > r2].sum()
    pC = post[r2 > r1].sum()
    return dict(P_prefers_brownie=pB, P_prefers_cupcake=pC,
                P_equal=1 - pB - pC,
                pref_index=pB - pC,                      # in [-1, 1]
                E_r1_minus_r2=post @ (r1 - r2))


if __name__ == "__main__":
    WS = W_CONDITIONS
    for K in ["unknown", "B prefers brownie", "B prefers cupcake"]:
        for W in WS:
            for d in [(2, 0), (1, 1), (0, 2)]:
                s = summarize(posterior(d, K=K, W=W))
                print(f"K={K:18s} W={W:24s} A takes {d}:  "
                      f"P(brownie)={s['P_prefers_brownie']:.2f}  "
                      f"P(cupcake)={s['P_prefers_cupcake']:.2f}  "
                      f"index={s['pref_index']:+.2f}")
