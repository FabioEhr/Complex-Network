"""random_walk_betweenness_directed.py

Exact *random‑walk / current‑flow betweenness* for a **directed** graph
using the pseudoinverse of the *random‑walk Laplacian* (Boley et al.,
2018 §2‑4).

───────────────────────────────────────────────────────────────────────
  Input               :  any NetworkX DiGraph (strongly‑connected)
  Output              :  dict {node: centrality}
  Complexity          :  one dense SVD  O(n³)   (n≈2 k → ~1‑2 min)
                         centrality     O(n²)   (4 M vector ops)
  Memory (n=2 063)    :  ≈ 32 MiB for dense 2 063×2 063 matrix
───────────────────────────────────────────────────────────────────────

Progress is printed for every major step and a tqdm‑bar tracks the
O(n²) accumulation.

The implementation follows formula (10) in Boley et al. (2018):

      N(i,j,k) = (m_ij − m_kj − m_ik + m_kk) · π_j

and
      betweenness(j) = Σ_{i≠j,k≠j}  N(i,j,k) / N(j,j,k)

which can be reduced to an **O(n²)** sweep once the pseudoinverse
**M = L_rw⁺** and the column‑sums of M are known (see derivation in
comments).

References
----------
* Ulrik Brandes & Daniel Fleischer. *STACS ‘05*.  Current‑flow
  betweenness centrality.
* Daniel Boley, Alejandro Buendia, Golshan Golnari. *Random Walk
  Laplacian and Network Centrality Measures*. arXiv:1808.02912
"""

import time
from typing import Dict, Any, Hashable

import numpy as np
import networkx as nx
from tqdm import tqdm


def _stationary_distribution(P: np.ndarray,
                             max_iter: int = 10_000,
                             tol: float = 1e-12) -> np.ndarray:
    """Power‑iteration to get the left stationary distribution π of P."""
    n = P.shape[0]
    π = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        π_new = π @ P
        if np.linalg.norm(π_new - π, 1) < tol:
            return π_new
        π = π_new
    return π  # warn: may not have converged


def directed_random_walk_betweenness(
    G: nx.DiGraph,
    assume_strongly_connected: bool = False,
    verbose: bool = True,
) -> Dict[Hashable, float]:
    """Exact random‑walk betweenness for a **directed** graph.

    Parameters
    ----------
    G : nx.DiGraph
        Weighted or unweighted, **strongly‑connected** digraph.
        If dangling nodes exist (out‑degree 0) they are given a uniform
        teleport to keep the chain irreducible.
    assume_strongly_connected : bool, default False
        If *True* skip the connectivity test (saves a few seconds
        on large graphs but may raise if the graph is really not SC).
    verbose : bool, default True
        Print progress and timings.

    Returns
    -------
    dict
        Mapping node → random‑walk betweenness centrality
        (unnormalized).
    """

    t_tot = time.perf_counter()
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}

    # ------------------------------------------------------------------
    # 1) Transition matrix P (row‑stochastic)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/5] Building transition matrix  P  ({n}×{n}) …", flush=True)
    P = np.zeros((n, n), dtype=float)

    for u in G:
        i = index[u]
        succ = list(G.successors(u))
        if succ:
            w_total = sum(G[u][v].get("weight", 1.0) for v in succ)
            for v in succ:
                j = index[v]
                P[i, j] = G[u][v].get("weight", 1.0) / w_total
        else:
            # dangling → uniform teleport
            P[i, :] = 1.0 / n

    # ------------------------------------------------------------------
    # 2) Stationary distribution π  (left eigenvector of P)
    # ------------------------------------------------------------------
    if verbose:
        print("[2/5] Power‑iteration for stationary distribution π …",
              flush=True)
    t0 = time.perf_counter()
    π = _stationary_distribution(P)
    if verbose:
        print(f"      converged in {time.perf_counter() - t0:6.2f}s")

    # ------------------------------------------------------------------
    # 3) Random‑walk Laplacian  L_rw  and its pseudoinverse  M
    # ------------------------------------------------------------------
    if verbose:
        print("[3/5] Forming Laplacian  L_rw  and dense pseudoinverse …",
              flush=True)
    t0 = time.perf_counter()
    I = np.eye(n, dtype=float)
    L_rw = np.diag(π) @ (I - P)            # Π · (I − P)
    M = np.linalg.pinv(L_rw, rcond=1e-12)  # Moore–Penrose pseudoinverse
    if verbose:
        print(f"      SVD done in {time.perf_counter() - t0:6.2f}s")

    # ------------------------------------------------------------------
    # 4)   O(n²) accumulation   (vectorised inner loop)
    # ------------------------------------------------------------------
    if verbose:
        print("[4/5] Accumulating betweenness (vectorised)…", flush=True)
    t0 = time.perf_counter()
    diag_M = np.diag(M)
    col_sum = M.sum(axis=0)
    bet = np.zeros(n, dtype=float)

    for j in tqdm(range(n), unit="node"):
        m_jj = diag_M[j]
        csum_j = col_sum[j]

        m_kj = M[:, j]         # column j
        m_jk = M[j, :]         # row j
        m_kk = diag_M          # broadcast

        denom = m_jj - m_kj - m_jk + m_kk
        # avoid /0 when k=j (value unused anyway)
        denom[j] = np.inf

        S = (csum_j - col_sum           # Σ_i m_ij − Σ_i m_ik
             - m_jj + m_jk              #        - m_jj + m_jk
             + (-n + 1) * m_kj          # (-n+1)·m_kj
             + (n - 1) * m_kk)          # (n−1)·m_kk
        S[j] = 0.0  # exclude k=j

        bet[j] = np.divide(S, denom,
                           out=np.zeros_like(S), where=denom != 0).sum()

    if verbose:
        print(f"      done in {time.perf_counter() - t0:6.2f}s")

    # ------------------------------------------------------------------
    # 5) Wrap‑up
    # ------------------------------------------------------------------
    centrality = {nodes[j]: float(bet[j]) for j in range(n)}
    if verbose:
        print(f"[5/5] All done in {time.perf_counter() - t_tot:6.2f}s")

    return centrality


if __name__ == "__main__":
    import glob, os
    import pandas as pd

    # process all dfz_s_*.parquet files in this directory
    for parquet_path in glob.glob("dfz_s_*.parquet"):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        df.index = df.columns
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())

        # compute betweenness centrality
        res = directed_random_walk_betweenness(G, verbose=True)

        # derive suffix (year or code) from filename
        basename = os.path.splitext(os.path.basename(parquet_path))[0]
        suffix = basename.replace("dfz_s_", "")

        # save results with suffix
        out_file = f"rw_betweenness_{suffix}.npy"
        print(f"Saving results to {out_file}...")
        np.save(out_file, res)
