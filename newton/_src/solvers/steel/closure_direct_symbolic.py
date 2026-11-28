# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side symbolic analysis for the sparse direct closure solve (``closure_direct_factor="sparse"``).

The direct solve factors ``K = H + J^T W J`` every substep. ``H`` (the augmented joint-space mass matrix) is nonzero
only between a DOF and its root-path ancestors, and each loop closure couples only the DOFs of its own support (the
DOFs between its two bodies), so ``K`` is sparse and its pattern never changes. This module computes, once per model:

- a fill-reducing ordering (minimum degree on the pattern of ``K``);
- the column structure of the Cholesky factor ``L`` (``P K P^T = L L^T``), its fundamental supernodes and their
  height levels in the supernodal elimination tree (all supernodes of one level are independent);
- the packed column storage of ``L``: column ``j`` of supernode ``s`` holds the rows ``R_s[j - c0_s:]``;
- for each supernode the list of descendant supernodes that update it (left-looking), with the relative row map;
- the gather tables that assemble ``K`` from ``H`` (ancestor layout) and the closure rows straight into ``L``'s
  storage, the row-wise structure of ``L`` (forward solve) and the rows of ``H`` (refinement products).

Everything here is plain NumPy; the device kernels live in ``closure_direct_sparse.py``.
"""

from __future__ import annotations

import numpy as np


def root_paths(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ancestor layout of a DOF tree: ``ancestors[d, s]`` is the ``s``-th DOF on ``d``'s root path (``s = 0``: ``d``)."""
    n = len(parent)
    chains = []
    for d in range(n):
        c, k = [], d
        while k >= 0:
            c.append(k)
            k = int(parent[k])
        chains.append(c)
    na = np.array([len(c) for c in chains], dtype=np.int32)
    ancestors = np.full((n, int(na.max()) if n else 1), -1, dtype=np.int32)
    for d, c in enumerate(chains):
        ancestors[d, : len(c)] = c
    return ancestors, na


def minimum_degree(adj: list[set[int]]) -> np.ndarray:
    """Deterministic minimum-degree elimination order (ties broken by the lowest index)."""
    n = len(adj)
    g = [set(a) for a in adj]
    alive = set(range(n))
    order = []
    while alive:
        k = min(alive, key=lambda i: (len(g[i]), i))
        order.append(k)
        nb = g[k]
        for a in nb:
            g[a] |= nb
            g[a].discard(a)
            g[a].discard(k)
        alive.discard(k)
    return np.asarray(order, dtype=np.int64)


class SparseCholeskyPlan:
    """Symbolic factorization of ``K = H + sum_k J_k^T W_k J_k`` for one articulation's DOF tree and closures.

    Args:
        parent: Parent DOF of each DOF (-1 for root DOFs), parents before children.
        supports: Per closure, the (local) DOFs its rows may touch.
        order: Optional elimination order (new -> old); minimum degree by default.
    """

    def __init__(self, parent: np.ndarray, supports: list[np.ndarray], order: np.ndarray | None = None):
        parent = np.asarray(parent, dtype=np.int64)
        n = len(parent)
        self.n = n
        self.ancestors, self.na = root_paths(parent)
        NA = self.ancestors.shape[1]
        self.NA = NA
        self.supports = [np.asarray(s, dtype=np.int64) for s in supports]
        self.KJ = max((len(s) for s in self.supports), default=1)

        # Pattern of K (original order).
        adj = [set() for _ in range(n)]
        for d in range(n):
            for s in range(1, int(self.na[d])):
                a = int(self.ancestors[d, s])
                adj[d].add(a)
                adj[a].add(d)
        for sup in self.supports:
            for a in sup.tolist():
                adj[a].update(sup.tolist())
                adj[a].discard(a)
        self.pattern_nnz = sum(len(a) for a in adj) + n
        perm = minimum_degree(adj) if order is None else np.asarray(order, dtype=np.int64)
        iperm = np.empty(n, dtype=np.int64)
        iperm[perm] = np.arange(n)
        self.perm, self.iperm = perm, iperm

        # Column structure of L in the new order (symbolic elimination).
        g = [set() for _ in range(n)]
        for d in range(n):
            new_d = int(iperm[d])
            g[new_d] = {int(iperm[a]) for a in adj[d]}
        cols = []
        for k in range(n):
            nb = sorted(x for x in g[k] if x > k)
            cols.append(nb)
            nbs = set(nb)
            for a in nb:
                g[a] |= nbs
                g[a].discard(a)
        etree = np.array([c[0] if c else -1 for c in cols], dtype=np.int64)
        nchild = np.bincount(etree[etree >= 0], minlength=n)

        # Fundamental supernodes.
        sn_c0 = [0]
        for j in range(1, n):
            p = j - 1
            if not (etree[p] == j and len(cols[p]) == len(cols[j]) + 1 and nchild[j] == 1):
                sn_c0.append(j)
        sn_c0.append(n)
        S = len(sn_c0) - 1
        sn_of = np.empty(n, dtype=np.int64)
        R = []
        for s in range(S):
            c0, c1 = sn_c0[s], sn_c0[s + 1]
            sn_of[c0:c1] = s
            R.append(list(range(c0, c1)) + cols[c1 - 1])
        sn_w = np.diff(sn_c0)
        sn_q = np.array([len(r) for r in R], dtype=np.int64)
        sparent = np.array([sn_of[etree[sn_c0[s + 1] - 1]] if etree[sn_c0[s + 1] - 1] >= 0 else -1 for s in range(S)])

        # Height levels: leaves at 0; a supernode runs after all of its descendants.
        height = np.zeros(S, dtype=np.int64)
        for s in range(S):  # children have lower indices than parents (postorder of the elimination tree)
            if sparent[s] >= 0:
                height[sparent[s]] = max(height[sparent[s]], height[s] + 1)
        nlev = int(height.max()) + 1 if S else 0
        lvl_sn = np.argsort(height, kind="stable")
        lvl_start = np.searchsorted(height[lvl_sn], np.arange(nlev + 1))

        # Packed column storage.
        col_off = np.zeros(n + 1, dtype=np.int64)
        rowidx = []
        for s in range(S):
            c0, c1 = sn_c0[s], sn_c0[s + 1]
            for j in range(c0, c1):
                rows = R[s][j - c0 :]
                col_off[j + 1] = col_off[j] + len(rows)
                rowidx.extend(rows)
        self.nnz = int(col_off[-1])
        rowidx = np.asarray(rowidx, dtype=np.int64)
        pos_in = [{r: p for p, r in enumerate(R[s])} for s in range(S)]

        def entry(i: int, j: int) -> int:
            s = int(sn_of[j])
            return int(col_off[j] + pos_in[s][i] - (j - sn_c0[s]))

        self.entry = entry

        # Left-looking update pairs (target s <- descendant d), grouped by target, ordered by d.
        pairs = [[] for _ in range(S)]
        rel = []
        for d in range(S):
            w, q = int(sn_w[d]), int(sn_q[d])
            p = w
            while p < q:
                s = int(sn_of[R[d][p]])
                p1 = p
                while p1 < q and sn_of[R[d][p1]] == s:
                    p1 += 1
                base = len(rel)
                rel.extend(pos_in[s][R[d][t]] for t in range(p, q))
                pairs[s].append((d, p, p1, base))
                p = p1
        upd_start = np.zeros(S + 1, dtype=np.int64)
        pr = []
        for s in range(S):
            pr.extend(pairs[s])
            upd_start[s + 1] = len(pr)
        pr = np.asarray(pr, dtype=np.int64).reshape(-1, 4)

        # Row-wise structure of L (off-diagonal entries of each row, by column): forward solve.
        rows_of = [[] for _ in range(n)]
        for j in range(n):
            for t in range(int(col_off[j]) + 1, int(col_off[j + 1])):
                rows_of[int(rowidx[t])].append((j, t))
        row_ptr = np.zeros(n + 1, dtype=np.int64)
        row_col, row_ent = [], []
        for i in range(n):
            for j, t in rows_of[i]:
                row_col.append(j)
                row_ent.append(t)
            row_ptr[i + 1] = len(row_col)

        # Assembly: each storage entry gathers its H entry (ancestor layout) and closure pair products.
        hidx = np.full(self.nnz, -1, dtype=np.int64)
        for d in range(n):
            for s in range(int(self.na[d])):
                a = int(self.ancestors[d, s])
                i, j = int(iperm[d]), int(iperm[a])
                hidx[entry(max(i, j), min(i, j))] = d * NA + s
        contrib = [[] for _ in range(self.nnz)]
        for k, sup in enumerate(self.supports):
            for a in range(len(sup)):
                for b in range(a, len(sup)):
                    i, j = int(iperm[sup[a]]), int(iperm[sup[b]])
                    contrib[entry(max(i, j), min(i, j))].append((k << 12) | (a << 6) | b)
        asm_ptr = np.zeros(self.nnz + 1, dtype=np.int64)
        for e in range(self.nnz):
            asm_ptr[e + 1] = asm_ptr[e] + len(contrib[e])
        asm_list = np.asarray([c for cs in contrib for c in cs], dtype=np.int64)
        # g0 = -J^T W r0 gathers per (new) DOF: (closure, support position).
        g_lists = [[] for _ in range(n)]
        for k, sup in enumerate(self.supports):
            for a, dd in enumerate(sup.tolist()):
                g_lists[int(iperm[dd])].append((k << 6) | a)
        g_ptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            g_ptr[i + 1] = g_ptr[i] + len(g_lists[i])
        g_list = np.asarray([c for cs in g_lists for c in cs], dtype=np.int64)
        # Rows of H in the new order (refinement products): (ancestor-layout index, new column).
        h_rows = [[] for _ in range(n)]
        for d in range(n):
            for s in range(int(self.na[d])):
                a = int(self.ancestors[d, s])
                h_rows[int(iperm[d])].append((d * NA + s, int(iperm[a])))
                if s > 0:
                    h_rows[int(iperm[a])].append((d * NA + s, int(iperm[d])))
        h_ptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            h_ptr[i + 1] = h_ptr[i] + len(h_rows[i])
        h_list = np.asarray([x for xs in h_rows for x in xs], dtype=np.int64).reshape(-1, 2)

        # Per target column: the update items (descendant d, its row position pj of the column), ordered by d.
        # Record: (c0_d, w_d | q_d << 16, rel base - p0, pj).
        col_items = [[] for _ in range(n)]
        for s in range(S):
            for d, p0, p1, base in pairs[s]:
                for pj in range(p0, p1):
                    col_items[int(R[d][pj])].append((sn_c0[d], int(sn_w[d]) | (int(sn_q[d]) << 16), base - p0, pj))
        ct_ptr = np.zeros(n + 1, dtype=np.int64)
        for j in range(n):
            ct_ptr[j + 1] = ct_ptr[j] + len(col_items[j])
        ct_rec = np.asarray([r for rs in col_items for r in rs], dtype=np.int64).reshape(-1, 4)
        # First row entry of each row whose column lies in the row's own supernode (forward-solve gather split).
        row_split = np.zeros(n, dtype=np.int64)
        for i in range(n):
            c0i = sn_c0[sn_of[i]]
            k = int(row_ptr[i])
            while k < row_ptr[i + 1] and row_col[k] < c0i:
                k += 1
            row_split[i] = k
        self.ct_ptr, self.ct_rec, self.row_split = ct_ptr, ct_rec, row_split
        # Source columns of each supernode's updates, flattened over its pairs: (pair index << 8) | column of d.
        sc_ptr = np.zeros(S + 1, dtype=np.int64)
        sc_list = []
        for s in range(S):
            for k in range(int(upd_start[s]), int(upd_start[s + 1])):
                d = int(pr[k][0])
                for c in range(int(sn_w[d])):
                    sc_list.append((k << 8) | c)
            sc_ptr[s + 1] = len(sc_list)
        if np.any(sn_w > 255):
            pass  # only descendants are expanded; checked by the device setup
        self.sc_ptr, self.sc_list = sc_ptr, np.asarray(sc_list, dtype=np.int64)
        # The same columns as records: (storage offset of the column minus its position, p0, q_d, rel base - p0).
        sc_rec = []
        for code in sc_list:
            k, c = code >> 8, code & 255
            d, p0, _, base = (int(v) for v in pr[k])
            sc_rec.append((int(col_off[sn_c0[d] + c]) - c, p0, int(sn_q[d]), base - p0))
        self.sc_rec = np.asarray(sc_rec, dtype=np.int64).reshape(-1, 4)

        self.S, self.nlev = S, nlev
        self.sn_c0 = np.asarray(sn_c0, dtype=np.int64)
        self.sn_w, self.sn_q, self.sparent, self.height = sn_w, sn_q, sparent, height
        self.lvl_sn, self.lvl_start = lvl_sn, lvl_start
        self.col_off, self.rowidx = col_off, rowidx
        self.upd_start, self.pairs, self.rel = upd_start, pr, np.asarray(rel, dtype=np.int64)
        self.row_ptr, self.row_col, self.row_ent = row_ptr, np.asarray(row_col), np.asarray(row_ent)
        self.hidx, self.asm_ptr, self.asm_list = hidx, asm_ptr, asm_list
        self.g_ptr, self.g_list = g_ptr, g_list
        self.h_ptr, self.h_list = h_ptr, h_list
        self.flops = float(sum(len(c) ** 2 for c in cols) / 2 + n)

    # ---------------------------------------------------------------------------------------------------------
    # Float64 host reference of the device algorithm (tests and diagnostics)
    # ---------------------------------------------------------------------------------------------------------

    def assemble(self, Ha: np.ndarray, Jt: list[np.ndarray]) -> np.ndarray:
        """Packed ``L`` storage holding ``P K P^T`` (lower), from ``Ha[d, s]`` and the scaled rows ``Jt[k][r, a]``."""
        Lv = np.zeros(self.nnz)
        Haf = Ha.reshape(-1)
        for e in range(self.nnz):
            v = Haf[self.hidx[e]] if self.hidx[e] >= 0 else 0.0
            for c in self.asm_list[self.asm_ptr[e] : self.asm_ptr[e + 1]].tolist():
                k, a, b = c >> 12, (c >> 6) & 63, c & 63
                v += float(Jt[k][:, a] @ Jt[k][:, b])
            Lv[e] = v
        return Lv

    def factor(self, Lv: np.ndarray) -> np.ndarray:
        """Left-looking supernodal Cholesky in the packed storage, level by level (as the device kernel)."""
        Lv = Lv.copy()
        co, ri = self.col_off, self.rowidx
        for lev in range(self.nlev):
            for s in self.lvl_sn[self.lvl_start[lev] : self.lvl_start[lev + 1]].tolist():
                c0, w, q = int(self.sn_c0[s]), int(self.sn_w[s]), int(self.sn_q[s])
                for d, p0, p1, base in self.pairs[self.upd_start[s] : self.upd_start[s + 1]].tolist():
                    c0d, wd, qd = int(self.sn_c0[d]), int(self.sn_w[d]), int(self.sn_q[d])
                    for pj in range(p0, p1):
                        t = int(ri[co[c0d] + pj]) - c0
                        for pi in range(pj, qd):
                            acc = sum(Lv[co[c0d + c] + pi - c] * Lv[co[c0d + c] + pj - c] for c in range(wd))
                            Lv[co[c0 + t] + self.rel[base + pi - p0] - t] -= acc
                for t in range(w):
                    o = co[c0 + t]
                    piv = np.sqrt(Lv[o])
                    Lv[o] = piv
                    Lv[o + 1 : o + q - t] /= piv
                    for t2 in range(t + 1, w):
                        o2 = co[c0 + t2]
                        Lv[o2 : o2 + q - t2] -= Lv[o + t2 - t : o + q - t] * Lv[o + t2 - t]
        return Lv

    def dense_L(self, Lv: np.ndarray) -> np.ndarray:
        L = np.zeros((self.n, self.n))
        for j in range(self.n):
            for t in range(int(self.col_off[j]), int(self.col_off[j + 1])):
                L[self.rowidx[t], j] = Lv[t]
        return L
