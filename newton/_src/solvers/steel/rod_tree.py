# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulated-body passes over rod trees, six lanes per tree.

A rod tree's joints carry up to six DOFs each, so every per-joint term of the articulated-body
algorithm is a 6x6 block. Six lanes own one row each of every block: products are row operations
over a shared copy of the other operand, and the joint block ``D`` is inverted by a Gauss-Jordan sweep
whose pivot rows are broadcast with shuffles. A joint whose parent is the previous joint's child (a
chain) hands its contribution to the parent in registers; other parents (branches) gather theirs in a
per-body accumulator. Five trees share a warp.

Values match the generic factorization: ``U = I_a S``, ``D = S^T U`` plus armature, implicit drive
stiffness and the joint's material block, and the child's articulated inertia ``I_a - U D^-1 U^T``
shifted to the parent's center of mass. With ``with_tau`` the same launch solves ``H qdd = tau`` through
the factorization it just built.
"""

import functools

import numpy as np
import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF

# Trees per warp and lanes per tree.
TREES_PER_WARP = 1
BLOCK_DIM = 64
# Floats per link of the chain passes' compact record: B, W, S, Q (6x6 row-major each) and e (padded to 4).
CHAIN_REC = 148
# Longest chain whose records the chain propagation pass stages in shared memory (longer ones read them from
# global memory in the same layout), and its block size: one warp per block.
CHAIN_STAGE_MAX = 64
CHAIN_BLOCK_DIM = 32

_PRELUDE = (
    "\n    #define TREES_PER_WARP "
    + str(TREES_PER_WARP)
    + "\n    #define CHAIN_REC "
    + str(CHAIN_REC)
    + "\n"
    + r"""
    const int lane = threadIdx.x & 31;
    const int grp = lane / 6;
    const int r = lane - grp * 6;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int tree = warp_global * TREES_PER_WARP + grp;
    const int src0 = grp * 6;
    // Trees in one warp have different lengths, so each six-lane group shuffles and syncs on its own.
    const unsigned GMASK = 0x3Fu << src0;
    const int wib = threadIdx.x >> 5;
    __shared__ float shm[2][TREES_PER_WARP][4][36];
    auto bcast = [&](float val, int k) { return __shfl_sync(GMASK, val, src0 + k); };
    auto gj = [&](float* m, float* inv) {
        #pragma unroll
        for (int k = 0; k < 6; ++k) inv[k] = (k == r) ? 1.0f : 0.0f;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            float pm[6], pi[6];
            #pragma unroll
            for (int k = 0; k < 6; ++k) { pm[k] = bcast(m[k], c); pi[k] = bcast(inv[k], c); }
            const float s = 1.0f / pm[c];
            const float f = (r == c) ? 0.0f : m[c];
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                pm[k] *= s; pi[k] *= s;
                if (r == c) { m[k] = pm[k]; inv[k] = pi[k]; }
                else { m[k] -= f * pm[k]; inv[k] -= f * pi[k]; }
            }
        }
    };
"""
)

# The factor runs every lane of its warp: lanes past the sixth repeat lanes 0-5 (same tree, same values), so every
# shuffle and warp sync takes the full mask. A partial mask compiles to a convergence check (MATCH.ANY and a
# divergent branch) on every shuffle, which dominated the factor's serial walk. Only lanes 0-5 (``lead``) update
# shared accumulators; identical stores from the repeating lanes are harmless.
_PRELUDE_FULL = (
    _PRELUDE.replace("const int grp = lane / 6;", "const int grp = 0;")
    .replace("const int r = lane - grp * 6;", "const int r = lane % 6;\n    const bool lead = lane < 6;")
    .replace("const unsigned GMASK = 0x3Fu << src0;", "const unsigned GMASK = 0xffffffffu;")
)

_FACTOR = (
    r"""
#if defined(__CUDA_ARCH__)
"""
    + _PRELUDE_FULL
    + r"""
    if (grp >= TREES_PER_WARP || tree >= ntrees) return;
    float* X = shm[wib][grp][0];
    float* Y = shm[wib][grp][1];
    float* Z = shm[wib][grp][2];
    float* V = shm[wib][grp][3];
    const int art = group_to_art.data[tree];
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int joint_start = articulation_joint_range.data[2 * art];
    const int joint_end = articulation_joint_range.data[2 * art + 1];
    if (joint_end <= joint_start) return;
    const int aug_count = aug_row_counts.data[art];
    const int* meta = joint_meta.data;
    const float* SF = propagation_joint_S_flat.data;
    const float* CR = propagation_body_com_rel.data;
    const int rb = r >= 3 ? 3 : 0;
    const int rr = r - rb;

    // A joint's inputs as lane r needs them, loaded one joint ahead of their use.
    struct In { int child, parent, gd, dc, flags, jid; wp::quat_t<wp::float32> q; float L[3][6]; float iw[6]; float s[6]; float m[6]; float e[3]; float diag; float sinv[6]; };
    const bool pre = use_prelude != 0;
    auto load = [&](int k, In& o) {
        o.child = meta[4 * k + 0];
        o.parent = meta[4 * k + 1];
        o.gd = meta[4 * k + 2];
        o.flags = meta[4 * k + 3];
        o.dc = o.flags & 7;
        o.jid = o.flags >> 8;
        if (pre) {
            #pragma unroll
            for (int c = 0; c < 6; ++c) { o.iw[c] = IW.data[k * 36 + r * 6 + c]; o.sinv[c] = SINV.data[k * 36 + r * 6 + c]; }
        } else {
            o.q = body_q_com.data[o.child].q;
            const wp::mat_t<6, 6, wp::float32>& L = body_I_m.data[o.child];
            #pragma unroll
            for (int a = 0; a < 3; ++a)
                #pragma unroll
                for (int c = 0; c < 6; ++c) o.L[a][c] = L.data[rb + a][c];
        }
        #pragma unroll
        for (int a = 0; a < 6; ++a) o.s[a] = (a < o.dc) ? SF[(o.gd + a) * 6 + r] : 0.0f;
        #pragma unroll
        for (int b = 0; b < 6; ++b) o.m[b] = (has_joint_material && r < o.dc && b < o.dc) ? joint_material.data[o.jid].data[r][b] : 0.0f;
        #pragma unroll
        for (int c = 0; c < 3; ++c) o.e[c] = (o.parent >= 0) ? CR[o.child * 3 + c] - CR[o.parent * 3 + c] : 0.0f;
        o.diag = pre ? KD.data[k * 6 + r] : ((r < o.dc) ? joint_armature.data[o.gd + r] : 0.0f);
    };

    // Leaf to root.
    float acc[6] = {0, 0, 0, 0, 0, 0};
    float acc_p = 0.0f;
    In cur, nxt;
    load(joint_end - 1, cur);
    for (int k = joint_end - 1; k >= joint_start; --k) {
        if (k - 1 >= joint_start) load(k - 1, nxt);
        const int dc = cur.dc;
        // Row r of the child's articulated inertia: its own inertia in world axes, the chain carry and the
        // branch accumulator.
        float ia[6];
        if (pre) {
            #pragma unroll
            for (int col = 0; col < 6; ++col) ia[col] = cur.iw[col] + ((cur.flags & 8) ? acc[col] : 0.0f);
        } else {
            const wp::mat_t<3, 3, wp::float32> R = wp::quat_to_matrix(wp::quat_inverse(cur.q));
            // Column rr of R, selected without a runtime index (which would spill R to local memory).
            float rc[3];
            #pragma unroll
            for (int a = 0; a < 3; ++a) rc[a] = (rr == 0) ? R.data[a][0] : ((rr == 1) ? R.data[a][1] : R.data[a][2]);
            #pragma unroll
            for (int col = 0; col < 6; ++col) {
                const int cb = col >= 3 ? 3 : 0;
                const int cc = col - cb;
                float v = 0.0f;
                #pragma unroll
                for (int a = 0; a < 3; ++a)
                    #pragma unroll
                    for (int b = 0; b < 3; ++b)
                        v += rc[a] * cur.L[a][cb + b] * R.data[b][cc];
                ia[col] = v + ((cur.flags & 8) ? acc[col] : 0.0f);
            }
        }
        float pa = (cur.flags & 8) ? acc_p : 0.0f;
        if (cur.flags & 32) {
            #pragma unroll
            for (int col = 0; col < 6; ++col) ia[col] += branch.data[cur.child * 36 + r * 6 + col];
            pa += branch_p.data[cur.child * 6 + r];
            // Every lane has read the accumulator (lanes past the sixth repeat lanes 0-5) before it is cleared.
            __syncwarp(GMASK);
            if (lead) {
                #pragma unroll
                for (int col = 0; col < 6; ++col) branch.data[cur.child * 36 + r * 6 + col] = 0.0f;
                branch_p.data[cur.child * 6 + r] = 0.0f;
            }
        }

        // U = I_a S (row r, column a), staged for D = S^T U.
        #pragma unroll
        for (int a = 0; a < 6; ++a) X[r * 6 + a] = cur.s[a];  // X[j][a] = S[a][j]
        __syncwarp(GMASK);
        float u_row[6];
        #pragma unroll
        for (int a = 0; a < 6; ++a) {
            float v = 0.0f;
            #pragma unroll
            for (int j = 0; j < 6; ++j) v += ia[j] * X[j * 6 + a];
            u_row[a] = v;
        }
        #pragma unroll
        for (int a = 0; a < 6; ++a) Y[r * 6 + a] = u_row[a];  // Y[j][a] = U[j][a]
        __syncwarp(GMASK);
        // Row r of D (r as the joint DOF), padded with identity rows past the joint's DOFs.
        float d_row[6], dinv[6], ihat[6];
        #pragma unroll
        for (int b = 0; b < 6; ++b) {
            float v = 0.0f;
            #pragma unroll
            for (int j = 0; j < 6; ++j) v += X[j * 6 + r] * Y[j * 6 + b];
            ihat[b] = v;  // S^T I_a S
            d_row[b] = (r < dc && b < dc) ? v + cur.m[b] : ((b == r && r >= dc) ? 1.0f : 0.0f);
        }
        if (r < dc) {
            float diag = cur.diag;
            for (int aug_i = 0; aug_i < (pre ? 0 : aug_count); ++aug_i) {
                const int row_index = art * max_dofs + aug_i;
                if (aug_row_dof_index.data[row_index] == cur.gd + r) {
                    const float K = aug_row_K.data[row_index];
                    if (K > 0.0f) diag += K;
                }
            }
            #pragma unroll
            for (int b = 0; b < 6; ++b) if (b == r) d_row[b] += diag;
        }
        gj(d_row, dinv);
        if (r < dc) {
            #pragma unroll
            for (int b = 0; b < 6; ++b) propagation_tree_D_inv.data[cur.jid * 36 + r * 6 + b] = (b < dc) ? dinv[b] : 0.0f;
        }
        #pragma unroll
        for (int a = 0; a < 6; ++a) if (a < dc) propagation_tree_U.data[(cur.gd + a) * 6 + r] = u_row[a];
        __syncwarp(GMASK);
        #pragma unroll
        for (int b = 0; b < 6; ++b) Z[r * 6 + b] = (r < dc) ? dinv[b] : 0.0f;  // Z[a][b] = D^-1[a][b]
        __syncwarp(GMASK);

        // Bias wrenches of tau: u = tau - S^T p, passed = p + U D^-1 u.
        float passed_p = pa;
        if (with_tau) {
            float pa_all[6];
            #pragma unroll
            for (int j = 0; j < 6; ++j) pa_all[j] = bcast(pa, j);
            float ut = 0.0f;
            if (r < dc) {
                ut = joint_tau.data[cur.gd + r];
                #pragma unroll
                for (int j = 0; j < 6; ++j) ut -= X[j * 6 + r] * pa_all[j];
                propagation_tree_u.data[cur.gd + r] = ut;
            }
            float c = 0.0f;
            #pragma unroll
            for (int b = 0; b < 6; ++b) c += Z[r * 6 + b] * bcast(ut, b);
            #pragma unroll
            for (int a = 0; a < 6; ++a) passed_p += u_row[a] * bcast(c, a);
        }

        if (!pre)
        // The propagation pass's records. B = T (I - U D^-1 S^T) carries a child's wrench to its parent (T moves
        // it to the parent's center of mass); its transpose (I - S D^-1 U^T) X carries the parent's twist change
        // to the child. W = -D^-1 S^T gives the joint's w = D^-1 u from the wrench and Q = -D^-1 U^T X its
        // acceleration from the parent's twist change. Lane r's backward record is [B row r, W row r]; its
        // forward record is [B column r, Q row r, S row r, e, 0]: contiguous, so the walk loads them as float4s.
        {
            float wv[6];
            #pragma unroll
            for (int b = 0; b < 6; ++b) {
                float v = 0.0f;
                #pragma unroll
                for (int a = 0; a < 6; ++a) v += u_row[a] * Z[a * 6 + b];
                wv[b] = v;
            }
            const float ex = cur.e[0], ey = cur.e[1], ez = cur.e[2];
            float brow[6];
            #pragma unroll
            for (int col = 0; col < 6; ++col) {
                float p = 0.0f;
                #pragma unroll
                for (int b = 0; b < 6; ++b) p += wv[b] * X[col * 6 + b];
                const float bv = ((col == r) ? 1.0f : 0.0f) - p;
                const float b0 = bcast(bv, 0), b1 = bcast(bv, 1), b2 = bcast(bv, 2);
                float v = bv;
                if (r == 3) v += ey * b2 - ez * b1;
                if (r == 4) v += ez * b0 - ex * b2;
                if (r == 5) v += ex * b1 - ey * b0;
                brow[col] = v;
            }
            float wrow[6], qrow[6];
            {
                float g[6];
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    float v = 0.0f, t = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 6; ++b) {
                        v += Z[r * 6 + b] * Y[j * 6 + b];
                        t += Z[r * 6 + b] * X[j * 6 + b];
                    }
                    g[j] = v;
                    wrow[j] = -t;
                }
                qrow[0] = -g[0];
                qrow[1] = -g[1];
                qrow[2] = -g[2];
                qrow[3] = -(g[3] - ez * g[1] + ey * g[2]);
                qrow[4] = -(g[4] + ez * g[0] - ex * g[2]);
                qrow[5] = -(g[5] - ey * g[0] + ex * g[1]);
            }
            #pragma unroll
            for (int col = 0; col < 6; ++col) V[r * 6 + col] = brow[col];
            __syncwarp(GMASK);
            float rec[24];
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                rec[j] = V[j * 6 + r];
                rec[6 + j] = qrow[j];
                rec[12 + j] = cur.s[j];
            }
            rec[18] = ex; rec[19] = ey; rec[20] = ez; rec[21] = 0.0f; rec[22] = 0.0f; rec[23] = 0.0f;
            float4* fr = reinterpret_cast<float4*>(propagation_tree_FR.data + k * 144 + r * 24);
            #pragma unroll
            for (int q = 0; q < 6; ++q) fr[q] = make_float4(rec[4 * q], rec[4 * q + 1], rec[4 * q + 2], rec[4 * q + 3]);
            float4* br = reinterpret_cast<float4*>(propagation_tree_BR.data + k * 72 + r * 12);
            br[0] = make_float4(brow[0], brow[1], brow[2], brow[3]);
            br[1] = make_float4(brow[4], brow[5], wrow[0], wrow[1]);
            br[2] = make_float4(wrow[2], wrow[3], wrow[4], wrow[5]);
            // The chain passes' compact record (see CHAIN_REC): B, W, S and Q row-major, then e.
            float* cr = propagation_tree_CR.data + k * CHAIN_REC;
            float2* c2 = reinterpret_cast<float2*>(cr + r * 6);
            c2[0] = make_float2(brow[0], brow[1]); c2[1] = make_float2(brow[2], brow[3]); c2[2] = make_float2(brow[4], brow[5]);
            c2 = reinterpret_cast<float2*>(cr + 36 + r * 6);
            c2[0] = make_float2(wrow[0], wrow[1]); c2[1] = make_float2(wrow[2], wrow[3]); c2[2] = make_float2(wrow[4], wrow[5]);
            c2 = reinterpret_cast<float2*>(cr + 72 + r * 6);
            c2[0] = make_float2(cur.s[0], cur.s[1]); c2[1] = make_float2(cur.s[2], cur.s[3]); c2[2] = make_float2(cur.s[4], cur.s[5]);
            c2 = reinterpret_cast<float2*>(cr + 108 + r * 6);
            c2[0] = make_float2(qrow[0], qrow[1]); c2[1] = make_float2(qrow[2], qrow[3]); c2[2] = make_float2(qrow[4], qrow[5]);
            if (r == 0) *reinterpret_cast<float4*>(cr + 144) = make_float4(ex, ey, ez, 0.0f);
            __syncwarp(GMASK);
        }

        if (cur.parent >= 0) {
            // Reduced inertia passed to the parent. With a full-rank joint it is U (D^-1 K I^-1) U^T, with
            // I = S^T I_a S and K = D - I the joint's own terms (material, armature, drives): no subtraction,
            // so a soft joint's near-cancelling I_a - U D^-1 U^T cannot lose the parent's inertia in single
            // precision, and every inverse is of a positive-definite block. A joint with fewer DOFs keeps the
            // subtracting form.
            float red[6];
            if (dc == 6) {
                float krow[6];
                #pragma unroll
                for (int b = 0; b < 6; ++b) krow[b] = cur.m[b];
                {
                    float diag = cur.diag;
                    for (int aug_i = 0; aug_i < (pre ? 0 : aug_count); ++aug_i) {
                        const int row_index = art * max_dofs + aug_i;
                        if (aug_row_dof_index.data[row_index] == cur.gd + r) {
                            const float K = aug_row_K.data[row_index];
                            if (K > 0.0f) diag += K;
                        }
                    }
                    #pragma unroll
                    for (int b = 0; b < 6; ++b) if (b == r) krow[b] += diag;
                }
                // G = D^-1 K (DOF rows).
                __syncwarp(GMASK);
                #pragma unroll
                for (int b = 0; b < 6; ++b) X[r * 6 + b] = krow[b];
                __syncwarp(GMASK);
                float g[6];
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 6; ++b) v += Z[r * 6 + b] * X[b * 6 + c];
                    g[c] = v;
                }
                if (pre) {
                    // Red = U (G S^-1): rows of G S^-1, then U times them.
                    __syncwarp(GMASK);
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) X[r * 6 + j] = cur.sinv[j];
                    __syncwarp(GMASK);
                    float mm[6];
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) {
                        float v = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < 6; ++c) v += g[c] * X[c * 6 + j];
                        mm[j] = v;
                    }
                    __syncwarp(GMASK);
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) X[r * 6 + j] = mm[j];
                    __syncwarp(GMASK);
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) {
                        float v = 0.0f;
                        #pragma unroll
                        for (int a = 0; a < 6; ++a) v += u_row[a] * X[a * 6 + j];
                        red[j] = v;
                    }
                } else {
                float iinv[6];
                gj(ihat, iinv);
                // M = G I^-1 (DOF rows).
                __syncwarp(GMASK);
                #pragma unroll
                for (int j = 0; j < 6; ++j) X[r * 6 + j] = iinv[j];
                __syncwarp(GMASK);
                float mm[6];
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) v += g[c] * X[c * 6 + j];
                    mm[j] = v;
                }
                // W = U M (spatial row r), then R = W U^T and its symmetric part.
                __syncwarp(GMASK);
                #pragma unroll
                for (int j = 0; j < 6; ++j) X[r * 6 + j] = mm[j];
                __syncwarp(GMASK);
                float wrow[6];
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int a = 0; a < 6; ++a) v += u_row[a] * X[a * 6 + j];
                    wrow[j] = v;
                }
                #pragma unroll
                for (int col = 0; col < 6; ++col) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 6; ++j) v += wrow[j] * Y[col * 6 + j];
                    red[col] = v;
                }
                }
                __syncwarp(GMASK);
                #pragma unroll
                for (int j = 0; j < 6; ++j) X[r * 6 + j] = red[j];
                __syncwarp(GMASK);
                #pragma unroll
                for (int j = 0; j < 6; ++j) red[j] = 0.5f * (red[j] + X[j * 6 + r]);
            } else {
                float w[6];
                #pragma unroll
                for (int b = 0; b < 6; ++b) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int a = 0; a < 6; ++a) v += u_row[a] * Z[a * 6 + b];
                    w[b] = v;
                }
                #pragma unroll
                for (int col = 0; col < 6; ++col) {
                    float v = ia[col];
                    #pragma unroll
                    for (int b = 0; b < 6; ++b) v -= w[b] * Y[col * 6 + b];
                    red[col] = v;
                }
            }
            const float ex = cur.e[0], ey = cur.e[1], ez = cur.e[2];
            // E = skew(e); Q = R Tv with Tv = [[I, -E], [0, I]].
            float qrow[6];
            qrow[0] = red[0]; qrow[1] = red[1]; qrow[2] = red[2];
            qrow[3] = red[3] - (red[1] * ez - red[2] * ey);
            qrow[4] = red[4] - (-red[0] * ez + red[2] * ex);
            qrow[5] = red[5] - (red[0] * ey - red[1] * ex);
            __syncwarp(GMASK);
            #pragma unroll
            for (int col = 0; col < 6; ++col) X[r * 6 + col] = qrow[col];
            __syncwarp(GMASK);
            // P = [[I, 0], [E, I]] Q: rows 3..5 add E times rows 0..2.
            float prow[6];
            #pragma unroll
            for (int col = 0; col < 6; ++col) {
                float v = qrow[col];
                if (r == 3) v += -ez * X[1 * 6 + col] + ey * X[2 * 6 + col];
                if (r == 4) v += ez * X[0 * 6 + col] - ex * X[2 * 6 + col];
                if (r == 5) v += -ey * X[0 * 6 + col] + ex * X[1 * 6 + col];
                prow[col] = v;
            }
            // Wrench shift: torque += e x force.
            const float f0 = bcast(passed_p, 0), f1 = bcast(passed_p, 1), f2 = bcast(passed_p, 2);
            float pp = passed_p;
            if (r == 3) pp += ey * f2 - ez * f1;
            if (r == 4) pp += ez * f0 - ex * f2;
            if (r == 5) pp += ex * f1 - ey * f0;
            if (cur.flags & 16) {
                #pragma unroll
                for (int col = 0; col < 6; ++col) acc[col] = prow[col];
                acc_p = pp;
            } else if (lead) {
                #pragma unroll
                for (int col = 0; col < 6; ++col) branch.data[cur.parent * 36 + r * 6 + col] += prow[col];
                branch_p.data[cur.parent * 6 + r] += pp;
            }
        }
        __syncwarp(GMASK);
        cur = nxt;
    }

    if (!with_tau) return;
    // Root to leaf: qdd = D^-1 (u - U^T a_parent) and the link accelerations.
    struct Fw { int child, parent, gd, dc, flags; float ur[6], d[6], sc[6], e[3], u; };
    auto loadf = [&](int k, Fw& o) {
        o.child = meta[4 * k + 0];
        o.parent = meta[4 * k + 1];
        o.gd = meta[4 * k + 2];
        o.flags = meta[4 * k + 3];
        o.dc = o.flags & 7;
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            o.ur[j] = (r < o.dc) ? propagation_tree_U.data[(o.gd + r) * 6 + j] : 0.0f;
            o.d[j] = (r < o.dc && j < o.dc) ? propagation_tree_D_inv.data[(o.flags >> 8) * 36 + r * 6 + j] : 0.0f;
            o.sc[j] = (j < o.dc) ? SF[(o.gd + j) * 6 + r] : 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < 3; ++c) o.e[c] = (o.parent >= 0) ? CR[o.child * 3 + c] - CR[o.parent * 3 + c] : 0.0f;
        o.u = (r < o.dc) ? propagation_tree_u.data[o.gd + r] : 0.0f;
    };
    __syncwarp(GMASK);
    float carry = 0.0f;
    Fw fc, fn;
    loadf(joint_start, fc);
    for (int k = joint_start; k < joint_end; ++k) {
        if (k + 1 < joint_end) loadf(k + 1, fn);
        float at = 0.0f;
        if (fc.parent >= 0) {
            at = (fc.flags & 16) ? carry : body_accel.data[fc.parent * 6 + r];
            const float w0 = bcast(at, 3), w1 = bcast(at, 4), w2 = bcast(at, 5);
            // lin += omega x e
            if (r == 0) at += w1 * fc.e[2] - w2 * fc.e[1];
            if (r == 1) at += w2 * fc.e[0] - w0 * fc.e[2];
            if (r == 2) at += w0 * fc.e[1] - w1 * fc.e[0];
        }
        float at_all[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) at_all[j] = bcast(at, j);
        float t = 0.0f;
        #pragma unroll
        for (int j = 0; j < 6; ++j) t += fc.ur[j] * at_all[j];
        const float term = fc.u - t;
        float qdd = 0.0f;
        #pragma unroll
        for (int b = 0; b < 6; ++b) qdd += fc.d[b] * bcast(term, b);
        if (r < fc.dc) joint_qdd.data[fc.gd + r] = qdd;
        float a_child = at;
        #pragma unroll
        for (int a = 0; a < 6; ++a) a_child += fc.sc[a] * bcast(qdd, a);
        if (fc.flags & 64) body_accel.data[fc.child * 6 + r] = a_child;
        carry = a_child;
        fc = fn;
    }
#endif
"""
)


# Packed factor: five trees per warp when every tree of the group walks the same joint pattern (replicated worlds,
# identical cables) and none is skipped, so all lanes stay convergent and keep full-mask shuffles. Lanes 30-31
# repeat group 4; groups past the last tree repeat it. Only lanes of real trees (``lead``) touch accumulators.
PACKED_TREES = 5
_FACTOR_PACKED = (
    _FACTOR.replace("const int grp = 0;", "const int grp = min(lane / 6, PACKED_TREES - 1);")
    .replace(
        "const int tree = warp_global * TREES_PER_WARP + grp;",
        "const int tree_raw = warp_global * PACKED_TREES + grp;\n"
        "    const int tree = (warp_global * PACKED_TREES < ntrees) ? min(tree_raw, ntrees - 1) : tree_raw;\n"
        "    const bool lead = lane < 30 && tree_raw < ntrees;",
    )
    .replace("const bool lead = lane < 6;", "")
    .replace("__shared__ float shm[2][TREES_PER_WARP][4][36];", "__shared__ float shm[2][PACKED_TREES][4][36];")
    .replace("if (grp >= TREES_PER_WARP || tree >= ntrees) return;", "if (tree >= ntrees) return;", 1)
    .replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\n    #define PACKED_TREES " + str(PACKED_TREES), 1)
)


def packed_launch_dim(ntrees: int) -> int:
    """Threads for ``ntrees`` trees of the packed factor: five per warp, rounded to whole blocks."""
    warps = (ntrees + PACKED_TREES - 1) // PACKED_TREES
    return ((warps * 32 + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM


@functools.cache
def factor_kernel(packed: bool = False):
    """Kernel factoring rod trees (and, with ``with_tau``, solving their unconstrained accelerations)."""

    @wp.func_native(_FACTOR_PACKED if packed else _FACTOR)
    def rod_tree_factor_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        propagation_body_com_rel: wp.array2d[float],
        joint_material: wp.array[wp.spatial_matrix],
        has_joint_material: int,
        joint_tau: wp.array[float],
        with_tau: int,
        branch: wp.array2d[float],
        branch_p: wp.array2d[float],
        body_accel: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        joint_qdd: wp.array[float],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        use_prelude: int,
        IW: wp.array[float],
        KD: wp.array[float],
        SINV: wp.array[float],
    ): ...

    def rod_tree_factor(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        propagation_body_com_rel: wp.array2d[float],
        joint_material: wp.array[wp.spatial_matrix],
        has_joint_material: int,
        joint_tau: wp.array[float],
        with_tau: int,
        branch: wp.array2d[float],
        branch_p: wp.array2d[float],
        body_accel: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        joint_qdd: wp.array[float],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        use_prelude: int,
        IW: wp.array[float],
        KD: wp.array[float],
        SINV: wp.array[float],
    ):
        _tid = wp.tid()
        rod_tree_factor_native(
            ntrees,
            joint_meta,
            group_to_art,
            articulation_active,
            articulation_joint_range,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_joint_S_flat,
            joint_armature,
            max_dofs,
            aug_row_counts,
            aug_row_dof_index,
            aug_row_K,
            body_I_m,
            body_q_com,
            propagation_body_com_rel,
            joint_material,
            has_joint_material,
            joint_tau,
            with_tau,
            branch,
            branch_p,
            body_accel,
            propagation_tree_u,
            propagation_tree_U,
            propagation_tree_D_inv,
            joint_qdd,
            propagation_tree_BR,
            propagation_tree_FR,
            propagation_tree_CR,
            use_prelude,
            IW,
            KD,
            SINV,
        )

    rod_tree_factor.__name__ = "rod_tree_factor" + ("_packed" if packed else "")
    rod_tree_factor.__qualname__ = rod_tree_factor.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_tree_factor)


def launch_dim(ntrees: int) -> int:
    """Threads for ``ntrees`` trees: one warp per five trees, rounded to whole blocks."""
    warps = (ntrees + TREES_PER_WARP - 1) // TREES_PER_WARP
    threads = warps * 32
    return ((threads + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM


_PROPAGATE = (
    r"""
#if defined(__CUDA_ARCH__)
"""
    + _PRELUDE_FULL
    + r"""
    if (grp >= TREES_PER_WARP || tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    // Chains take the scan kernel.
    if (chain_flag.data[art] != 0) return;
    // Worlds whose response the cached GEMV already applied skip the tree walk.
    const int guard_world = art_to_world.data[art];
    if (guard_world >= 0 && cache_world_flag.data[guard_world] != 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int joint_start = articulation_joint_range.data[2 * art];
    const int joint_end = articulation_joint_range.data[2 * art + 1];
    if (joint_end <= joint_start) return;
    const int* meta = joint_meta.data;
    // The factor's per-lane records (see the factor): backward [B row r, W row r], forward [B column r, Q row r,
    // S row r, e, 0], loaded as float4s.
    const float4* BR = reinterpret_cast<const float4*>(propagation_tree_BR.data);
    const float4* FR = reinterpret_cast<const float4*>(propagation_tree_FR.data);
    // Leaf to root, with the factor's maps: the parent gains B p, and the joint keeps w = D^-1 u = W p
    // (u = -S^T p). Only the wrench broadcast is on the chain's critical path.
    // A joint's loads are issued one joint ahead of their use.
    struct Back { int child, parent, gd, dc, flags; float4 m0, m1, m2; float imp; };
    auto loadb = [&](int k, Back& o) {
        o.child = meta[4 * k + 0];
        o.parent = meta[4 * k + 1];
        o.gd = meta[4 * k + 2];
        o.flags = meta[4 * k + 3];
        o.dc = o.flags & 7;
        o.imp = propagation_body_impulses.data[o.child * 6 + r];
        const float4* q = BR + k * 18 + r * 3;
        o.m0 = q[0];
        o.m1 = q[1];
        o.m2 = q[2];
    };
    // Trees without deferred impulses are consistent since their last pass and skip the root-to-leaf pass.
    int any = 0;
    float carry = 0.0f;
    auto stepb = [&](const Back& cur) {
        if (cur.imp != 0.0f) any = 1;
        float pa = -cur.imp + ((cur.flags & 8) ? carry : 0.0f);
        if (cur.flags & 32) {
            pa += branch_p.data[cur.child * 6 + r];
            __syncwarp(GMASK);
            if (lead) branch_p.data[cur.child * 6 + r] = 0.0f;
        }
        float p[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) p[j] = bcast(pa, j);
        const float passed = cur.m0.x * p[0] + cur.m0.y * p[1] + cur.m0.z * p[2] + cur.m0.w * p[3] + cur.m1.x * p[4] + cur.m1.y * p[5];
        if (cur.parent >= 0) {
            if (cur.flags & 16) carry = passed;
            else if (lead) branch_p.data[cur.parent * 6 + r] += passed;
        }
        const float w = cur.m1.z * p[0] + cur.m1.w * p[1] + cur.m2.x * p[2] + cur.m2.y * p[3] + cur.m2.z * p[4] + cur.m2.w * p[5];
        if (r < cur.dc) propagation_tree_u.data[cur.gd + r] = w;
    };
    {
        Back cur, nxt;
        loadb(joint_end - 1, cur);
        for (int k = joint_end - 1; k >= joint_start; --k) {
            if (k - 1 >= joint_start) loadb(k - 1, nxt);
            stepb(cur);
            cur = nxt;
        }
    }
    if (__any_sync(GMASK, any) == 0) return;
    __syncwarp(GMASK);

    // Root to leaf: the twist change delta = B^T delta_parent + S w and qdd = Q delta_parent + w; v_out gains
    // qdd. The link's twist is its old twist (X old_parent + S v_old, carried alongside) plus delta, which
    // replaces the row solve's estimate.
    struct Fwd { int child, parent, gd, dc, flags; float4 m[6]; float w, v; };
    auto loadf = [&](int k, Fwd& o) {
        o.child = meta[4 * k + 0];
        o.parent = meta[4 * k + 1];
        o.gd = meta[4 * k + 2];
        o.flags = meta[4 * k + 3];
        o.dc = o.flags & 7;
        const float4* q = FR + k * 36 + r * 6;
        #pragma unroll
        for (int t = 0; t < 6; ++t) o.m[t] = q[t];
        o.w = (r < o.dc) ? propagation_tree_u.data[o.gd + r] : 0.0f;
        o.v = (r < o.dc) ? v_out.data[o.gd + r] : 0.0f;
    };
    float carry_d = 0.0f;
    float carry_o = 0.0f;
    auto stepf = [&](const Fwd& cur) {
        float at = 0.0f;
        float old = 0.0f;
        if (cur.parent >= 0) {
            // The previous joint's child is this joint's parent along a chain (its flag 16 seen from below).
            if (cur.flags & 16) {
                at = carry_d;
                old = carry_o;
            } else {
                at = body_delta.data[cur.parent * 6 + r];
                old = propagation_body_qd.data[cur.parent * 6 + r] - at;
            }
        }
        // Record floats: [0..5] B column r, [6..11] Q row r, [12..17] S row r, [18..20] e.
        float f[24];
        #pragma unroll
        for (int t = 0; t < 6; ++t) { f[4 * t] = cur.m[t].x; f[4 * t + 1] = cur.m[t].y; f[4 * t + 2] = cur.m[t].z; f[4 * t + 3] = cur.m[t].w; }
        float at_all[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) at_all[j] = bcast(at, j);
        const float o0 = bcast(old, 3), o1 = bcast(old, 4), o2 = bcast(old, 5);
        if (r == 0) old += o1 * f[20] - o2 * f[19];
        if (r == 1) old += o2 * f[18] - o0 * f[20];
        if (r == 2) old += o0 * f[19] - o1 * f[18];
        float dlt = 0.0f;
        float qdd = cur.w;
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            dlt += f[j] * at_all[j];
            qdd += f[6 + j] * at_all[j];
        }
        #pragma unroll
        for (int a = 0; a < 6; ++a) {
            dlt += f[12 + a] * bcast(cur.w, a);
            old += f[12 + a] * bcast(cur.v, a);
        }
        if (r < cur.dc) v_out.data[cur.gd + r] = cur.v + qdd;
        const float vel = old + dlt;
        if (cur.flags & 64) body_delta.data[cur.child * 6 + r] = dlt;
        propagation_body_qd.data[cur.child * 6 + r] = vel;
        propagation_body_impulses.data[cur.child * 6 + r] = 0.0f;
        carry_d = dlt;
        carry_o = old;
    };
    {
        Fwd cur, nxt;
        loadf(joint_start, cur);
        for (int k = joint_start; k < joint_end; ++k) {
            if (k + 1 < joint_end) loadf(k + 1, nxt);
            stepf(cur);
            cur = nxt;
        }
    }
#endif
"""
)


@functools.cache
def propagate_kernel():
    """Kernel applying each rod tree's accumulated link impulses to its joint velocities and link twists."""

    @wp.func_native(_PROPAGATE)
    def rod_tree_propagate_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        art_to_world: wp.array[int],
        cache_world_flag: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        branch_p: wp.array2d[float],
        body_delta: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ): ...

    def rod_tree_propagate(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        art_to_world: wp.array[int],
        cache_world_flag: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        branch_p: wp.array2d[float],
        body_delta: wp.array2d[float],
        propagation_tree_u: wp.array[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ):
        _tid = wp.tid()
        rod_tree_propagate_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            art_to_world,
            cache_world_flag,
            articulation_joint_range,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_tree_BR,
            propagation_tree_FR,
            branch_p,
            body_delta,
            propagation_tree_u,
            propagation_body_impulses,
            propagation_body_qd,
            v_out,
        )

    rod_tree_propagate.__name__ = "rod_tree_propagate"
    rod_tree_propagate.__qualname__ = "rod_tree_propagate"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_tree_propagate)


_RESPONSE = (
    r"""
#if defined(__CUDA_ARCH__)
"""
    + _PRELUDE_FULL
    + r"""
    if (grp >= TREES_PER_WARP || tree >= ntrees) return;
    float* A = shm[wib][grp][0];
    float* B = shm[wib][grp][1];
    float* C = shm[wib][grp][2];
    float* E = shm[wib][grp][3];
    const int art = group_to_art.data[tree];
    // Chains take the scan kernel.
    if (chain_flag.data[art] != 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int joint_start = articulation_joint_range.data[2 * art];
    const int joint_end = articulation_joint_range.data[2 * art + 1];
    const int* meta = joint_meta.data;
    const float* SF = propagation_joint_S_flat.data;
    const float* UF = propagation_tree_U.data;
    const float* DI = propagation_tree_D_inv.data;
    const float* CR = propagation_body_com_rel.data;
    // A joint's inputs as lane r needs them, loaded one joint ahead of their use.
    struct In { int child, parent, gd, dc, flags; float weight, s_row[6], u_col[6], dinv[6], u_row[6], s_col[6], e[3]; };
    auto load = [&](int k, In& o) {
        o.child = meta[4 * k + 0];
        o.parent = meta[4 * k + 1];
        o.gd = meta[4 * k + 2];
        o.flags = meta[4 * k + 3];
        o.dc = o.flags & 7;
        o.weight = (float)max(propagation_joint_split.data[o.flags >> 8], 1);
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            o.s_row[j] = (r < o.dc) ? SF[(o.gd + r) * 6 + j] : 0.0f;
            o.u_col[j] = (j < o.dc) ? UF[(o.gd + j) * 6 + r] : 0.0f;
            o.dinv[j] = (r < o.dc && j < o.dc) ? DI[(o.flags >> 8) * 36 + r * 6 + j] : 0.0f;
            o.u_row[j] = (r < o.dc) ? UF[(o.gd + r) * 6 + j] : 0.0f;
            o.s_col[j] = (j < o.dc) ? SF[(o.gd + j) * 6 + r] : 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < 3; ++c) o.e[c] = (o.parent >= 0) ? CR[o.child * 3 + c] - CR[o.parent * 3 + c] : 0.0f;
    };
    // Root to leaf. For a unit wrench at the child the joint passes P = I - U D^-1 S^T to the parent; the
    // parent's response moves it back as PD = Tv R_parent Tv^T P, and the child's response is
    // PD + S D^-1 (S^T - U^T PD). The split response weights the joint's own term by its row-bearing bodies.
    // Along a chain the parent's response rows stay in registers (ce, cs).
    float ce[6], cs[6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) { ce[c] = 0.0f; cs[c] = 0.0f; }
    In cur, nxt;
    if (joint_end > joint_start) load(joint_start, cur);
    for (int k = joint_start; k < joint_end; ++k) {
        if (k + 1 < joint_end) load(k + 1, nxt);
        const int child = cur.child;
        const int parent = cur.parent;
        const int dc = cur.dc;
        #pragma unroll
        for (int j = 0; j < 6; ++j) { A[r * 6 + j] = cur.s_row[j]; B[r * 6 + j] = cur.dinv[j]; }
        __syncwarp(GMASK);
        // M1 = U D^-1 (row r over DOFs b).
        float m1[6];
        #pragma unroll
        for (int b = 0; b < 6; ++b) {
            float v = 0.0f;
            #pragma unroll
            for (int a = 0; a < 6; ++a) v += cur.u_col[a] * B[a * 6 + b];
            m1[b] = v;
        }
        float pd[6], ps[6];
        if (parent >= 0) {
            // P = I - M1 S^T (row r), then Pp = Tv^T P: rows 3..5 add E times rows 0..2.
            float prow[6];
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = (c == r) ? 1.0f : 0.0f;
                #pragma unroll
                for (int b = 0; b < 6; ++b) v -= m1[b] * A[b * 6 + c];
                prow[c] = v;
            }
            const float ex = cur.e[0], ey = cur.e[1], ez = cur.e[2];
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const float p0 = bcast(prow[c], 0), p1 = bcast(prow[c], 1), p2 = bcast(prow[c], 2);
                if (r == 3) prow[c] += ey * p2 - ez * p1;
                if (r == 4) prow[c] += ez * p0 - ex * p2;
                if (r == 5) prow[c] += ex * p1 - ey * p0;
            }
            __syncwarp(GMASK);
            #pragma unroll
            for (int c = 0; c < 6; ++c) C[r * 6 + c] = prow[c];
            __syncwarp(GMASK);
            // R_parent Pp (exact and split), row r.
            float fe[6], fs[6];
            if (cur.flags & 16) {
                #pragma unroll
                for (int j = 0; j < 6; ++j) { fe[j] = ce[j]; fs[j] = cs[j]; }
            } else {
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    fe[j] = propagation_body_response.data[parent * 36 + r * 6 + j];
                    fs[j] = propagation_body_response_split.data[parent * 36 + r * 6 + j];
                }
            }
            float re[6], rs[6];
            #pragma unroll
            for (int c = 0; c < 6; ++c) { re[c] = 0.0f; rs[c] = 0.0f; }
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                #pragma unroll
                for (int c = 0; c < 6; ++c) { re[c] += fe[j] * C[j * 6 + c]; rs[c] += fs[j] * C[j * 6 + c]; }
            }
            // PD = Tv (R Pp): rows 0..2 add omega x e from rows 3..5.
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const float e3 = bcast(re[c], 3), e4 = bcast(re[c], 4), e5 = bcast(re[c], 5);
                const float s3 = bcast(rs[c], 3), s4 = bcast(rs[c], 4), s5 = bcast(rs[c], 5);
                pd[c] = re[c]; ps[c] = rs[c];
                if (r == 0) { pd[c] += e4 * ez - e5 * ey; ps[c] += s4 * ez - s5 * ey; }
                if (r == 1) { pd[c] += e5 * ex - e3 * ez; ps[c] += s5 * ex - s3 * ez; }
                if (r == 2) { pd[c] += e3 * ey - e4 * ex; ps[c] += s3 * ey - s4 * ex; }
            }
        } else {
            #pragma unroll
            for (int c = 0; c < 6; ++c) { pd[c] = 0.0f; ps[c] = 0.0f; }
        }
        // G = S^T - U^T PD (lane a = r), then H = D^-1 G, and the child's response PD + S H.
        __syncwarp(GMASK);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { A[r * 6 + c] = pd[c]; E[r * 6 + c] = ps[c]; }
        __syncwarp(GMASK);
        float ge[6], gs[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            float te = 0.0f, ts = 0.0f;
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                te += cur.u_row[j] * A[j * 6 + c];
                ts += cur.u_row[j] * E[j * 6 + c];
            }
            ge[c] = (r < dc) ? cur.s_row[c] - te : 0.0f;
            gs[c] = (r < dc) ? cur.weight * cur.s_row[c] - ts : 0.0f;
        }
        __syncwarp(GMASK);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { C[r * 6 + c] = ge[c]; B[r * 6 + c] = gs[c]; }
        __syncwarp(GMASK);
        float he[6], hs[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            float v = 0.0f, w = 0.0f;
            #pragma unroll
            for (int b = 0; b < 6; ++b) { v += cur.dinv[b] * C[b * 6 + c]; w += cur.dinv[b] * B[b * 6 + c]; }
            he[c] = v;
            hs[c] = w;
        }
        __syncwarp(GMASK);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { A[r * 6 + c] = he[c]; E[r * 6 + c] = hs[c]; }
        __syncwarp(GMASK);
        // S column r of each DOF: S[a][r].
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            float ve = pd[c], vs = ps[c];
            #pragma unroll
            for (int a = 0; a < 6; ++a) {
                ve += cur.s_col[a] * A[a * 6 + c];
                vs += cur.s_col[a] * E[a * 6 + c];
            }
            propagation_body_response.data[child * 36 + r * 6 + c] = ve;
            propagation_body_response_split.data[child * 36 + r * 6 + c] = vs;
            ce[c] = ve;
            cs[c] = vs;
        }
        __syncwarp(GMASK);
        cur = nxt;
    }
#endif
"""
)


@functools.cache
def response_kernel():
    """Kernel writing every rod-tree link's 6x6 response and its split twin."""

    @wp.func_native(_RESPONSE)
    def rod_tree_response_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_joint_split: wp.array[int],
        propagation_body_response: wp.array3d[float],
        propagation_body_response_split: wp.array3d[float],
    ): ...

    def rod_tree_response(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_joint_split: wp.array[int],
        propagation_body_response: wp.array3d[float],
        propagation_body_response_split: wp.array3d[float],
    ):
        _tid = wp.tid()
        rod_tree_response_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            joint_parent,
            joint_child,
            joint_qd_start,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            propagation_joint_split,
            propagation_body_response,
            propagation_body_response_split,
        )

    rod_tree_response.__name__ = "rod_tree_response"
    rod_tree_response.__qualname__ = "rod_tree_response"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_tree_response)


def joint_meta(model, joint_ranges) -> np.ndarray:
    """Static per-joint walk record, in each tree's depth-first walk order: child, parent body, first DOF, flags.

    The walks visit a tree's joints in depth-first order, so a chain's joints are consecutive and hand their
    terms along in registers whatever order the model lists them in (an interleaved Y-junction, say). Flags:
    DOF count (bits 0-2), the next joint (deeper) is this child's chain child (8), this joint's parent is the
    previous joint's child (16, a chain step up), this child also receives branch contributions (32), this
    child is the parent of a joint that is not its chain successor (64, its velocity change is read back),
    and the model's joint index from bit 8 (for arrays shared with the other kernels).
    """
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    qd_start = model.joint_qd_start.numpy()
    meta = np.zeros((max(model.joint_count, 1), 4), dtype=np.int32)
    for start, end in joint_ranges:
        joints = range(start, end)
        child_of = {int(children[k]): k for k in joints}
        by_parent: dict[int, list[int]] = {}
        for k in joints:
            by_parent.setdefault(int(parents[k]), []).append(k)
        # Depth-first order: a joint's first child joint follows it directly.
        order: list[int] = []
        stack = [k for k in reversed(joints) if int(parents[k]) not in child_of]
        while stack:
            k = stack.pop()
            order.append(k)
            stack.extend(reversed(by_parent.get(int(children[k]), [])))
        if len(order) != end - start:
            raise ValueError("rod tree joints do not form a tree")
        for pos, k in enumerate(order):
            child, parent = int(children[k]), int(parents[k])
            dc = int(qd_start[k + 1] - qd_start[k])
            flags = min(dc, 7)
            nxt = order[pos + 1] if pos + 1 < len(order) else -1
            prv = order[pos - 1] if pos > 0 else -1
            if nxt >= 0 and int(parents[nxt]) == child:
                flags |= 8
            if prv >= 0 and int(children[prv]) == parent and parent >= 0:
                flags |= 16
            if any(kk != nxt for kk in by_parent.get(child, ())):
                flags |= 32 | 64
            meta[start + pos] = (child, parent, int(qd_start[k]), flags | (k << 8))
    return meta.reshape(-1)


_REFRESH = (
    r"""
#if defined(__CUDA_ARCH__)
"""
    + _PRELUDE_FULL
    + r"""
    if (grp >= TREES_PER_WARP || tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    // Chains take the scan kernel.
    if (chain_flag.data[art] != 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int joint_start = articulation_joint_range.data[2 * art];
    const int joint_end = articulation_joint_range.data[2 * art + 1];
    const int* meta = joint_meta.data;
    const float* SF = propagation_joint_S_flat.data;
    const float* CR = propagation_body_com_rel.data;
    // Root to leaf: each link's twist is its parent's moved to its center of mass plus S v_out.
    float carry = 0.0f;
    for (int k = joint_start; k < joint_end; ++k) {
        const int child = meta[4 * k + 0];
        const int parent = meta[4 * k + 1];
        const int gd = meta[4 * k + 2];
        const int flags = meta[4 * k + 3];
        const int dc = flags & 7;
        float vt = 0.0f;
        float ex = 0.0f, ey = 0.0f, ez = 0.0f;
        if (parent >= 0) {
            vt = (flags & 16) ? carry : propagation_body_qd.data[parent * 6 + r];
            ex = CR[child * 3 + 0] - CR[parent * 3 + 0];
            ey = CR[child * 3 + 1] - CR[parent * 3 + 1];
            ez = CR[child * 3 + 2] - CR[parent * 3 + 2];
        }
        const float o0 = bcast(vt, 3), o1 = bcast(vt, 4), o2 = bcast(vt, 5);
        if (r == 0) vt += o1 * ez - o2 * ey;
        if (r == 1) vt += o2 * ex - o0 * ez;
        if (r == 2) vt += o0 * ey - o1 * ex;
        #pragma unroll
        for (int a = 0; a < 6; ++a) if (a < dc) vt += SF[(gd + a) * 6 + r] * v_out.data[gd + a];
        propagation_body_qd.data[child * 6 + r] = vt;
        carry = vt;
    }
#endif
"""
)


@functools.cache
def refresh_kernel():
    """Kernel recomputing every rod-tree link's twist from the joint velocities."""

    @wp.func_native(_REFRESH)
    def rod_tree_refresh_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ): ...

    def rod_tree_refresh(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ):
        _tid = wp.tid()
        rod_tree_refresh_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            v_out,
            propagation_body_qd,
        )

    rod_tree_refresh.__name__ = "rod_tree_refresh"
    rod_tree_refresh.__qualname__ = "rod_tree_refresh"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_tree_refresh)


_CHAIN_PROPAGATE = (
    "\n    #define CHAIN_REC "
    + str(CHAIN_REC)
    + "\n    #define CONTRACT_KMAX 32"
    + r"""
#if defined(__CUDA_ARCH__)
    // One warp per chain. Lane s owns the chain's links [s*L, s*L + L): it composes its segment's maps in
    // registers, and the segments combine in a warp scan, so a pass costs about L + log2(32) map products
    // instead of one dependent step per link. The pass is bound by memory latency, not arithmetic, so a chain
    // that fits stages its links' records, indices and impulses in shared memory first, with wide coalesced
    // loads, and every pass reads them there.
    const int lane = threadIdx.x & 31;
    const int tree = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    if (chain_flag.data[art] == 0) return;
    const int guard_world = art_to_world.data[art];
    if (guard_world >= 0 && cache_world_flag.data[guard_world] != 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int js = articulation_joint_range.data[2 * art];
    const int n = articulation_joint_range.data[2 * art + 1] - js;
    if (n <= 0) return;
    const int L = (n + 31) / 32;
    const int lo = min(lane * L, n);
    const int hi = min(lo + L, n);
    const int* meta = joint_meta.data;
    const unsigned FULL = 0xffffffffu;

    #define CTR_SMEM (CONTRACT_KMAX * 90)
    __shared__ __align__(16) float s_rec[(STAGE_MAX * CHAIN_REC > CTR_SMEM) ? STAGE_MAX * CHAIN_REC : CTR_SMEM];
    __shared__ int s_link[(STAGE_MAX * 3 > CONTRACT_KMAX) ? STAGE_MAX * 3 : CONTRACT_KMAX];
    __shared__ float s_imp[STAGE_MAX * 6];
    __shared__ float s_u[STAGE_MAX * 6];
    __shared__ float s_v[STAGE_MAX * 6];
    const bool staged = n <= STAGE_MAX;
    // Contracted iterations (contract_max > 0): a chain with few active links keeps its impulses and only sets its
    // active links' twists from their sum (see _CONTRACT_PREP); the others propagate as usual.
    const int Kc = contract_max ? n_active.data[art] : 0;
    if (Kc > 0 && Kc <= contract_max && 4 * Kc <= n) {
        const int K = Kc;
        float* s_P = s_rec;
        float* s_R = s_rec + CONTRACT_KMAX * 36;
        float* s_f = s_rec + CONTRACT_KMAX * 72;
        float* s_a = s_f + CONTRACT_KMAX * 6;
        float* s_d = s_a + CONTRACT_KMAX * 6;
        int* s_b = s_link;
        float* QD = propagation_body_qd.data;
        for (int i = lane; i < K; i += 32) s_b[i] = active_body.data[js + i];
        __syncwarp(FULL);
        for (int e = lane; e < K * 36; e += 32) {
            s_P[e] = active_P.data[js * 36 + e];
            s_R[e] = propagation_body_response.data[s_b[e / 36] * 36 + e % 36];
        }
        for (int e = lane; e < K * 6; e += 32) s_f[e] = propagation_body_impulses.data[s_b[e / 6] * 6 + e % 6];
        __syncwarp(FULL);
        if (lane == 0) {
            float a[6];
            #pragma unroll
            for (int r = 0; r < 6; ++r) { a[r] = s_f[(K - 1) * 6 + r]; s_a[(K - 1) * 6 + r] = a[r]; }
            for (int i = K - 2; i >= 0; --i) {
                const float* P = s_P + i * 36;
                float na[6];
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    float v = s_f[i * 6 + r];
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) v += P[r * 6 + c] * a[c];
                    na[r] = v;
                }
                #pragma unroll
                for (int r = 0; r < 6; ++r) { a[r] = na[r]; s_a[i * 6 + r] = na[r]; }
            }
            float bv[6] = {0, 0, 0, 0, 0, 0};
            for (int i = 0; i < K; ++i) {
                const float* R = s_R + i * 36;
                float t[6];
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    float d = bv[r], rf = bv[r];
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) { d += R[r * 6 + c] * s_a[i * 6 + c]; rf += R[r * 6 + c] * s_f[i * 6 + c]; }
                    s_d[i * 6 + r] = d;
                    t[r] = rf;
                }
                if (i + 1 < K) {
                    const float* P = s_P + i * 36;
                    #pragma unroll
                    for (int r = 0; r < 6; ++r) {
                        float v = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < 6; ++c) v += P[c * 6 + r] * t[c];
                        bv[r] = v;
                    }
                }
            }
        }
        __syncwarp(FULL);
        for (int e = lane; e < K * 6; e += 32) QD[s_b[e / 6] * 6 + e % 6] = active_qd0.data[js * 6 + e] + s_d[e];
        return;
    }
    const float* gcr = propagation_tree_CR.data + js * CHAIN_REC;
    int any = 0;
    if (staged) {
        const float4* g4 = reinterpret_cast<const float4*>(gcr);
        float4* d4 = reinterpret_cast<float4*>(s_rec);
        for (int q = lane; q < n * (CHAIN_REC / 4); q += 32) d4[q] = g4[q];
        for (int k = lane; k < n; k += 32) {
            const int4 m = *reinterpret_cast<const int4*>(meta + 4 * (js + k));
            s_link[3 * k] = m.x;
            s_link[3 * k + 1] = m.z;
            s_link[3 * k + 2] = m.w & 7;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const float v = propagation_body_impulses.data[m.x * 6 + c];
                s_imp[6 * k + c] = v;
                if (v != 0.0f) any = 1;
            }
        }
        __syncwarp(FULL);
    }
    const float* rec = staged ? s_rec : gcr;
    auto child_of = [&](int k) { return staged ? s_link[3 * k] : meta[4 * (js + k)]; };
    auto gd_of = [&](int k) { return staged ? s_link[3 * k + 1] : meta[4 * (js + k) + 2]; };
    auto dc_of = [&](int k) { return staged ? s_link[3 * k + 2] : (meta[4 * (js + k) + 3] & 7); };
    auto imp_of = [&](int k, int c) {
        return staged ? s_imp[6 * k + c] : propagation_body_impulses.data[meta[4 * (js + k)] * 6 + c];
    };
    // y = G x for G at offset off of link k's record (row-major), or its transpose.
    auto mv = [&](int k, int off, const float* x, float* y) {
        const float* g = rec + k * CHAIN_REC + off;
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            float v = 0.0f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) v += g[r * 6 + c] * x[c];
            y[r] = v;
        }
    };
    auto mtv = [&](int k, int off, const float* x, float* y) {
        const float* g = rec + k * CHAIN_REC + off;
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            float v = 0.0f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) v += g[c * 6 + r] * x[c];
            y[r] = v;
        }
    };
    auto loadm = [&](int k, int off, float* m, bool transpose) {
        const float* g = rec + k * CHAIN_REC + off;
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) m[r * 6 + c] = transpose ? g[c * 6 + r] : g[r * 6 + c];
    };
    auto matmul = [](const float* a, const float* b, float* out) {
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                #pragma unroll
                for (int q = 0; q < 6; ++q) v += a[r * 6 + q] * b[q * 6 + c];
                out[r * 6 + c] = v;
            }
    };
    // Warp scan of segment maps y = x + M y_in; on return x is the segment's inflow (backward: from the next
    // segment, forward: from the previous one). M is consumed.
    auto scan = [&](float* x, float* M, bool backward) {
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            float xs[6], Ms[36], nm[36];
            #pragma unroll
            for (int r = 0; r < 6; ++r) xs[r] = backward ? __shfl_down_sync(FULL, x[r], d) : __shfl_up_sync(FULL, x[r], d);
            #pragma unroll
            for (int e = 0; e < 36; ++e) Ms[e] = backward ? __shfl_down_sync(FULL, M[e], d) : __shfl_up_sync(FULL, M[e], d);
            if (backward ? (lane + d < 32) : (lane >= d)) {
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    float v = x[r];
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) v += M[r * 6 + c] * xs[c];
                    x[r] = v;
                }
                matmul(M, Ms, nm);
                #pragma unroll
                for (int e = 0; e < 36; ++e) M[e] = nm[e];
            }
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            const float v = backward ? __shfl_down_sync(FULL, x[r], 1) : __shfl_up_sync(FULL, x[r], 1);
            x[r] = (backward ? lane < 31 : lane > 0) ? v : 0.0f;
        }
    };

    // Backward: p_k = -imp_k + B_{k+1} p_{k+1}. The segment gives p_lo = a + C p_hi.
    float a[6] = {0, 0, 0, 0, 0, 0};
    float C[36];
    #pragma unroll
    for (int e = 0; e < 36; ++e) C[e] = (e % 7 == 0) ? 1.0f : 0.0f;
    for (int k = hi - 1; k >= lo; --k) {
        float na[6];
        if (k + 1 < n) {
            float B[36], nc[36];
            loadm(k + 1, 0, B, false);
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                float v = 0.0f;
                #pragma unroll
                for (int c = 0; c < 6; ++c) v += B[r * 6 + c] * a[c];
                na[r] = v;
            }
            matmul(B, C, nc);
            #pragma unroll
            for (int e = 0; e < 36; ++e) C[e] = nc[e];
        } else {
            #pragma unroll
            for (int r = 0; r < 6; ++r) na[r] = 0.0f;
            #pragma unroll
            for (int e = 0; e < 36; ++e) C[e] = 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            const float im = imp_of(k, r);
            if (!staged && im != 0.0f) any = 1;
            a[r] = na[r] - im;
        }
    }
    if (__any_sync(FULL, any) == 0) return;
    scan(a, C, true);
    // Fix-up: each link's wrench with the true inflow, and w = W p.
    for (int k = hi - 1; k >= lo; --k) {
        float p[6];
        if (k + 1 < n) {
            mv(k + 1, 0, a, p);
        } else {
            #pragma unroll
            for (int r = 0; r < 6; ++r) p[r] = 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) p[r] -= imp_of(k, r);
        float w[6];
        mv(k, 36, p, w);
        const int gd = gd_of(k);
        const int dc = dc_of(k);
        for (int r = 0; r < dc; ++r) propagation_tree_u.data[gd + r] = w[r];
        if (staged) {
            #pragma unroll
            for (int r = 0; r < 6; ++r) s_u[6 * k + r] = (r < dc) ? w[r] : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) a[r] = p[r];
    }
    __syncwarp(FULL);
    auto u_of = [&](int k, float* w) {
        const int gd = gd_of(k);
        const int dc = dc_of(k);
        #pragma unroll
        for (int r = 0; r < 6; ++r) w[r] = staged ? s_u[6 * k + r] : ((r < dc) ? propagation_tree_u.data[gd + r] : 0.0f);
    };

    // Forward: d_k = A_k d_{k-1} + S_k w_k with A_k = B_k^T; the segment gives d_{hi-1} = b + H d_{lo-1}.
    float b[6] = {0, 0, 0, 0, 0, 0};
    float H[36];
    #pragma unroll
    for (int e = 0; e < 36; ++e) H[e] = (e % 7 == 0) ? 1.0f : 0.0f;
    for (int k = lo; k < hi; ++k) {
        float w[6], sw[6];
        u_of(k, w);
        mv(k, 72, w, sw);
        if (k > 0) {
            float A[36], nh[36], nb[6];
            loadm(k, 0, A, true);
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                float v = sw[r];
                #pragma unroll
                for (int c = 0; c < 6; ++c) v += A[r * 6 + c] * b[c];
                nb[r] = v;
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) b[r] = nb[r];
            matmul(A, H, nh);
            #pragma unroll
            for (int e = 0; e < 36; ++e) H[e] = nh[e];
        } else {
            #pragma unroll
            for (int r = 0; r < 6; ++r) b[r] = sw[r];
            #pragma unroll
            for (int e = 0; e < 36; ++e) H[e] = 0.0f;
        }
    }
    scan(b, H, false);
    // Fix-up: each link's twist change carries to the next link and its joint acceleration goes to v_out.
    for (int k = lo; k < hi; ++k) {
        float w[6], d[6], qdd[6];
        u_of(k, w);
        mv(k, 72, w, d);
        mv(k, 108, b, qdd);
        if (k > 0) {
            float t[6];
            mtv(k, 0, b, t);
            #pragma unroll
            for (int r = 0; r < 6; ++r) d[r] += t[r];
        }
        const int child = child_of(k);
        const int gd = gd_of(k);
        const int dc = dc_of(k);
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            if (r < dc) {
                const float nv = v_out.data[gd + r] + qdd[r] + w[r];
                v_out.data[gd + r] = nv;
                if (staged) s_v[6 * k + r] = nv;
            } else if (staged) {
                s_v[6 * k + r] = 0.0f;
            }
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            propagation_body_impulses.data[child * 6 + r] = 0.0f;
            b[r] = d[r];
        }
    }
    __syncwarp(FULL);

    // The links' twists from the updated joint velocities (as the refresh forms them, so they carry no
    // accumulated rounding): o_k = X(e_k) o_{k-1} + S_k v_k, where X(e) adds omega x e to the linear part and
    // shifts compose by adding their offsets. The segment gives o_{hi-1} = X(E) o_{lo-1} + t.
    auto shift = [](float* o, const float* e) {
        const float w0 = o[3], w1 = o[4], w2 = o[5];
        o[0] += w1 * e[2] - w2 * e[1];
        o[1] += w2 * e[0] - w0 * e[2];
        o[2] += w0 * e[1] - w1 * e[0];
    };
    auto link_twist = [&](int k, float* sv, float* e) {
        float v[6];
        if (staged) {
            #pragma unroll
            for (int r = 0; r < 6; ++r) v[r] = s_v[6 * k + r];
        } else {
            const int gd = gd_of(k);
            const int dc = dc_of(k);
            #pragma unroll
            for (int r = 0; r < 6; ++r) v[r] = (r < dc) ? v_out.data[gd + r] : 0.0f;
        }
        mv(k, 72, v, sv);
        const float* er = rec + k * CHAIN_REC + 144;
        #pragma unroll
        for (int c = 0; c < 3; ++c) e[c] = (k > 0) ? er[c] : 0.0f;
    };
    float t[6] = {0, 0, 0, 0, 0, 0};
    float E[3] = {0, 0, 0};
    for (int k = lo; k < hi; ++k) {
        float sv[6], e[3];
        link_twist(k, sv, e);
        shift(t, e);
        #pragma unroll
        for (int r = 0; r < 6; ++r) t[r] += sv[r];
        #pragma unroll
        for (int c = 0; c < 3; ++c) E[c] += e[c];
    }
    #pragma unroll
    for (int dd = 1; dd < 32; dd <<= 1) {
        float ts[6], Es[3];
        #pragma unroll
        for (int r = 0; r < 6; ++r) ts[r] = __shfl_up_sync(FULL, t[r], dd);
        #pragma unroll
        for (int c = 0; c < 3; ++c) Es[c] = __shfl_up_sync(FULL, E[c], dd);
        if (lane >= dd) {
            shift(ts, E);
            #pragma unroll
            for (int r = 0; r < 6; ++r) t[r] += ts[r];
            #pragma unroll
            for (int c = 0; c < 3; ++c) E[c] += Es[c];
        }
    }
    float o[6];
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        const float v = __shfl_up_sync(FULL, t[r], 1);
        o[r] = (lane > 0) ? v : 0.0f;
    }
    for (int k = lo; k < hi; ++k) {
        float sv[6], e[3];
        link_twist(k, sv, e);
        shift(o, e);
        #pragma unroll
        for (int r = 0; r < 6; ++r) o[r] += sv[r];
        const int child = child_of(k);
        #pragma unroll
        for (int r = 0; r < 6; ++r) propagation_body_qd.data[child * 6 + r] = o[r];
    }
#endif
"""
)


@functools.cache
def chain_propagate_kernel(staged: bool = True):
    """Kernel propagating each chain-shaped rod tree's link impulses with a warp scan (see ``_CHAIN_PROPAGATE``).

    ``staged`` reserves shared memory for chains of up to CHAIN_STAGE_MAX links: the faster pass for a lone chain,
    and a cap on how many chains share an SM, so a launch whose chains fill the GPU on their own takes the
    unstaged variant.
    """

    @wp.func_native("\n    #define STAGE_MAX " + str(CHAIN_STAGE_MAX if staged else 1) + "\n" + _CHAIN_PROPAGATE)
    def rod_chain_propagate_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        art_to_world: wp.array[int],
        cache_world_flag: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        propagation_tree_u: wp.array[float],
        n_active: wp.array[int],
        contract_max: int,
        active_body: wp.array[int],
        active_P: wp.array[float],
        propagation_body_response: wp.array3d[float],
        active_qd0: wp.array[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ): ...

    def rod_chain_propagate(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        art_to_world: wp.array[int],
        cache_world_flag: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        propagation_tree_u: wp.array[float],
        n_active: wp.array[int],
        contract_max: int,
        active_body: wp.array[int],
        active_P: wp.array[float],
        propagation_body_response: wp.array3d[float],
        active_qd0: wp.array[float],
        propagation_body_impulses: wp.array2d[float],
        propagation_body_qd: wp.array2d[float],
        v_out: wp.array[float],
    ):
        _tid = wp.tid()
        rod_chain_propagate_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            art_to_world,
            cache_world_flag,
            articulation_joint_range,
            propagation_tree_BR,
            propagation_tree_FR,
            propagation_tree_CR,
            propagation_tree_u,
            n_active,
            contract_max,
            active_body,
            active_P,
            propagation_body_response,
            active_qd0,
            propagation_body_impulses,
            propagation_body_qd,
            v_out,
        )

    rod_chain_propagate.__name__ = "rod_chain_propagate" + ("" if staged else "_unstaged")
    rod_chain_propagate.__qualname__ = rod_chain_propagate.__name__
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_chain_propagate)


def chain_launch_dim(ntrees: int) -> int:
    """Threads for ``ntrees`` chains: one warp each, rounded to whole blocks."""
    return ((ntrees * 32 + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM


def chain_flags(model, art_ranges, meta: np.ndarray) -> np.ndarray:
    """Per articulation: 1 when its rod tree is a single chain in walk order (every joint hangs off the last).

    ``art_ranges`` lists ``(articulation, first joint, joint end)`` of the rod trees.
    """
    flags = np.zeros(max(model.articulation_count, 1), dtype=np.int32)
    m = meta.reshape(-1, 4)
    for art, start, end in art_ranges:
        if end - start < 2:
            continue
        f = m[start:end, 3]
        if np.all((f[1:] & 16) != 0) and not np.any((f & (32 | 64)) != 0):
            flags[art] = 1
    return flags


_CHAIN_RESPONSE = r"""
#if defined(__CUDA_ARCH__)
    // One warp per chain: R_k = M_k R_{k-1} M_k^T + w_k K_k with M_k = B_k^T and K_k = S_k D_k^-1 S_k^T (the split
    // twin weights K_k by the joint's row-bearing bodies). These maps compose as (M2, N2)(M1, N1) =
    // (M2 M1, M2 N1 M2^T + N2), so lane s composes its segment's links and the segments combine in a warp scan.
    const int lane = threadIdx.x & 31;
    const int tree = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    if (chain_flag.data[art] == 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int js = articulation_joint_range.data[2 * art];
    const int n = articulation_joint_range.data[2 * art + 1] - js;
    if (n <= 0) return;
    const int L = (n + 31) / 32;
    const int lo = min(lane * L, n);
    const int hi = min(lo + L, n);
    const int* meta = joint_meta.data;
    const float* CRR = propagation_tree_CR.data;
    const float* DI = propagation_tree_D_inv.data;
    const unsigned FULL = 0xffffffffu;

    // out = a b, out = a b a^T (symmetric b).
    auto matmul = [](const float* a, const float* b, float* out) {
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                #pragma unroll
                for (int t = 0; t < 6; ++t) v += a[r * 6 + t] * b[t * 6 + c];
                out[r * 6 + c] = v;
            }
    };
    auto congruence_add = [&](const float* m, const float* x, float* acc) {
        float t[36];
        matmul(m, x, t);
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                #pragma unroll
                for (int q = 0; q < 6; ++q) v += t[r * 6 + q] * m[c * 6 + q];
                acc[r * 6 + c] += v;
            }
    };
    // Link k's M (zero at the root) and K = S D^-1 S^T, and its split weight.
    auto link = [&](int k, float* Mk, float* Kk, float& weight) {
        const int p4 = 4 * (js + k);
        const int flags = meta[p4 + 3];
        const int dc = flags & 7;
        const int jid = flags >> 8;
        const float* br = CRR + (js + k) * 148;
        const float* fr = br;
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) Mk[r * 6 + c] = (k > 0) ? br[c * 6 + r] : 0.0f;
        float SD[36];
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int b = 0; b < 6; ++b) {
                float v = 0.0f;
                #pragma unroll
                for (int a = 0; a < 6; ++a)
                    if (a < dc && b < dc) v += fr[72 + r * 6 + a] * DI[jid * 36 + a * 6 + b];
                SD[r * 6 + b] = v;
            }
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                #pragma unroll
                for (int b = 0; b < 6; ++b) v += SD[r * 6 + b] * fr[72 + c * 6 + b];
                Kk[r * 6 + c] = v;
            }
        weight = (float)max(propagation_joint_split.data[jid], 1);
    };
    // Without joints shared by several rows (every split weight 1) the split response is the exact one: skip it.
    int local_split = 0;
    for (int k = lo; k < hi; ++k)
        if (propagation_joint_split.data[meta[4 * (js + k) + 3] >> 8] > 1) local_split = 1;
    const bool split = __any_sync(FULL, local_split) != 0;
    // Segment map: R_{hi-1} = M R_{lo-1} M^T + N (exact) and + Ns (split).
    float M[36], N[36], Ns[36];
    #pragma unroll
    for (int e = 0; e < 36; ++e) { M[e] = (e % 7 == 0) ? 1.0f : 0.0f; N[e] = 0.0f; Ns[e] = 0.0f; }
    for (int k = lo; k < hi; ++k) {
        float Mk[36], Kk[36], weight;
        link(k, Mk, Kk, weight);
        float nn[36], ns[36], nm[36];
        #pragma unroll
        for (int e = 0; e < 36; ++e) { nn[e] = Kk[e]; ns[e] = weight * Kk[e]; }
        congruence_add(Mk, N, nn);
        if (split) congruence_add(Mk, Ns, ns);
        matmul(Mk, M, nm);
        #pragma unroll
        for (int e = 0; e < 36; ++e) { N[e] = nn[e]; Ns[e] = ns[e]; M[e] = nm[e]; }
    }
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        float Mp[36], Np[36], Nsp[36];
        #pragma unroll
        for (int e = 0; e < 36; ++e) {
            Mp[e] = __shfl_up_sync(FULL, M[e], d);
            Np[e] = __shfl_up_sync(FULL, N[e], d);
            Nsp[e] = split ? __shfl_up_sync(FULL, Ns[e], d) : 0.0f;
        }
        if (lane >= d) {
            congruence_add(M, Np, N);
            if (split) congruence_add(M, Nsp, Ns);
            float nm[36];
            matmul(M, Mp, nm);
            #pragma unroll
            for (int e = 0; e < 36; ++e) M[e] = nm[e];
        }
    }
    // The segment's inflow is the previous segment's last response.
    float R[36], Rs[36];
    #pragma unroll
    for (int e = 0; e < 36; ++e) {
        const float a = __shfl_up_sync(FULL, N[e], 1);
        const float b = split ? __shfl_up_sync(FULL, Ns[e], 1) : 0.0f;
        R[e] = (lane > 0) ? a : 0.0f;
        Rs[e] = (lane > 0) ? b : 0.0f;
    }
    for (int k = lo; k < hi; ++k) {
        float Mk[36], Kk[36], weight;
        link(k, Mk, Kk, weight);
        float nr[36], ns[36];
        #pragma unroll
        for (int e = 0; e < 36; ++e) { nr[e] = Kk[e]; ns[e] = weight * Kk[e]; }
        congruence_add(Mk, R, nr);
        if (split) congruence_add(Mk, Rs, ns);
        const int child = meta[4 * (js + k)];
        #pragma unroll
        for (int e = 0; e < 36; ++e) {
            R[e] = nr[e];
            Rs[e] = ns[e];
            propagation_body_response.data[child * 36 + e] = nr[e];
            propagation_body_response_split.data[child * 36 + e] = split ? ns[e] : nr[e];
        }
    }
#endif
"""


@functools.cache
def chain_response_kernel():
    """Kernel writing each chain-shaped rod tree's link responses with a warp scan (see ``_CHAIN_RESPONSE``)."""

    @wp.func_native(_CHAIN_RESPONSE)
    def rod_chain_response_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_joint_split: wp.array[int],
        propagation_body_response: wp.array3d[float],
        propagation_body_response_split: wp.array3d[float],
    ): ...

    def rod_chain_response(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        propagation_tree_D_inv: wp.array3d[float],
        propagation_joint_split: wp.array[int],
        propagation_body_response: wp.array3d[float],
        propagation_body_response_split: wp.array3d[float],
    ):
        _tid = wp.tid()
        rod_chain_response_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            propagation_tree_BR,
            propagation_tree_FR,
            propagation_tree_CR,
            propagation_tree_D_inv,
            propagation_joint_split,
            propagation_body_response,
            propagation_body_response_split,
        )

    rod_chain_response.__name__ = "rod_chain_response"
    rod_chain_response.__qualname__ = "rod_chain_response"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_chain_response)


_CHAIN_REFRESH = r"""
#if defined(__CUDA_ARCH__)
    #define CHAIN_REC_N 148
    const int lane = threadIdx.x & 31;
    const int tree = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    if (chain_flag.data[art] == 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int js = articulation_joint_range.data[2 * art];
    const int n = articulation_joint_range.data[2 * art + 1] - js;
    if (n <= 0) return;
    const int L = (n + 31) / 32;
    const int lo = min(lane * L, n);
    const int hi = min(lo + L, n);
    const int* meta = joint_meta.data;
    const float* CRR = propagation_tree_CR.data;
    const unsigned FULL = 0xffffffffu;
    auto matvec_rows = [](const float* q, int stride, int off, const float* x, float* y) {
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            float v = 0.0f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) v += q[r * stride + off + c] * x[c];
            y[r] = v;
        }
    };
    // The links' twists from the updated joint velocities (as the refresh forms them, so they carry no
    // accumulated rounding): o_k = X(e_k) o_{k-1} + S_k v_k, where X(e) adds omega x e to the linear part and
    // shifts compose by adding their offsets. The segment gives o_{hi-1} = X(E) o_{lo-1} + t.
    auto shift = [](float* o, const float* e) {
        const float w0 = o[3], w1 = o[4], w2 = o[5];
        o[0] += w1 * e[2] - w2 * e[1];
        o[1] += w2 * e[0] - w0 * e[2];
        o[2] += w0 * e[1] - w1 * e[0];
    };
    auto link_twist = [&](int k, float* sv, float* e) {
        const int p4 = 4 * (js + k);
        const int gd = meta[p4 + 2];
        const int dc = meta[p4 + 3] & 7;
        const float* fr = CRR + (js + k) * CHAIN_REC_N;
        float v[6];
        #pragma unroll
        for (int r = 0; r < 6; ++r) v[r] = (r < dc) ? v_out.data[gd + r] : 0.0f;
        matvec_rows(fr, 6, 72, v, sv);
        #pragma unroll
        for (int c = 0; c < 3; ++c) e[c] = (k > 0) ? fr[144 + c] : 0.0f;
    };
    float t[6] = {0, 0, 0, 0, 0, 0};
    float E[3] = {0, 0, 0};
    for (int k = lo; k < hi; ++k) {
        float sv[6], e[3];
        link_twist(k, sv, e);
        shift(t, e);
        #pragma unroll
        for (int r = 0; r < 6; ++r) t[r] += sv[r];
        #pragma unroll
        for (int c = 0; c < 3; ++c) E[c] += e[c];
    }
    #pragma unroll
    for (int dd = 1; dd < 32; dd <<= 1) {
        float ts[6], Es[3];
        #pragma unroll
        for (int r = 0; r < 6; ++r) ts[r] = __shfl_up_sync(FULL, t[r], dd);
        #pragma unroll
        for (int c = 0; c < 3; ++c) Es[c] = __shfl_up_sync(FULL, E[c], dd);
        if (lane >= dd) {
            shift(ts, E);
            #pragma unroll
            for (int r = 0; r < 6; ++r) t[r] += ts[r];
            #pragma unroll
            for (int c = 0; c < 3; ++c) E[c] += Es[c];
        }
    }
    float o[6];
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        const float v = __shfl_up_sync(FULL, t[r], 1);
        o[r] = (lane > 0) ? v : 0.0f;
    }
    for (int k = lo; k < hi; ++k) {
        float sv[6], e[3];
        link_twist(k, sv, e);
        shift(o, e);
        #pragma unroll
        for (int r = 0; r < 6; ++r) o[r] += sv[r];
        const int child = meta[4 * (js + k)];
        #pragma unroll
        for (int r = 0; r < 6; ++r) propagation_body_qd.data[child * 6 + r] = o[r];
    }
#endif
"""


@functools.cache
def chain_refresh_kernel():
    """Kernel recomputing each chain-shaped rod tree's link twists from v_out with a warp scan."""

    @wp.func_native(_CHAIN_REFRESH)
    def rod_chain_refresh_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ): ...

    def rod_chain_refresh(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        v_out: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ):
        _tid = wp.tid()
        rod_chain_refresh_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            propagation_tree_FR,
            propagation_tree_CR,
            v_out,
            propagation_body_qd,
        )

    rod_chain_refresh.__name__ = "rod_chain_refresh"
    rod_chain_refresh.__qualname__ = "rod_chain_refresh"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_chain_refresh)


_CHAIN_WORLD_TWISTS = r"""
#if defined(__CUDA_ARCH__)
    // One warp per chain: in the articulation-origin frame a link's twist is its parent's plus S qd, so the
    // twists are a prefix sum along the chain; each is then moved to its body's center of mass.
    const int lane = threadIdx.x & 31;
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (t >= narts) return;
    const int art = integrated.data[t];
    if (chain_flag.data[art] == 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int js = articulation_joint_range.data[2 * art];
    const int n = articulation_joint_range.data[2 * art + 1] - js;
    if (n <= 0) return;
    const int L = (n + 31) / 32;
    const int lo = min(lane * L, n);
    const int hi = min(lo + L, n);
    const int* meta = joint_meta.data;
    const float* S = reinterpret_cast<const float*>(joint_S_s.data);
    const unsigned FULL = 0xffffffffu;
    auto link = [&](int k, float* v) {
        const int gd = meta[4 * (js + k) + 2];
        const int dc = meta[4 * (js + k) + 3] & 7;
        for (int a = 0; a < dc; ++a) {
            const float sp = speed.data[gd + a];
            #pragma unroll
            for (int c = 0; c < 6; ++c) v[c] += S[(gd + a) * 6 + c] * sp;
        }
    };
    float acc[6] = {0, 0, 0, 0, 0, 0};
    for (int k = lo; k < hi; ++k) link(k, acc);
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        float o[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) o[c] = __shfl_up_sync(FULL, acc[c], d);
        if (lane >= d) {
            #pragma unroll
            for (int c = 0; c < 6; ++c) acc[c] += o[c];
        }
    }
    float v[6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) {
        const float x = __shfl_up_sync(FULL, acc[c], 1);
        v[c] = lane > 0 ? x : 0.0f;
    }
    const wp::vec_t<3, wp::float32> origin = articulation_origin.data[art];
    for (int k = lo; k < hi; ++k) {
        link(k, v);
        const int b = meta[4 * (js + k)];
        const wp::vec_t<3, wp::float32> cx = wp::transform_point(body_q.data[b], body_com.data[b]);
        const float rx = cx[0] - origin[0], ry = cx[1] - origin[1], rz = cx[2] - origin[2];
        float* out = reinterpret_cast<float*>(body_qd_out.data) + b * 6;
        out[0] = v[0] + (v[4] * rz - v[5] * ry);
        out[1] = v[1] + (v[5] * rx - v[3] * rz);
        out[2] = v[2] + (v[3] * ry - v[4] * rx);
        out[3] = v[3];
        out[4] = v[4];
        out[5] = v[5];
    }
#endif
"""


@functools.cache
def chain_world_twists_kernel():
    """Kernel writing each chain-shaped rod articulation's world COM twists from its joint velocities."""

    @wp.func_native(_CHAIN_WORLD_TWISTS)
    def rod_chain_world_twists_native(
        narts: int,
        integrated: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_meta: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        speed: wp.array[float],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        articulation_origin: wp.array[wp.vec3],
        body_qd_out: wp.array[wp.spatial_vector],
    ): ...

    def rod_chain_world_twists(  # pragma: no cover - compiled by wp.kernel
        narts: int,
        integrated: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        joint_meta: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        speed: wp.array[float],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        articulation_origin: wp.array[wp.vec3],
        body_qd_out: wp.array[wp.spatial_vector],
    ):
        _tid = wp.tid()
        rod_chain_world_twists_native(
            narts,
            integrated,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            joint_meta,
            joint_S_s,
            speed,
            body_q,
            body_com,
            articulation_origin,
            body_qd_out,
        )

    rod_chain_world_twists.__name__ = "rod_chain_world_twists"
    rod_chain_world_twists.__qualname__ = "rod_chain_world_twists"
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(
        rod_chain_world_twists
    )


# Contracted chains: within one propagation solve a chain's factorization is fixed, so the twists of its
# contact-active links are an exact linear function of the impulses on those links alone. With the active links
# p_1 < ... < p_K in walk order, R_i the body response at p_i and P_i = B_{p_i + 1} ... B_{p_{i+1}} the product of
# the factor's backward maps between consecutive active links, the twist change at p_i is R_i a_i + b_i with
# a_i = f_i + P_i a_{i+1} (from the tip) and b_{i+1} = P_i^T (b_i + R_i f_i) (from the root); this is the
# same-articulation chain response of the refine pass extended to every pair. An iteration then costs O(K) small
# matrix-vector products per chain instead of a pass over all its links, and the full propagation of the summed
# impulses runs once after the iterations.

# Most active links a contracted chain may have; chains with more keep the full propagation every iteration.
CONTRACT_MAX = 32

_CONTRACT_PREP = r"""
#if defined(__CUDA_ARCH__)
    #define LONG_SPAN 8
    // One warp per chain: list the active links in walk order (ballot compaction) and compose P_i.
    const int lane = threadIdx.x & 31;
    const int tree = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    if (chain_flag.data[art] == 0) return;
    const int js = articulation_joint_range.data[2 * art];
    const int n = articulation_joint_range.data[2 * art + 1] - js;
    const unsigned FULL = 0xffffffffu;
    int count = 0;
    if (articulation_active.data && articulation_active.data[art] == 0) {
        n_active.data[art] = 0;
        return;
    }
    for (int base = 0; base < n; base += 32) {
        const int k = base + lane;
        int child = -1;
        bool act = false;
        if (k < n) {
            child = joint_meta.data[4 * (js + k)];
            act = body_seen.data[child] != 0;
        }
        const unsigned m = __ballot_sync(FULL, act);
        if (act) {
            const int i = count + __popc(m & ((1u << lane) - 1u));
            active_pos.data[js + i] = k;
            active_body.data[js + i] = child;
        }
        count += __popc(m);
    }
    if (lane == 0) n_active.data[art] = count;
    __syncwarp(FULL);
    const float* BR = propagation_tree_CR.data;
    auto matmul_b = [&](float* M, int k) {
        const float* br = BR + (js + k) * 148;
        float nm[36];
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float v = 0.0f;
                #pragma unroll
                for (int t = 0; t < 6; ++t) v += M[r * 6 + t] * br[t * 6 + c];
                nm[r * 6 + c] = v;
            }
        #pragma unroll
        for (int e = 0; e < 36; ++e) M[e] = nm[e];
    };
    // Long spans: the whole warp composes each, lane chunks first, then an ordered tree reduction.
    for (int i = 0; i + 1 < count; ++i) {
        const int p = active_pos.data[js + i];
        const int q = active_pos.data[js + i + 1];
        if (q - p <= LONG_SPAN) continue;
        const int m = q - p;
        const int per = (m + 31) / 32;
        float M[36];
        #pragma unroll
        for (int e = 0; e < 36; ++e) M[e] = (e % 7 == 0) ? 1.0f : 0.0f;
        for (int k = p + 1 + lane * per; k <= q && k < p + 1 + (lane + 1) * per; ++k) matmul_b(M, k);
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            float Ms[36], nm[36];
            #pragma unroll
            for (int e = 0; e < 36; ++e) Ms[e] = __shfl_down_sync(FULL, M[e], d);
            if ((lane & (2 * d - 1)) == 0) {
                #pragma unroll
                for (int r = 0; r < 6; ++r)
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) {
                        float v = 0.0f;
                        #pragma unroll
                        for (int t = 0; t < 6; ++t) v += M[r * 6 + t] * Ms[t * 6 + c];
                        nm[r * 6 + c] = v;
                    }
                #pragma unroll
                for (int e = 0; e < 36; ++e) M[e] = nm[e];
            }
        }
        if (lane == 0) {
            float4* out = reinterpret_cast<float4*>(active_P.data + (js + i) * 36);
            #pragma unroll
            for (int q4 = 0; q4 < 9; ++q4) out[q4] = make_float4(M[4 * q4], M[4 * q4 + 1], M[4 * q4 + 2], M[4 * q4 + 3]);
        }
    }
    // Short spans: one lane each.
    for (int i = lane; i + 1 < count; i += 32) {
        const int p = active_pos.data[js + i];
        const int q = active_pos.data[js + i + 1];
        if (q - p > LONG_SPAN) continue;
        float M[36];
        #pragma unroll
        for (int e = 0; e < 36; ++e) M[e] = (e % 7 == 0) ? 1.0f : 0.0f;
        for (int k = p + 1; k <= q; ++k) {
            const float* br = BR + (js + k) * 148;
            float nm[36];
            #pragma unroll
            for (int r = 0; r < 6; ++r)
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < 6; ++t) v += M[r * 6 + t] * br[t * 6 + c];
                    nm[r * 6 + c] = v;
                }
            #pragma unroll
            for (int e = 0; e < 36; ++e) M[e] = nm[e];
        }
        float4* out = reinterpret_cast<float4*>(active_P.data + (js + i) * 36);
        #pragma unroll
        for (int q4 = 0; q4 < 9; ++q4) out[q4] = make_float4(M[4 * q4], M[4 * q4 + 1], M[4 * q4 + 2], M[4 * q4 + 3]);
    }
#endif
"""

_CONTRACT_UPDATE = (
    "\n    #define KMAX "
    + str(CONTRACT_MAX)
    + r"""
#if defined(__CUDA_ARCH__)
    // One warp per chain with 1..KMAX active links (chains with more keep the full propagation every iteration).
    // snapshot != 0 keeps the active twists before the first sweep; otherwise the active twists become the
    // snapshot plus the exact response to the impulses summed so far, on lane 0 over values staged in shared memory.
    const int lane = threadIdx.x & 31;
    const int tree = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tree >= ntrees) return;
    const int art = group_to_art.data[tree];
    if (chain_flag.data[art] == 0) return;
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int K = n_active.data[art];
    const int js = articulation_joint_range.data[2 * art];
    const int nlinks = articulation_joint_range.data[2 * art + 1] - js;
    if (K <= 0 || K > KMAX || 4 * K > nlinks) return;
    const unsigned FULL = 0xffffffffu;
    float* QD = propagation_body_qd.data;
    if (snapshot) {
        for (int e = lane; e < K * 6; e += 32) {
            const int b = active_body.data[js + e / 6];
            active_qd0.data[js * 6 + e] = QD[b * 6 + e % 6];
        }
        return;
    }
    __shared__ __align__(16) float s_P[KMAX * 36];
    __shared__ __align__(16) float s_R[KMAX * 36];
    __shared__ float s_f[KMAX * 6];
    __shared__ float s_a[KMAX * 6];
    __shared__ float s_d[KMAX * 6];
    __shared__ int s_b[KMAX];
    for (int i = lane; i < K; i += 32) s_b[i] = active_body.data[js + i];
    __syncwarp(FULL);
    for (int e = lane; e < K * 36; e += 32) {
        s_P[e] = active_P.data[js * 36 + e];
        s_R[e] = propagation_body_response.data[s_b[e / 36] * 36 + e % 36];
    }
    for (int e = lane; e < K * 6; e += 32) s_f[e] = propagation_body_impulses.data[s_b[e / 6] * 6 + e % 6];
    __syncwarp(FULL);
    if (lane == 0) {
        // a_i = f_i + P_i a_{i+1}, from the tip.
        float a[6];
        #pragma unroll
        for (int r = 0; r < 6; ++r) { a[r] = s_f[(K - 1) * 6 + r]; s_a[(K - 1) * 6 + r] = a[r]; }
        for (int i = K - 2; i >= 0; --i) {
            const float* P = s_P + i * 36;
            float na[6];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                float v = s_f[i * 6 + r];
                #pragma unroll
                for (int c = 0; c < 6; ++c) v += P[r * 6 + c] * a[c];
                na[r] = v;
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) { a[r] = na[r]; s_a[i * 6 + r] = na[r]; }
        }
        // Twist changes: R_i a_i + b_i with b_{i+1} = P_i^T (b_i + R_i f_i), from the root.
        float bv[6] = {0, 0, 0, 0, 0, 0};
        for (int i = 0; i < K; ++i) {
            const float* R = s_R + i * 36;
            float t[6];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                float d = bv[r], rf = bv[r];
                #pragma unroll
                for (int c = 0; c < 6; ++c) { d += R[r * 6 + c] * s_a[i * 6 + c]; rf += R[r * 6 + c] * s_f[i * 6 + c]; }
                s_d[i * 6 + r] = d;
                t[r] = rf;
            }
            if (i + 1 < K) {
                const float* P = s_P + i * 36;
                #pragma unroll
                for (int r = 0; r < 6; ++r) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 6; ++c) v += P[c * 6 + r] * t[c];
                    bv[r] = v;
                }
            }
        }
    }
    __syncwarp(FULL);
    for (int e = lane; e < K * 6; e += 32) QD[s_b[e / 6] * 6 + e % 6] = active_qd0.data[js * 6 + e] + s_d[e];
#endif
"""
)


@functools.cache
def contract_prep_kernel():
    """Kernel listing each chain's active links and composing the maps between them (see ``_CONTRACT_PREP``)."""

    @wp.func_native(_CONTRACT_PREP)
    def rod_contract_prep_native(
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        body_seen: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        n_active: wp.array[int],
        active_pos: wp.array[int],
        active_body: wp.array[int],
        active_P: wp.array[float],
    ): ...

    def rod_contract_prep(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        joint_meta: wp.array[int],
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        body_seen: wp.array[int],
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
        n_active: wp.array[int],
        active_pos: wp.array[int],
        active_body: wp.array[int],
        active_P: wp.array[float],
    ):
        _tid = wp.tid()
        rod_contract_prep_native(
            ntrees,
            joint_meta,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            body_seen,
            propagation_tree_BR,
            propagation_tree_CR,
            n_active,
            active_pos,
            active_body,
            active_P,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_contract_prep)


@functools.cache
def contract_update_kernel():
    """Kernel applying each chain's contracted response to its active links (see ``_CONTRACT_UPDATE``)."""

    @wp.func_native(_CONTRACT_UPDATE)
    def rod_contract_update_native(
        ntrees: int,
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        n_active: wp.array[int],
        active_body: wp.array[int],
        active_P: wp.array[float],
        propagation_body_response: wp.array3d[float],
        propagation_body_impulses: wp.array2d[float],
        snapshot: int,
        active_qd0: wp.array[float],
        active_a: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ): ...

    def rod_contract_update(  # pragma: no cover - compiled by wp.kernel
        ntrees: int,
        group_to_art: wp.array[int],
        chain_flag: wp.array[int],
        articulation_active: wp.array[int],
        articulation_joint_range: wp.array[int],
        n_active: wp.array[int],
        active_body: wp.array[int],
        active_P: wp.array[float],
        propagation_body_response: wp.array3d[float],
        propagation_body_impulses: wp.array2d[float],
        snapshot: int,
        active_qd0: wp.array[float],
        active_a: wp.array[float],
        propagation_body_qd: wp.array2d[float],
    ):
        _tid = wp.tid()
        rod_contract_update_native(
            ntrees,
            group_to_art,
            chain_flag,
            articulation_active,
            articulation_joint_range,
            n_active,
            active_body,
            active_P,
            propagation_body_response,
            propagation_body_impulses,
            snapshot,
            active_qd0,
            active_a,
            propagation_body_qd,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_contract_update)


# Factor epilogue: the propagation records (BR, FR, CR) of every rod joint, six lanes per joint, from the U and D^-1
# the factor stored; the factor itself then keeps only the recursion. Same arithmetic as the factor's own records.
_FACTOR_EPILOGUE = (
    "\n#if defined(__CUDA_ARCH__)\n"
    + _PRELUDE.replace(
        "const int tree = warp_global * TREES_PER_WARP + grp;", "const int tree = warp_global * 5 + grp;"
    )
    + r"""
    if (grp >= 5) return;
    const int slot = tree;
    if (slot >= nslots) return;
    const int k = walk_slots.data[slot];
    const int art = slot_art.data[slot];
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    __shared__ float eshm[2][5][4][36];
    float* X = eshm[wib][grp][0];
    float* Y = eshm[wib][grp][1];
    float* Z = eshm[wib][grp][2];
    float* V = eshm[wib][grp][3];
    const int child = joint_meta.data[4 * k + 0];
    const int parent = joint_meta.data[4 * k + 1];
    const int gd = joint_meta.data[4 * k + 2];
    const int flags = joint_meta.data[4 * k + 3];
    const int dc = flags & 7;
    const int jid = flags >> 8;
    const float* SF = propagation_joint_S_flat.data;
    const float* CR = propagation_body_com_rel.data;
    float s_row[6], u_row[6];
    #pragma unroll
    for (int a = 0; a < 6; ++a) {
        s_row[a] = (a < dc) ? SF[(gd + a) * 6 + r] : 0.0f;
        u_row[a] = (a < dc) ? propagation_tree_U.data[(gd + a) * 6 + r] : 0.0f;
    }
    const float e0 = (parent >= 0) ? CR[child * 3 + 0] - CR[parent * 3 + 0] : 0.0f;
    const float e1 = (parent >= 0) ? CR[child * 3 + 1] - CR[parent * 3 + 1] : 0.0f;
    const float e2 = (parent >= 0) ? CR[child * 3 + 2] - CR[parent * 3 + 2] : 0.0f;
    #pragma unroll
    for (int a = 0; a < 6; ++a) X[r * 6 + a] = s_row[a];
    #pragma unroll
    for (int a = 0; a < 6; ++a) Y[r * 6 + a] = u_row[a];
    #pragma unroll
    for (int b = 0; b < 6; ++b)
        Z[r * 6 + b] = (r < dc && b < dc) ? propagation_tree_D_inv.data[jid * 36 + r * 6 + b] : 0.0f;
    __syncwarp(GMASK);
"""
    + r"""
        // The propagation pass's records. B = T (I - U D^-1 S^T) carries a child's wrench to its parent (T moves
        // it to the parent's center of mass); its transpose (I - S D^-1 U^T) X carries the parent's twist change
        // to the child. W = -D^-1 S^T gives the joint's w = D^-1 u from the wrench and Q = -D^-1 U^T X its
        // acceleration from the parent's twist change. Lane r's backward record is [B row r, W row r]; its
        // forward record is [B column r, Q row r, S row r, e, 0]: contiguous, so the walk loads them as float4s.
        {
            float wv[6];
            #pragma unroll
            for (int b = 0; b < 6; ++b) {
                float v = 0.0f;
                #pragma unroll
                for (int a = 0; a < 6; ++a) v += u_row[a] * Z[a * 6 + b];
                wv[b] = v;
            }
            const float ex = e0, ey = e1, ez = e2;
            float brow[6];
            #pragma unroll
            for (int col = 0; col < 6; ++col) {
                float p = 0.0f;
                #pragma unroll
                for (int b = 0; b < 6; ++b) p += wv[b] * X[col * 6 + b];
                const float bv = ((col == r) ? 1.0f : 0.0f) - p;
                const float b0 = bcast(bv, 0), b1 = bcast(bv, 1), b2 = bcast(bv, 2);
                float v = bv;
                if (r == 3) v += ey * b2 - ez * b1;
                if (r == 4) v += ez * b0 - ex * b2;
                if (r == 5) v += ex * b1 - ey * b0;
                brow[col] = v;
            }
            float wrow[6], qrow[6];
            {
                float g[6];
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    float v = 0.0f, t = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 6; ++b) {
                        v += Z[r * 6 + b] * Y[j * 6 + b];
                        t += Z[r * 6 + b] * X[j * 6 + b];
                    }
                    g[j] = v;
                    wrow[j] = -t;
                }
                qrow[0] = -g[0];
                qrow[1] = -g[1];
                qrow[2] = -g[2];
                qrow[3] = -(g[3] - ez * g[1] + ey * g[2]);
                qrow[4] = -(g[4] + ez * g[0] - ex * g[2]);
                qrow[5] = -(g[5] - ey * g[0] + ex * g[1]);
            }
            #pragma unroll
            for (int col = 0; col < 6; ++col) V[r * 6 + col] = brow[col];
            __syncwarp(GMASK);
            float rec[24];
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                rec[j] = V[j * 6 + r];
                rec[6 + j] = qrow[j];
                rec[12 + j] = s_row[j];
            }
            rec[18] = ex; rec[19] = ey; rec[20] = ez; rec[21] = 0.0f; rec[22] = 0.0f; rec[23] = 0.0f;
            // The walk kernels' records; chain-only groups read the compact record alone.
            if (write_walk_records) {
            float4* fr = reinterpret_cast<float4*>(propagation_tree_FR.data + k * 144 + r * 24);
            #pragma unroll
            for (int q = 0; q < 6; ++q) fr[q] = make_float4(rec[4 * q], rec[4 * q + 1], rec[4 * q + 2], rec[4 * q + 3]);
            float4* br = reinterpret_cast<float4*>(propagation_tree_BR.data + k * 72 + r * 12);
            br[0] = make_float4(brow[0], brow[1], brow[2], brow[3]);
            br[1] = make_float4(brow[4], brow[5], wrow[0], wrow[1]);
            br[2] = make_float4(wrow[2], wrow[3], wrow[4], wrow[5]);
            }
            // The chain passes' compact record (see CHAIN_REC): B, W, S and Q row-major, then e.
            float* cr = propagation_tree_CR.data + k * CHAIN_REC;
            float2* c2 = reinterpret_cast<float2*>(cr + r * 6);
            c2[0] = make_float2(brow[0], brow[1]); c2[1] = make_float2(brow[2], brow[3]); c2[2] = make_float2(brow[4], brow[5]);
            c2 = reinterpret_cast<float2*>(cr + 36 + r * 6);
            c2[0] = make_float2(wrow[0], wrow[1]); c2[1] = make_float2(wrow[2], wrow[3]); c2[2] = make_float2(wrow[4], wrow[5]);
            c2 = reinterpret_cast<float2*>(cr + 72 + r * 6);
            c2[0] = make_float2(s_row[0], s_row[1]); c2[1] = make_float2(s_row[2], s_row[3]); c2[2] = make_float2(s_row[4], s_row[5]);
            c2 = reinterpret_cast<float2*>(cr + 108 + r * 6);
            c2[0] = make_float2(qrow[0], qrow[1]); c2[1] = make_float2(qrow[2], qrow[3]); c2[2] = make_float2(qrow[4], qrow[5]);
            if (r == 0) *reinterpret_cast<float4*>(cr + 144) = make_float4(ex, ey, ez, 0.0f);
            __syncwarp(GMASK);
        }

"""
    + r"""
#endif
"""
)


# Factor prelude: everything a rod joint's step of the factorization needs that does not depend on its children,
# one thread per joint, so the serial walk keeps only the recursion. Per walk slot k: the child's inertia in world
# axes (IW, 36), the joint's diagonal terms (armature and implicit drive stiffness, KD, 6) and, for a full-rank
# joint, S^-1 (SINV, 36): with S invertible, I_hat^-1 U^T = S^-1, so the reduced inertia U D^-1 K I_hat^-1 U^T is
# U D^-1 K S^-1 and needs no second inversion.
_FACTOR_PRELUDE = r"""
#if defined(__CUDA_ARCH__)
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= nslots) return;
    const int k = walk_slots.data[slot];
    const int art = slot_art.data[slot];
    if (articulation_active.data && articulation_active.data[art] == 0) return;
    const int child = joint_meta.data[4 * k + 0];
    const int gd = joint_meta.data[4 * k + 2];
    const int flags = joint_meta.data[4 * k + 3];
    const int dc = flags & 7;
    const float* SF = propagation_joint_S_flat.data;
    const wp::mat_t<3, 3, wp::float32> R = wp::quat_to_matrix(wp::quat_inverse(body_q_com.data[child].q));
    const wp::mat_t<6, 6, wp::float32>& L = body_I_m.data[child];
    float* iw = IW.data + k * 36;
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        const int rb = r >= 3 ? 3 : 0;
        const int rr = r - rb;
        #pragma unroll
        for (int col = 0; col < 6; ++col) {
            const int cb = col >= 3 ? 3 : 0;
            const int cc = col - cb;
            float v = 0.0f;
            #pragma unroll
            for (int a = 0; a < 3; ++a)
                #pragma unroll
                for (int b = 0; b < 3; ++b)
                    v += R.data[a][rr] * L.data[rb + a][cb + b] * R.data[b][cc];
            iw[r * 6 + col] = v;
        }
    }
    const int aug_count = aug_row_counts.data[art];
    for (int r = 0; r < 6; ++r) {
        float diag = 0.0f;
        if (r < dc) {
            diag = joint_armature.data[gd + r];
            for (int aug_i = 0; aug_i < aug_count; ++aug_i) {
                const int row_index = art * max_dofs + aug_i;
                if (aug_row_dof_index.data[row_index] == gd + r) {
                    const float Kd = aug_row_K.data[row_index];
                    if (Kd > 0.0f) diag += Kd;
                }
            }
        }
        KD.data[k * 6 + r] = diag;
    }
    float* sinv = SINV.data + k * 36;
    bool done = false;
    if (dc == 6) {
        // A rod (or free) joint's S is [[A, C], [0, A]] with A its axes in world (a rotation) and C the lever arms of
        // the angular columns: S^-1 = [[A^-1, -A^-1 C A^-1], [0, A^-1]].
        float lower = 0.0f;
        wp::mat_t<3, 3, wp::float32> A, C;
        for (int r = 0; r < 3; ++r)
            for (int a = 0; a < 3; ++a) {
                A.data[r][a] = SF[(gd + a) * 6 + r];
                C.data[r][a] = SF[(gd + 3 + a) * 6 + r];
                lower = fmaxf(lower, fabsf(SF[(gd + a) * 6 + 3 + r]));
                lower = fmaxf(lower, fabsf(SF[(gd + 3 + a) * 6 + 3 + r] - SF[(gd + a) * 6 + r]));
            }
        if (lower <= 1e-6f) {
            const wp::mat_t<3, 3, wp::float32> Ai = wp::inverse(A);
            const wp::mat_t<3, 3, wp::float32> T = wp::mul(wp::mul(Ai, C), Ai);
            for (int r = 0; r < 3; ++r)
                for (int a = 0; a < 3; ++a) {
                    sinv[r * 6 + a] = Ai.data[r][a];
                    sinv[r * 6 + 3 + a] = -T.data[r][a];
                    sinv[(3 + r) * 6 + a] = 0.0f;
                    sinv[(3 + r) * 6 + 3 + a] = Ai.data[r][a];
                }
            done = true;
        }
    }
    if (dc == 6 && !done) {
        // Any other full-rank S: Gauss-Jordan with static indices (registers, no pivot swaps).
        float m[36], inv[36];
        #pragma unroll
        for (int r = 0; r < 6; ++r)
            #pragma unroll
            for (int a = 0; a < 6; ++a) {
                m[r * 6 + a] = SF[(gd + a) * 6 + r];
                inv[r * 6 + a] = (r == a) ? 1.0f : 0.0f;
            }
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const float sc = 1.0f / m[c * 6 + c];
            #pragma unroll
            for (int e = 0; e < 6; ++e) { m[c * 6 + e] *= sc; inv[c * 6 + e] *= sc; }
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                if (r == c) continue;
                const float f = m[r * 6 + c];
                #pragma unroll
                for (int e = 0; e < 6; ++e) { m[r * 6 + e] -= f * m[c * 6 + e]; inv[r * 6 + e] -= f * inv[c * 6 + e]; }
            }
        }
        #pragma unroll
        for (int e = 0; e < 36; ++e) sinv[e] = inv[e];
    }
#endif
"""


@functools.cache
def factor_prelude_kernel():
    """Kernel forming each rod joint's child-independent factor terms (see ``_FACTOR_PRELUDE``)."""

    @wp.func_native(_FACTOR_PRELUDE)
    def rod_factor_prelude_native(
        nslots: int,
        walk_slots: wp.array[int],
        slot_art: wp.array[int],
        articulation_active: wp.array[int],
        joint_meta: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        IW: wp.array[float],
        KD: wp.array[float],
        SINV: wp.array[float],
    ): ...

    def rod_factor_prelude(  # pragma: no cover - compiled by wp.kernel
        nslots: int,
        walk_slots: wp.array[int],
        slot_art: wp.array[int],
        articulation_active: wp.array[int],
        joint_meta: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        joint_armature: wp.array[float],
        max_dofs: int,
        aug_row_counts: wp.array[int],
        aug_row_dof_index: wp.array[int],
        aug_row_K: wp.array[float],
        body_I_m: wp.array[wp.spatial_matrix],
        body_q_com: wp.array[wp.transform],
        IW: wp.array[float],
        KD: wp.array[float],
        SINV: wp.array[float],
    ):
        _tid = wp.tid()
        rod_factor_prelude_native(
            nslots,
            walk_slots,
            slot_art,
            articulation_active,
            joint_meta,
            propagation_joint_S_flat,
            joint_armature,
            max_dofs,
            aug_row_counts,
            aug_row_dof_index,
            aug_row_K,
            body_I_m,
            body_q_com,
            IW,
            KD,
            SINV,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_factor_prelude)


@functools.cache
def factor_epilogue_kernel():
    """Kernel writing every rod joint's propagation records from the factor's U and D^-1 (see ``_FACTOR_EPILOGUE``)."""

    @wp.func_native(_FACTOR_EPILOGUE)
    def rod_factor_epilogue_native(
        nslots: int,
        walk_slots: wp.array[int],
        slot_art: wp.array[int],
        articulation_active: wp.array[int],
        joint_meta: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        write_walk_records: int,
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
    ): ...

    def rod_factor_epilogue(  # pragma: no cover - compiled by wp.kernel
        nslots: int,
        walk_slots: wp.array[int],
        slot_art: wp.array[int],
        articulation_active: wp.array[int],
        joint_meta: wp.array[int],
        propagation_joint_S_flat: wp.array2d[float],
        propagation_body_com_rel: wp.array2d[float],
        propagation_tree_U: wp.array2d[float],
        propagation_tree_D_inv: wp.array3d[float],
        write_walk_records: int,
        propagation_tree_BR: wp.array2d[float],
        propagation_tree_FR: wp.array2d[float],
        propagation_tree_CR: wp.array[float],
    ):
        _tid = wp.tid()
        rod_factor_epilogue_native(
            nslots,
            walk_slots,
            slot_art,
            articulation_active,
            joint_meta,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            write_walk_records,
            propagation_tree_BR,
            propagation_tree_FR,
            propagation_tree_CR,
        )

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(rod_factor_epilogue)


def epilogue_launch_dim(nslots: int) -> int:
    """Threads for ``nslots`` joints: five six-lane groups per warp, rounded to whole blocks."""
    warps = (nslots + 4) // 5
    return ((warps * 32 + BLOCK_DIM - 1) // BLOCK_DIM) * BLOCK_DIM
