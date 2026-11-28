# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU Cholesky for articulation factors exceeding a single CUDA tile."""

import functools
import inspect

import warp as wp

from .friction_patches import WARP_DETERMINISM_OFF

wp.set_module_options(WARP_DETERMINISM_OFF)

_PANEL = 32


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    constexpr int B = 32;
    __shared__ double tile[B][B];
    const int n = H.shape[1];
    const int m = min(B, n-base);
    const float* h = H.data + group*n*n;
    float* l = L.data + group*n*n;
    for (int k = lane; k < m*m; k += blockDim.x) {
        int i = k/m, j=k-i*m;
        if (j > i) continue;
        double value = h[(base+i)*n+base+j];
        if (i == j) value += R.data[group*n+base+i];
        for (int p=0; p<base; ++p) value -= static_cast<double>(l[(base+i)*n+p])*l[(base+j)*n+p];
        tile[i][j] = value;
    }
    __syncthreads();
    for (int j=0; j<m; ++j) {
        // A pivot that lost definiteness (or is NaN) is floored: the factor stays finite instead of turning NaN.
        if (!lane) tile[j][j] = sqrt(tile[j][j] > 1.0e-30 ? tile[j][j] : 1.0e-30);
        __syncthreads();
        if (lane>j && lane<m) tile[lane][j] /= tile[j][j];
        __syncthreads();
        for (int k=lane; k<m*m; k+=blockDim.x) {
            int a=k/m, b=k-a*m;
            if (a>=b && b>j) tile[a][b] -= tile[a][j]*tile[b][j];
        }
        __syncthreads();
    }
    for (int k=lane; k<m*m; k+=blockDim.x) {
        int i=k/m, j=k-i*m;
        if (j<=i) l[(base+i)*n+base+j]=static_cast<float>(tile[i][j]);
    }
#endif
""")
def _factor_tile(group: int, base: int, lane: int, H: wp.array3d[float], R: wp.array2d[float], L: wp.array3d[float]):
    """Factor the diagonal panel after subtracting earlier panel products."""


@wp.kernel(enable_backward=False)
def _diagonal(
    base: int,
    H: wp.array3d[float],
    R: wp.array2d[float],
    arts: wp.array[int],
    active: wp.array[int],
    L: wp.array3d[float],
):
    group, lane = wp.tid()
    if active[arts[group]] != 0:
        _factor_tile(group, base, lane, H, R, L)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    const int n=H.shape[1], j=base+lane;
    const float* h=H.data+group*n*n;
    float* l=L.data+group*n*n;
    double value=h[row*n+j];
    for (int p=0;p<base;++p) value-=static_cast<double>(l[row*n+p])*l[j*n+p];
    for (int p=0;p<32;++p) {
        if (p==lane) value/=l[j*n+j];
        double pivot=__shfl_sync(0xffffffff,value,p);
        if (lane>p) value-=pivot*l[j*n+base+p];
    }
    l[row*n+j]=static_cast<float>(value);
#endif
""")
def _factor_row(group: int, base: int, row: int, lane: int, H: wp.array3d[float], L: wp.array3d[float]):
    """Solve a trailing row against the current 32-column diagonal panel."""


@wp.kernel(enable_backward=False)
def _panel(base: int, H: wp.array3d[float], arts: wp.array[int], active: wp.array[int], L: wp.array3d[float]):
    group, row, lane = wp.tid()
    if active[arts[group]] != 0:
        _factor_row(group, base, base + 32 + row, lane, H, L)


_SHARED_THREADS = 512
# 512 threads leave 128 registers each; older Warp builds the kernel without the cap.
_REGISTER_CAP = {"cuda_max_registers": 128} if "cuda_max_registers" in inspect.signature(wp.kernel).parameters else {}
# Each thread keeps its entries of the packed lower triangle as float pairs in registers.
_SHARED_MAX_SIZE = 128


@functools.cache
def _shared_kernel(n: int):
    """Return a one-block right-looking Cholesky that accumulates each entry in compensated single precision.

    Every thread owns fixed entries of the packed lower triangle and keeps each as an unevaluated float sum
    ``hi + lo``. A rank-one update subtracts the exact product of two rounded factor entries (``fma``
    recovers its error) with ``TwoSum``. This reproduces the double accumulation of single-precision factor
    entries done by the panel kernels, in one launch.
    """
    tri = n * (n + 1) // 2
    per = (tri + _SHARED_THREADS - 1) // _SHARED_THREADS

    def unrolled(body: str) -> str:
        return "\n".join(body.replace("{e}", str(e)).replace("{tri}", str(tri)) for e in range(per))

    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int N = {n};
    constexpr int T = {_SHARED_THREADS};
    __shared__ float col[N];
    __shared__ float pivot;
    const int tid = threadIdx.x;
    const float* h = H.data + (size_t)group * N * N;
    const float* r = R.data + (size_t)group * N;
    float* l = L.data + (size_t)group * N * N;
    int ij[{per}];  // row << 16 | column, row -1 when the entry is outside the triangle or finished
    float hi[{per}], lo[{per}];
{
        unrolled('''    {
        const int t = tid + {e} * T;
        int i = -1, j = 0;
        if (t < {tri}) {
            i = (int)((sqrtf(8.0f * (float)t + 1.0f) - 1.0f) * 0.5f);
            while (i * (i + 1) / 2 > t) --i;
            while ((i + 1) * (i + 2) / 2 <= t) ++i;
            j = t - i * (i + 1) / 2;
        }
        ij[{e}] = (i << 16) | j; lo[{e}] = 0.0f; hi[{e}] = 0.0f;
        if (i >= 0) {
            hi[{e}] = h[i * N + j];
            if (i == j) {
                // The regularized diagonal h + r as an exact float pair.
                const float a = hi[{e}], b = r[i];
                const float sum = a + b, bb = sum - a;
                hi[{e}] = sum;
                lo[{e}] = (a - (sum - bb)) + (b - bb);
            }
        }
    }''')
    }
    for (int k = 0; k < N; ++k) {{
{unrolled('''        if (ij[{e}] == ((k << 16) | k)) pivot = hi[{e}] + lo[{e}];''')}
        __syncthreads();
        // A pivot that lost definiteness (or is NaN) is floored, so the column's reciprocal stays finite.
        const float diagonal = sqrtf(pivot > 1.0e-20f ? pivot : 1.0e-20f);
{
        unrolled('''        if (ij[{e}] >= 0 && (ij[{e}] & 0xFFFF) == k) {
            const int i = ij[{e}] >> 16;
            const float value = i == k ? diagonal : (hi[{e}] + lo[{e}]) / diagonal;
            col[i] = value;
            l[i * N + k] = value;
            ij[{e}] = -1;
        }''')
    }
        __syncthreads();
{
        unrolled('''        if (ij[{e}] >= 0 && (ij[{e}] & 0xFFFF) > k) {
            const float a = col[ij[{e}] >> 16], b = col[ij[{e}] & 0xFFFF];
            const float p = a * b;
            const float pe = fmaf(a, b, -p);
            const float sum = hi[{e}] - p;
            const float bb = sum - hi[{e}];
            const float err = (hi[{e}] - (sum - bb)) + (-p - bb);
            hi[{e}] = sum;
            lo[{e}] += err - pe;
        }''')
    }
    }}
#endif
"""

    @wp.func_native(snippet)
    def factor_native(group: int, H: wp.array3d[float], R: wp.array2d[float], L: wp.array3d[float]): ...

    def shared_factor(
        H: wp.array3d[float], R: wp.array2d[float], arts: wp.array[int], active: wp.array[int], L: wp.array3d[float]
    ):
        group, _thread = wp.tid()
        if active[arts[group]] != 0:
            factor_native(group, H, R, L)

    name = f"cholesky_shared_compensated_{n}"
    shared_factor.__name__ = shared_factor.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", **_REGISTER_CAP)(shared_factor)


def factor(H, R, arts, active, L, *, single_block=True):
    """Factor ``H + diag(R) = L L^T`` for every active group.

    Factors up to ``_SHARED_MAX_SIZE`` use one compensated single-precision launch; larger ones and
    ``single_block=False`` use the double-precision panels.
    """
    groups, size, _ = H.shape
    if single_block and size <= _SHARED_MAX_SIZE:
        wp.launch(
            _shared_kernel(size),
            dim=(groups, _SHARED_THREADS),
            inputs=[H, R, arts, active],
            outputs=[L],
            block_dim=_SHARED_THREADS,
            device=H.device,
        )
        return
    for base in range(0, size, _PANEL):
        wp.launch(
            _diagonal, dim=(groups, 128), inputs=[base, H, R, arts, active], outputs=[L], block_dim=128, device=H.device
        )
        trailing = size - base - _PANEL
        if trailing > 0:
            wp.launch(
                _panel,
                dim=(groups, trailing, _PANEL),
                inputs=[base, H, arts, active],
                outputs=[L],
                block_dim=_PANEL,
                device=H.device,
            )
