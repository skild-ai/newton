# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Latency-bound Gauss-Seidel sweep of dense rows, one contact per update.

The fused matrix-free sweep visits a contact as three dependent row updates.
The normal row and the first tangent each wait for a warp reduction, and the
second tangent's response is loaded from global memory inside that chain.
With one articulation spanning a whole cable, every contact conflicts with
every other contact, so a world is a single warp and that chain is its cost.

This sweep visits each contact once. Its normal and tangent rows share one
warp reduction. The tangent residuals pick up the normal update through the
cached products ``J_t Y_n``. All three rows update the velocity together.
The arithmetic is the serial Gauss-Seidel step in exact arithmetic, in the
same row order. The rows of the next contacts are prefetched into registers,
wrapping across sweeps because a launch's rows do not change.
"""

import functools

import warp as wp

from .friction import FRICTION_PAIR_CUDA
from .friction_patches import WARP_DETERMINISM_OFF
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)

wp.set_module_options(WARP_DETERMINISM_OFF)

# Per first-tangent row: scale, largest, smallest, vx, vy of the pair operator, J_t1 Y_n, J_t2 Y_n.
OP_WIDTH = 8

_SINGLE, _CONTACT, _PAIR = 0, 1, 2


@functools.cache
def get_prepare_kernel(max_rows: int, dofs: int):
    """Return the kernel that caches each friction pair's operator and its coupling to the normal row."""
    M, D = int(max_rows), int(dofs)
    snippet = f"""
#if defined(__CUDA_ARCH__)
{FRICTION_PAIR_CUDA}
    int n = count.data[world];
    if (n > {M}) n = {M};
    if (row >= n || row + 1 >= n) return;
    const int base = world * {M} + row;
    const int parent = row_parent.data[base];
    if (row_type.data[base] != {PGS_CONSTRAINT_TYPE_FRICTION} || parent + 1 != row) return;
    const bool normal = parent >= 0 && row_type.data[world * {M} + parent] == {PGS_CONSTRAINT_TYPE_CONTACT};
    const float* Jt1 = J.data + (size_t)base * {D};
    const float* Jt2 = Jt1 + {D};
    const float* Yt2 = Y.data + (size_t)(base + 1) * {D};
    const float* Yn = Y.data + (size_t)(world * {M} + (parent >= 0 ? parent : 0)) * {D};
    float c = 0.0f, k1 = 0.0f, k2 = 0.0f;
    for (int d = lane; d < {D}; d += 32) {{
        const float j1 = Jt1[d];
        c += j1 * Yt2[d];
        if (normal) {{
            const float yn = Yn[d];
            k1 += j1 * yn;
            k2 += Jt2[d] * yn;
        }}
    }}
    for (int s = 16; s > 0; s >>= 1) {{
        c += __shfl_xor_sync(0xFFFFFFFFu, c, s);
        k1 += __shfl_xor_sync(0xFFFFFFFFu, k1, s);
        k2 += __shfl_xor_sync(0xFFFFFFFFu, k2, s);
    }}
    if (lane == 0) {{
        float scale, largest, smallest, vx, vy;
        friction_pair_setup(diag.data[base], c, diag.data[base + 1], scale, largest, smallest, vx, vy);
        float* o = op.data + (size_t)base * {OP_WIDTH};
        o[0] = scale; o[1] = largest; o[2] = smallest; o[3] = vx; o[4] = vy; o[5] = k1; o[6] = k2;
    }}
#endif
"""

    @wp.func_native(snippet)
    def prepare_native(
        world: int,
        row: int,
        lane: int,
        count: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        diag: wp.array2d[float],
        J: wp.array3d[float],
        Y: wp.array3d[float],
        op: wp.array3d[float],
    ): ...

    def prepare(
        count: wp.array[int],
        row_type: wp.array2d[int],
        row_parent: wp.array2d[int],
        diag: wp.array2d[float],
        J: wp.array3d[float],
        Y: wp.array3d[float],
        op: wp.array3d[float],
    ):
        world, row, lane = wp.tid()
        prepare_native(world, row, lane, count, row_type, row_parent, diag, J, Y, op)

    name = f"contact_unit_prepare_{M}_{D}"
    prepare.__name__ = prepare.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(prepare)


def _sweep_snippet(M: int, D: int, depth: int) -> str:
    E = (D + 31) // 32
    sets = depth + 1

    def guard(k: int) -> str:
        return "" if 32 * (k + 1) <= D else f"if (lane + {32 * k} < {D}) "

    def decl(s: int) -> str:
        regs = ", ".join(f"J{s}_{r}_{k} = 0.0f, Y{s}_{r}_{k} = 0.0f" for r in range(3) for k in range(E))
        return (
            f"    float {regs};\n"
            f"    int kind{s} = 0, row{s} = 0, type{s} = 0;\n"
            f"    float diag{s} = 0.0f, rhs{s}_0 = 0.0f, rhs{s}_1 = 0.0f, rhs{s}_2 = 0.0f, w{s} = 1.0f, mu{s} = 0.0f;\n"
            f"    float op{s}_0 = 1.0f, op{s}_1 = 0.0f, op{s}_2 = 0.0f, op{s}_3 = 1.0f, op{s}_4 = 0.0f,"
            f" op{s}_5 = 0.0f, op{s}_6 = 0.0f;\n"
        )

    def load(s: int, unit: str) -> str:
        rows = []
        for r in range(3):
            loads = " ".join(
                f"{guard(k)}{{ J{s}_{r}_{k} = Jr[{32 * k} + lane]; Y{s}_{r}_{k} = Yr[{32 * k} + lane]; }}"
                for k in range(E)
            )
            rows.append(
                f"            if (rows > {r}) {{ const float* Jr = J_world.data + jy + (size_t)(row{s} + {r}) * {D};"
                f" const float* Yr = Y_world.data + jy + (size_t)(row{s} + {r}) * {D}; {loads} }}"
            )
        rows = "\n".join(rows)
        return f"""        {{
            const int code = s_unit[{unit}];
            kind{s} = code >> 20;
            row{s} = code & 0xFFFFF;
            const int rows = kind{s} == {_CONTACT} ? 3 : (kind{s} == {_PAIR} ? 2 : 1);
{rows}
            const int b = off + row{s};
            type{s} = s_type[row{s}];
            diag{s} = world_diag.data[b];
            rhs{s}_0 = rhs_bias.data[b];
            w{s} = regularize != 0 ? world_row_w.data[b] : 1.0f;
            if (kind{s} != {_SINGLE}) {{
                rhs{s}_1 = rhs_bias.data[b + 1];
                rhs{s}_2 = kind{s} == {_CONTACT} ? rhs_bias.data[b + 2] : 0.0f;
                const int t = kind{s} == {_CONTACT} ? b + 1 : b;
                mu{s} = world_row_mu.data[t];
                const float* o = op.data + (size_t)t * {OP_WIDTH};
                op{s}_0 = o[0]; op{s}_1 = o[1]; op{s}_2 = o[2]; op{s}_3 = o[3]; op{s}_4 = o[4];
                op{s}_5 = o[5]; op{s}_6 = o[6];
            }}
        }}
"""

    def dot(s: int, r: int) -> str:
        return " ".join(f"{guard(k)}acc{r} += J{s}_{r}_{k} * s_v[lane + {32 * k}];" for k in range(E))

    def update(s: int, rows: int) -> str:
        terms = {
            1: "Y{s}_0_{k} * d0",
            2: "Y{s}_0_{k} * d0 + Y{s}_1_{k} * d1",
            3: "Y{s}_0_{k} * d0 + Y{s}_1_{k} * d1 + Y{s}_2_{k} * d2",
        }[rows]
        return " ".join(
            f"{guard(k)}s_v[lane + {32 * k}] += " + terms.replace("{s}", str(s)).replace("{k}", str(k)) + ";"
            for k in range(E)
        )

    def process(s: int) -> str:
        return f"""        if (kind{s} == {_CONTACT}) {{
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
            {dot(s, 0)}
            {dot(s, 1)}
            {dot(s, 2)}
            warp_sum3(acc0, acc1, acc2);
            const float old_n = s_lam[row{s}];
            float new_n = old_n;
            if (diag{s} > 0.0f) {{
                const float residual = acc0 + rhs{s}_0;
                const float delta = -residual / diag{s} * w{s} - (1.0f - w{s}) * old_n;
                new_n = fmaxf(old_n + omega * delta, 0.0f);
            }}
            const float d0 = new_n - old_n;
            float d1 = 0.0f, d2 = 0.0f;
            const float old_t1 = s_lam[row{s} + 1];
            const float old_t2 = s_lam[row{s} + 2];
            float new_t1 = 0.0f, new_t2 = 0.0f;
            if (friction) {{
                // A patch's normal rows form a cycle through their parent links and pool their load.
                float load = new_n;
                for (int p = s_parent[row{s}]; p >= 0 && p != row{s}; p = s_parent[p]) load += s_lam[p];
                const float radius = fmaxf(mu{s} * load, 0.0f);
                const float r0 = acc1 + op{s}_5 * d0 + rhs{s}_1;
                const float r1 = acc2 + op{s}_6 * d0 + rhs{s}_2;
                const float2 pair = friction_pair_solve(op{s}_0, op{s}_1, op{s}_2, op{s}_3, op{s}_4,
                    r0, r1, old_t1, old_t2, radius, omega);
                const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = mag > radius ? radius / mag : 1.0f;
                new_t1 = pair.x * scale;
                new_t2 = pair.y * scale;
                d1 = new_t1 - old_t1;
                d2 = new_t2 - old_t2;
            }}
            __syncwarp();
            if (lane == 0) {{
                s_lam[row{s}] = new_n;
                s_lam[row{s} + 1] = new_t1;
                s_lam[row{s} + 2] = new_t2;
            }}
            if (d0 != 0.0f || d1 != 0.0f || d2 != 0.0f) {{
                changed = 1;
                {update(s, 3)}
            }}
            __syncwarp();
        }} else if (kind{s} == {_PAIR}) {{
            if (friction) {{
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
                {dot(s, 0)}
                {dot(s, 1)}
                warp_sum3(acc0, acc1, acc2);
                const int normal = s_parent[row{s}];
                float load = s_lam[normal];
                for (int p = s_parent[normal]; p >= 0 && p != normal; p = s_parent[p]) load += s_lam[p];
                const float radius = fmaxf(mu{s} * load, 0.0f);
                const float old_t1 = s_lam[row{s}];
                const float old_t2 = s_lam[row{s} + 1];
                const float2 pair = friction_pair_solve(op{s}_0, op{s}_1, op{s}_2, op{s}_3, op{s}_4,
                    acc0 + rhs{s}_0, acc1 + rhs{s}_1, old_t1, old_t2, radius, omega);
                const float mag = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = mag > radius ? radius / mag : 1.0f;
                const float d0 = pair.x * scale - old_t1;
                const float d1 = pair.y * scale - old_t2;
                __syncwarp();
                if (lane == 0) {{
                    s_lam[row{s}] = pair.x * scale;
                    s_lam[row{s} + 1] = pair.y * scale;
                }}
                if (d0 != 0.0f || d1 != 0.0f) {{
                    changed = 1;
                    {update(s, 2)}
                }}
            }} else {{
                __syncwarp();
                if (lane == 0) {{
                    s_lam[row{s}] = 0.0f;
                    s_lam[row{s} + 1] = 0.0f;
                }}
            }}
            __syncwarp();
        }} else if (diag{s} > 0.0f) {{
            float acc0 = 0.0f;
            {dot(s, 0)}
            acc0 = warp_sum(acc0);
            const float old_i = s_lam[row{s}];
            const float residual = acc0 + rhs{s}_0;
            const float delta = -residual / diag{s} * w{s} - (1.0f - w{s}) * old_i;
            float new_i = old_i + omega * delta;
            if (new_i < 0.0f && (type{s} == {PGS_CONSTRAINT_TYPE_CONTACT} || type{s} == {PGS_CONSTRAINT_TYPE_JOINT_LIMIT}))
                new_i = 0.0f;
            const float d0 = new_i - old_i;
            __syncwarp();
            if (lane == 0) s_lam[row{s}] = new_i;
            if (d0 != 0.0f) {{
                changed = 1;
                {update(s, 1)}
            }}
            __syncwarp();
        }}
"""

    decls = "".join(decl(s) for s in range(sets))
    prologue = "".join(load(s, f"{s} % units") for s in range(sets))
    steps = []
    for s in range(sets):
        steps.append(f"""            {{
                if (step >= total) break;
                const int global_iter = iteration_offset + iter;
                const bool friction = global_iter >= friction_start_iteration;
{process(s)}
                int next = unit + {sets};
                while (next >= units) next -= units;
{load(s, "next")}
                ++step;
                if (++unit == units) {{
                    if (global_iter >= friction_start_iteration && __ballot_sync(0xFFFFFFFFu, changed != 0) == 0u) {{
                        step = total;
                        break;
                    }}
                    changed = 0;
                    unit = 0;
                    ++iter;
                }}
            }}
""")
    steps = "".join(steps)

    return f"""
#if defined(__CUDA_ARCH__)
{FRICTION_PAIR_CUDA}
    auto warp_sum = [](float value) {{
        for (int s = 16; s > 0; s >>= 1) value += __shfl_xor_sync(0xFFFFFFFFu, value, s);
        return value;
    }};
    auto warp_sum3 = [](float& a, float& b, float& c) {{
        for (int s = 16; s > 0; s >>= 1) {{
            a += __shfl_xor_sync(0xFFFFFFFFu, a, s);
            b += __shfl_xor_sync(0xFFFFFFFFu, b, s);
            c += __shfl_xor_sync(0xFFFFFFFFu, c, s);
        }}
    }};
    const int lane = threadIdx.x;
    int m = world_constraint_count.data[world];
    if (m <= 0) return;
    if (m > {M}) m = {M};
    const int off = world * {M};
    const size_t jy = (size_t)world * {M} * {D};
    const int dof_base = world * {D};

    __shared__ float s_v[{D}];
    __shared__ float s_lam[{M}];
    __shared__ unsigned char s_type[{M}];
    __shared__ short s_parent[{M}];
    __shared__ int s_unit[{2 * M}];
    __shared__ int s_units;

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_type[i] = (unsigned char)world_row_type.data[off + i];
        s_parent[i] = (short)world_row_parent.data[off + i];
    }}
    for (int d = lane; d < {D}; d += 32) {{
        const int g = world_dof_indices.data[dof_base + d];
        s_v[d] = g >= 0 ? v_out.data[g] : 0.0f;
    }}
    __syncwarp();
    if (lane == 0) {{
        // Units in row order: a contact normal with its tangent pair, a tangent pair, or one row.
        int u = 0;
        for (int i = 0; i < m; ++i) {{
            const int t = s_type[i];
            if (t == {PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT}) continue;
            if (t == {PGS_CONSTRAINT_TYPE_FRICTION}) {{
                if (i == s_parent[i] + 1 && i + 1 < m) s_unit[u++] = i | ({_PAIR} << 20);
                continue;
            }}
            if (t == {PGS_CONSTRAINT_TYPE_CONTACT} && i + 2 < m && s_type[i + 1] == {PGS_CONSTRAINT_TYPE_FRICTION}
                && s_parent[i + 1] == i) {{
                s_unit[u++] = i | ({_CONTACT} << 20);
                i += 2;
                continue;
            }}
            s_unit[u++] = i;
        }}
        // Contacts between two responding articulations (a jaw on a rod) are revisited after each pass.
        // A light segment squeezed between heavy bodies converges slowly under uniform sweeps.
        const int pass = u;
        for (int r = 0; r < coupled_passes; ++r)
            for (int v = 0; v < pass && u < {2 * M}; ++v) {{
                const int code = s_unit[v];
                const int row = code & 0xFFFFF;
                if ((code >> 20) != {_PAIR} && s_type[row] == {PGS_CONSTRAINT_TYPE_CONTACT}
                    && op.data[(size_t)(off + row) * {OP_WIDTH} + 7] != 0.0f)
                    s_unit[u++] = code;
            }}
        s_units = u;
    }}
    __syncwarp();
    const int units = s_units;
    if (units > 0 && iterations > 0) {{
{decls}
{prologue}
        const int total = iterations * units;
        int step = 0, unit = 0, iter = 0, changed = 0;
        while (step < total) {{
{steps}
        }}
    }}
    __syncwarp();
    for (int d = lane; d < {D}; d += 32) {{
        const int g = world_dof_indices.data[dof_base + d];
        if (g >= 0) v_out.data[g] = s_v[d];
    }}
    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
#endif
"""


@functools.cache
def get_sweep_kernel(max_rows: int, dofs: int, depth: int = 1, fast_math: bool = False):
    """Return the contact-unit sweep for ``max_rows`` dense rows over ``dofs`` world DOFs."""
    M, D = int(max_rows), int(dofs)
    snippet = _sweep_snippet(M, D, int(depth))

    @wp.func_native(snippet)
    def sweep_native(
        world: int,
        world_constraint_count: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_row_w: wp.array2d[float],
        op: wp.array3d[float],
        iterations: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        iteration_offset: int,
        coupled_passes: int,
        v_out: wp.array[float],
    ): ...

    def sweep(
        world_constraint_count: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_row_w: wp.array2d[float],
        op: wp.array3d[float],
        iterations: int,
        omega: float,
        regularize: int,
        friction_start_iteration: int,
        iteration_offset: int,
        coupled_passes: int,
        v_out: wp.array[float],
    ):
        world, _lane = wp.tid()
        sweep_native(
            world,
            world_constraint_count,
            world_dof_indices,
            rhs_bias,
            world_diag,
            world_impulses,
            J_world,
            Y_world,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_w,
            op,
            iterations,
            omega,
            regularize,
            friction_start_iteration,
            iteration_offset,
            coupled_passes,
            v_out,
        )

    name = f"contact_unit_sweep_{M}_{D}_d{int(depth)}" + ("_fm" if fast_math else "")
    sweep.__name__ = sweep.__qualname__ = name
    options = dict(WARP_DETERMINISM_OFF)
    if fast_math:
        options["fast_math"] = True
    return wp.kernel(enable_backward=False, module="unique", module_options=options)(sweep)


@wp.kernel(enable_backward=False)
def _flag_coupled_contacts(
    count: wp.array[int],
    path: wp.array[int],
    world: wp.array[int],
    slot: wp.array[int],
    art_a: wp.array[int],
    art_b: wp.array[int],
    response: wp.array[int],
    op: wp.array3d[float],
):
    c = wp.tid()
    if c >= count[0] or path[c] != 0 or slot[c] < 0:
        return
    a = art_a[c]
    b = art_b[c]
    coupled = a >= 0 and b >= 0 and a != b and response[a] > 0 and response[b] > 0
    op[world[c], slot[c], 7] = wp.where(coupled, 1.0, 0.0)


class ContactUnitSweep:
    """Dense-row Gauss-Seidel of one world per warp, visiting each contact as one update."""

    # Each lane keeps two contacts' rows in registers, six floats per 32 DOFs.
    MAX_DOFS = 256

    @staticmethod
    def fits(rows: int, dofs: int) -> bool:
        """Whether a world's rows and DOFs fit the sweep's registers and static shared memory."""
        # s_v, s_lam, s_type, s_parent and the doubled unit list, inside 48 KiB.
        shared = 4 * dofs + rows * (4 + 1 + 2 + 8) + 16
        return 0 < dofs <= ContactUnitSweep.MAX_DOFS and 0 < rows and shared <= 48 * 1024

    def __init__(self, solver, depth: int = 1, coupled_passes: int = 0):
        self.rows = int(solver._articulation_constraint_max)
        self.dofs = int(solver._max_world_dofs)
        self.device = solver.model.device
        self.op = wp.zeros((solver._world_count, self.rows, OP_WIDTH), dtype=float, device=self.device)
        self.prepare_kernel = get_prepare_kernel(self.rows, self.dofs)
        # Approximate division and square root change the sweep by about 1e-6 relative.
        self.sweep_kernel = get_sweep_kernel(self.rows, self.dofs, depth, fast_math=True)
        self.coupled_passes = int(coupled_passes)

    def flag_coupled(self, solver, contacts):
        """Mark the normal rows of contacts between two responding articulations for this step."""
        if contacts is None or not contacts.rigid_contact_max:
            return
        wp.launch(
            _flag_coupled_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                solver._contact_path,
                solver._contact_world,
                solver._contact_slot,
                solver._contact_art_a,
                solver._contact_art_b,
                solver._articulation_response_dof_count,
            ],
            outputs=[self.op],
            device=self.device,
        )

    def solve(self, solver, rhs, iterations, *, regularize, friction_start_iteration, iteration_offset):
        worlds = solver._world_count
        wp.launch(
            self.prepare_kernel,
            dim=(worlds, self.rows, 32),
            inputs=[
                solver._constraint_count,
                solver._row_type,
                solver._row_parent,
                solver._diag,
                solver._J_world,
                solver._Y_world,
            ],
            outputs=[self.op],
            block_dim=32,
            device=self.device,
        )
        wp.launch(
            self.sweep_kernel,
            dim=(worlds, 32),
            inputs=[
                solver._constraint_count,
                solver._world_dof_indices,
                rhs,
                solver._diag,
                solver._impulses,
                solver._J_world,
                solver._Y_world,
                solver._row_type,
                solver._row_parent,
                solver._row_mu,
                solver._row_w,
                self.op,
                int(iterations),
                float(solver._relaxation),
                int(regularize),
                int(friction_start_iteration),
                int(iteration_offset),
                self.coupled_passes,
            ],
            outputs=[solver._v_out],
            block_dim=32,
            device=self.device,
        )
