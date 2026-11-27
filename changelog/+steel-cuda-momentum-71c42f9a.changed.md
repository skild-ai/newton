Enable Nesterov momentum by default on CUDA in `SolverSteel`. At a fixed iteration count, CUDA trajectories can change.
CPU uses plain sweeps. Pass `nesterov_momentum=False` on CUDA to compare the plain sweep across devices.
Both settings use the selected row schedule, factorization and solver. Check both accuracy and step time before reducing the sweep count.

In a fixed-sweep RTX PRO 6000 Blackwell benchmark, time per substep changed by −6.6% for a serial 36-box pile, −2.0% and +1.6% for colored 36- and 27-box piles, +0.8% for Ant, and +7.7% and +4.7% for serial and colored pinch grasps. The humanoid changed by +0.8% with row-capacity overflow in both runs. The 36-box piles settled with less residual kinetic energy. The 27-box colored pile retained more energy and its deepest penetration increased from 0.056 to 0.065 mm. [GPU measurements](https://github.com/skild-ai/newton/pull/55#pullrequestreview-6082877357)

A separate capacity-checked benchmark on the same GPU model measured +7.6% for a serial 36-box pile and +8.1% and +7.5% for colored 36- and 27-box piles at 12 sweeps. In a fused eight-sphere tower, 10 accelerated sweeps reduced the maximum normal-velocity residual to 0.0373 m/s, below the 0.0442 m/s measured with 16 plain sweeps.
