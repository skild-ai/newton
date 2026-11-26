Enable Nesterov momentum by default on CUDA in `SolverSteel`. At a fixed iteration count, CUDA trajectories can change.
CPU uses plain sweeps. Pass `nesterov_momentum=False` on CUDA to compare the plain sweep across devices.
Both settings use the selected row schedule, factorization and solver. Check both accuracy and step time before reducing the sweep count.

In a fixed-sweep RTX PRO 6000 Blackwell benchmark of the [reviewed revision](https://github.com/skild-ai/newton/pull/55#pullrequestreview-6066961057), time per substep changed by −0.8% for a serial 36-box pile, +22.8% and +11.4% for colored 36- and 27-box piles, +0.5% for Ant, and +2.5% and +4.8% for serial and colored pinch grasps. The humanoid changed by +0.4% with row-capacity overflow in both runs. The colored piles retained more residual kinetic energy at the same iteration count.
