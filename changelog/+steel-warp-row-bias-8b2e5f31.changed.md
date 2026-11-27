Speed up `SolverSteel`'s dense-row bias and constraint-force-mixing passes by splitting each world's rows over a warp instead of one thread; results are bitwise unchanged. No migration is required.
