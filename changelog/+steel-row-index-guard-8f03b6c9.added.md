`SolverSteel` refuses a propagation row budget whose 32-bit row offsets would overflow over every world (rows per world times worlds times six), instead of corrupting the last worlds' rows.
