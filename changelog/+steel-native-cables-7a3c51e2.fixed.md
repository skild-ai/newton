Fix the Franka cable examples passing their IK joint limits once per world, which ran the IK solve out of shared memory at 1,024 worlds.
