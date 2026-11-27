Add native cables to `SolverSteel`, which solves rod articulations together with arms and rigid bodies in one solve:

  - Rod material on CUDA: stretch, shear, bend, twist, intrinsic curvature, damping and Dahl hysteresis, folded into the articulated factorization; rings close through a rod joint.
  - `inextensible` on `ModelBuilder.add_joint_rod` and `ModelBuilder.add_rod`, for rod joints with bend and twist only.
  - Rods under both contact responses: dense generalized rows (`"immediate"`), or rows on the touched link propagated through the rod tree (`"propagation"`), with `propagation_interval`, `propagation_momentum` and `propagation_weighted_split`.
  - Contact warm starting from Steel's own contact matching in body frames, so collision pipelines need no contact matching.
  - Contact units, one record per contact in the colored propagation sweep, on by default from 256 worlds without loop closures, sized by `propagation_contact_max` and `propagation_same_articulation_contact_max`.
  - `coupled_contact_passes`, which revisits contacts between two responding articulations, such as a gripper jaw on a cable, in every iteration.
  - A construction-time error when a propagation row budget's 32-bit offsets would overflow over all worlds.
