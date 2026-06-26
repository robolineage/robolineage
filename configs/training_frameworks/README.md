# Training Framework Profiles

RoboLineage does not require a training repository to import RoboLineage code.
Framework profiles describe how accepted rollouts are staged, converted, trained,
evaluated, and located after training. The Framework Discovery Agent can generate
these profiles from a repository tree and the operator's normal commands; the
examples here are sanitized templates for common workflow families used in the
paper.

- `act_hdf5.example.yaml`: ACT-style action chunking with HDF5 episodes.
- `diffusion_policy.example.yaml`: Diffusion Policy-style dataset and checkpoint commands.
- `lerobot_vla.example.yaml`: LeRobot/VLA-style manifest export and fine-tuning.

Replace paths, commands, and output globs with the target lab repository values.
