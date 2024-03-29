import gin
import gym as legacy_gym
import gymnasium as gym
import pkg_resources

config_file_path = pkg_resources.resource_filename(
    "a2perf", "domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin"
)
gin.parse_config_files_and_bindings([config_file_path], None, finalize_config=False)
motion_file_path = pkg_resources.resource_filename(
    "a2perf", "domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt"
)

gym.envs.register(
    id="QuadrupedLocomotion-v0",
    apply_api_compatibility=True,
    entry_point="a2perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env",
    kwargs={
        "motion_files": [motion_file_path],
        "enable_rendering": False,
        "mode": "train",
        "num_parallel_envs": 1,
    },
)

legacy_gym.envs.register(
    id="QuadrupedLocomotion-v0",
    entry_point="a2perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env",
    kwargs={
        "motion_files": [motion_file_path],
        "enable_rendering": False,
        "mode": "train",
        "num_parallel_envs": 1,
    },
)
