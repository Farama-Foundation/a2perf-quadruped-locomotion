import gin
import gymnasium as gym
import pkg_resources

config_file_path = pkg_resources.resource_filename(
    "a2perf", "domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin"
)
gin.parse_config_files_and_bindings([config_file_path], None, finalize_config=False)

motion_files = {
    "pace": pkg_resources.resource_filename(
        "a2perf",
        "domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt",
    ),
    "trot": pkg_resources.resource_filename(
        "a2perf",
        "domains/quadruped_locomotion/motion_imitation/data/motions/dog_trot.txt",
    ),
    "spin": pkg_resources.resource_filename(
        "a2perf",
        "domains/quadruped_locomotion/motion_imitation/data/motions/dog_spin.txt",
    ),
}


def register_env(gait, motion_file):
    env_id = f"QuadrupedLocomotion-Dog{gait.capitalize()}-v0"

    gym.envs.register(
        id=env_id,
        apply_api_compatibility=True,
        entry_point="a2perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env",
        kwargs={
            "motion_files": [motion_file],
            "enable_rendering": False,
            "mode": "train",
            "num_parallel_envs": 1,
        },
    )


for gait, motion_file in motion_files.items():
    register_env(gait, motion_file)
