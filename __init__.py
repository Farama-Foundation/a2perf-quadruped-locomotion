import gin
import gym as legacy_gym
import gymnasium as gym
import pkg_resources

# Construct the path to the config file using pkg_resources
config_file_path = pkg_resources.resource_filename(
    'a2perf',
    'domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin'
)

gin.parse_config_files_and_bindings([config_file_path], None,
                                    finalize_config=False)

# Construct the path to the motion file using pkg_resources
motion_file_path = pkg_resources.resource_filename(
    'a2perf',
    'domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt'
)

# Register the environments with the relative path
gym.envs.register(
    id='QuadrupedLocomotion-v0',
    apply_api_compatibility=True,
    entry_point='a2perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env',
    kwargs={
        'motion_files': [motion_file_path],
        'enable_rendering': False, 'mode': 'train'
    }
)

legacy_gym.envs.register(
    id='QuadrupedLocomotion-v0',
    entry_point='a2perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env',
    kwargs={
        'motion_files': [motion_file_path],
        'enable_rendering': False, 'mode': 'train'
    }
)
