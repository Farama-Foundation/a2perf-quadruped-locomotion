import gin
import gym as legacy_gym
import gymnasium as gym
import os

config_file_path = os.path.join(os.path.dirname(__file__), 'motion_imitation',
                                'configs', 'envdesign.gin')

gin.parse_config_files_and_bindings([config_file_path], None,
                                    finalize_config=False)

gym.envs.register(
    id='QuadrupedLocomotion-v0',
    apply_api_compatibility=True,
    entry_point='rl_perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env',
    kwargs={
        'motion_files': [
            "/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt"],
        'enable_rendering': False, 'mode': 'train'}
)

legacy_gym.envs.register(
    id='QuadrupedLocomotion-v0',
    entry_point='rl_perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env',
    kwargs={
        'motion_files': [
            "/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt"],
        'enable_rendering': False, 'mode': 'train'}
)
