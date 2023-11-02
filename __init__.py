import gin
import gym
import os

config_file_path = os.path.join(os.path.dirname(__file__), 'motion_imitation' , 'configs', 'envdesign.gin')


gin.parse_config_files_and_bindings([config_file_path], None,
                                    finalize_config=False)

gym.envs.register(
        id='QuadrupedLocomotion-v0',
        entry_point='rl_perf.domains.quadruped_locomotion.motion_imitation.envs.env_builder:build_imitation_env',
        kwargs={'motion_files': ["dog_pace.txt"], 'enable_rendering': False, 'mode': 'train'}
        )


