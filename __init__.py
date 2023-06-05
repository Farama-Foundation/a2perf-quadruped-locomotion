import gin
import gym

# gin.parse_config_files_and_bindings(["../rl_perf/domains/quadruped_locomotion/motion_imitation/configs/envdesign.gin"], None,
#                                     finalize_config=False)

gym.envs.register(
        id='QuadrupedLocomotionEnv-v0',
        entry_point='rl_perf.domains.quadruped_locomotion.motion_imitation.run_utils:RunUtils',
        kwargs={'motion_file': None}
        )
