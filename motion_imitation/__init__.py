import gym

gym.envs.register(
        id='QuadrupedLocomotionEnv-v0',
        entry_point='motion_imitation.run_utils:RunUtils',
        )