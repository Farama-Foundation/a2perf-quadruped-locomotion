import gin
import gymnasium as gym
from absl import app

def main(_):
    # Create the environment.
    gin.parse_config_files_and_bindings(["./motion_imitation/configs/envdesign.gin"], None)
    env = gym.make('QuadrupedLocomotion-v0')
    print(env)


if __name__ == '__main__':
    app.run(main)