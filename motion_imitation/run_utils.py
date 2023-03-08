import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time
import gym
import gin

from rl_perf.domains.quadruped_locomotion.motion_imitation.envs import env_builder as env_builder
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import imitation_policies as imitation_policies
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True

"""Wrapper to make the environment suitable for OpenAI gym."""


@gin.configurable('QuadrupedLocomotionEnv')
class RunUtils(gym.Env):
    def __init__(self,
                 seed=None,
                 benchmark=False,
                 mode="train",
                 motion_file="motion_imitation/data/motions/dog_pace.txt",
                 visualize=False,
                 output_dir="output",
                 num_test_episodes=None,
                 model_file="",
                 total_timesteps=2e8,
                 int_save_freq=0):

        self._seed = seed
        self._benchmark = benchmark
        self._mode = mode
        self._motion_file = motion_file
        self._visualize = visualize
        self._output_dir = output_dir
        self._num_test_episodes = num_test_episodes
        self._model_file = model_file
        self._total_timesteps = total_timesteps
        self._int_save_freq = int_save_freq

        if self._mode not in ['train', 'test']:
            assert False, "Unsupported mode: " + args.mode

        self._num_procs = MPI.COMM_WORLD.Get_size()
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        # Build enironment       
        self._built_env = self.build_environment()

        # Build model
        self._built_model = self.build_model(self._built_env, TIMESTEPS_PER_ACTORBATCH, OPTIM_BATCHSIZE)

        self.observation_space = self._built_env.observation_space
        self.action_space = self._built_env.action_space

    def build_environment(self):
        '''Build Environment'''
        enable_env_rand = ENABLE_ENV_RANDOMIZER and (self._mode != "test")
        env = env_builder.build_imitation_env(motion_files=[self._motion_file],
                                              num_parallel_envs=self._num_procs,
                                              mode=self._mode,
                                              enable_randomizer=enable_env_rand,
                                              enable_rendering=self._visualize)
        return env

    def set_rand_seed(self, seed=None):
        if seed is None:
            seed = int(time.time())

        seed += 97 * MPI.COMM_WORLD.Get_rank()

        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        return

    def build_model(self, env, timesteps_per_actorbatch, optim_batchsize):
        policy_kwargs = {
                "net_arch": [{"pi": [512, 256],
                              "vf": [512, 256]}],
                "act_fun": tf.nn.relu
                }

        timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / self._num_procs))
        optim_batchsize = int(np.ceil(float(optim_batchsize) / self._num_procs))

        model = ppo_imitation.PPOImitation(
                policy=imitation_policies.ImitationPolicy,
                env=env,
                gamma=0.95,
                timesteps_per_actorbatch=timesteps_per_actorbatch,
                clip_param=0.2,
                optim_epochs=1,
                optim_stepsize=1e-5,
                optim_batchsize=optim_batchsize,
                lam=0.95,
                adam_epsilon=1e-5,
                schedule='constant',
                policy_kwargs=policy_kwargs,
                tensorboard_log=self._output_dir,
                verbose=1)

        if self._model_file != "":
            model.load_parameters(self._model_file)

        return model

    def train(self, model, env):
        if (self._output_dir == ""):
            save_path = None
        else:
            save_path = os.path.join(self._output_dir, "model.zip")
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

        callbacks = []
        # Save a checkpoint every n steps
        if (self._output_dir != ""):
            if (self._int_save_freq > 0):
                int_dir = os.path.join(self._output_dir, "intermedate")
                callbacks.append(CheckpointCallback(save_freq=self._int_save_freq, save_path=int_dir,
                                                    name_prefix='model'))

        model.learn(total_timesteps=self._total_timesteps, save_path=save_path, callback=callbacks)

        return

    def test(self, model, env):
        curr_return = 0
        sum_return = 0
        episode_count = 0

        if self._num_test_episodes is not None:
            num_local_episodes = int(np.ceil(float(self._num_test_episodes) / self._num_procs))
        else:
            num_local_episodes = np.inf

        o = env.reset()
        while episode_count < num_local_episodes:
            a, _ = model.predict(o, deterministic=True)
            o, r, done, info = env.step(a)
            curr_return += r

            if done:
                o = env.reset()
                sum_return += curr_return
                episode_count += 1

        sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
        episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

        mean_return = sum_return / episode_count

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Mean Return: " + str(mean_return))
            print("Episode Count: " + str(episode_count))

        return

    def get_built_env(self):
        return self._built_env

    def get_built_model(self):
        return self._built_model

    def build_new_env(self):
        '''Build enironment'''
        self._built_env = self.build_environment()

    def build_new_model(self):
        '''Build model'''
        self._built_model = self.build_model(self._built_env, TIMESTEPS_PER_ACTORBATCH, OPTIM_BATCHSIZE)

    def step(self, action):
        return self._built_env.step(action)

    def reset(self):
        return self._built_env.reset()

    def close(self):
        self._built_env.close()

    def render(self, mode):
        return self._built_env.render(mode)

    def __getattr__(self, attr):
        return getattr(self._built_env, attr)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--benchmark", dest="benchmark", action="store_true", default=False)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str,
                            default="motion_imitation/data/motions/dog_pace.txt")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps

    args = arg_parser.parse_args()

    run = RunUtils(seed=args.seed,
                   benchmark=args.benchmark,
                   mode=args.mode,
                   motion_file=args.motion_file,
                   visualize=args.visualize,
                   output_dir=args.output_dir,
                   num_test_episodes=args.num_test_episodes,
                   model_file=args.model_file,
                   total_timesteps=args.total_timesteps,
                   int_save_freq=args.int_save_freq)

    # env = run.build_environment()
    # model = run.build_model(env, TIMESTEPS_PER_ACTORBATCH, OPTIM_BATCHSIZE)
    # run.test(model, env)

    run.test(run.get_built_model(), run.get_built_env())
