from functools import reduce
import os
import time
from collections import deque
import pickle
import warnings

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.ddpg import DDPG


class DDPGImitation(DDPG):
    """
    Deep Deterministic Policy Gradient (DDPG) model

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param memory_policy: (ReplayBuffer) the replay buffer
        (if None, default to baselines.deepq.replay_buffer.ReplayBuffer)

        .. deprecated:: 2.6.0
            This parameter will be removed in a future version

    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evaluation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf), normalize_returns must be set to True.
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evaluation environment
    :param memory_limit: (int) the max number of transitions to store, size of the replay buffer

        .. deprecated:: 2.6.0
            Use `buffer_size` instead.

    :param buffer_size: (int) the max number of transitions to store, size of the replay buffer
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for DDPG normally but can help exploring when using HER + DDPG.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):

        super(DDPGImitation, self).__init__(
            policy=policy, env=env, gamma=gamma, memory_policy=memory_policy, eval_env=eval_env,
            nb_train_steps=nb_train_steps, nb_rollout_steps=nb_rollout_steps, nb_eval_steps=nb_eval_steps,
            param_noise=param_noise, action_noise=action_noise, normalize_observations=normalize_observations,
            tau=tau, batch_size=batch_size, param_noise_adaption_interval=param_noise_adaption_interval,
            normalize_returns=normalize_returns, enable_popart=enable_popart, observation_range=observation_range,
            critic_l2_reg=critic_l2_reg, return_range=return_range, actor_lr=actor_lr, critic_lr=critic_lr,
            clip_norm=clip_norm, reward_scale=reward_scale, render=render, render_eval=render_eval,
            memory_limit=memory_limit, buffer_size=buffer_size, random_exploration=random_exploration,
            verbose=verbose, tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log, seed=seed,
            n_cpu_tf_sess=n_cpu_tf_sess)

    def learn(self, total_timesteps, callback=None, log_interval=-1, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()

            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            episode_successes = []

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                callback.on_training_start(locals(), globals())

                while True:
                    callback.on_rollout_start()
                    # Perform rollouts.
                    for _ in range(self.nb_rollout_steps):

                        if total_steps >= total_timesteps:
                            callback.on_training_end()
                            return self

                        # Predict next action.
                        action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                        assert action.shape == self.env.action_space.shape

                        # Execute next action.
                        if rank == 0 and self.render:
                            self.env.render()

                        # Randomly sample actions from a uniform distribution
                        # with a probability self.random_exploration (used in HER + DDPG)
                        if np.random.rand() < self.random_exploration:
                            # actions sampled from action space are from range specific to the environment
                            # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                            unscaled_action = self.action_space.sample()
                            action = scale_action(self.action_space, unscaled_action)
                        else:
                            # inferred actions need to be transformed to environment action_space before stepping
                            unscaled_action = unscale_action(self.action_space, action)

                        new_obs, reward, done, info = self.env.step(unscaled_action)

                        self.num_timesteps += 1

                        if callback.on_step() is False:
                            callback.on_training_end()
                            return self

                        step += 1
                        total_steps += 1
                        if rank == 0 and self.render:
                            self.env.render()

                        # Book-keeping.
                        epoch_actions.append(action)
                        epoch_qs.append(q_value)

                        # Store only the unnormalized version
                        if self._vec_normalize_env is not None:
                            new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                            reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                        else:
                            # Avoid changing the original ones
                            obs_, new_obs_, reward_ = obs, new_obs, reward

                        self._store_transition(obs_, action, reward_, new_obs_, done)
                        obs = new_obs
                        # Save the unnormalized observation
                        if self._vec_normalize_env is not None:
                            obs_ = new_obs_

                        episode_reward += reward_
                        episode_step += 1

                        if writer is not None:
                            ep_rew = np.array([reward_]).reshape((1, -1))
                            ep_done = np.array([done]).reshape((1, -1))
                            tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                writer, self.num_timesteps)

                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            episode_rewards_history.append(episode_reward)
                            epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            epoch_episodes += 1
                            episodes += 1

                            maybe_is_success = info.get('is_success')
                            if maybe_is_success is not None:
                                episode_successes.append(float(maybe_is_success))

                            self._reset()
                            if not isinstance(self.env, VecEnv):
                                obs = self.env.reset()

                    callback.on_rollout_end()
                    # Train.
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []
                    for t_train in range(self.nb_train_steps):
                        # Not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size):
                            break

                        # Adapt param noise, if necessary.
                        if len(self.replay_buffer) >= self.batch_size and \
                                t_train % self.param_noise_adaption_interval == 0:
                            distance = self._adapt_param_noise()
                            epoch_adaptive_distances.append(distance)

                        # weird equation to deal with the fact the nb_train_steps will be different
                        # to nb_rollout_steps
                        step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                self.num_timesteps - self.nb_rollout_steps)

                        critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)
                        self._update_target_net()

                    # Evaluate.
                    eval_episode_rewards = []
                    eval_qs = []
                    if self.eval_env is not None:
                        eval_episode_reward = 0.
                        for _ in range(self.nb_eval_steps):
                            if total_steps >= total_timesteps:
                                return self

                            eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                            unscaled_action = unscale_action(self.action_space, eval_action)
                            eval_obs, eval_r, eval_done, _ = self.eval_env.step(unscaled_action)
                            if self.render_eval:
                                self.eval_env.render()
                            eval_episode_reward += eval_r

                            eval_qs.append(eval_q)
                            if eval_done:
                                if not isinstance(self.env, VecEnv):
                                    eval_obs = self.eval_env.reset()
                                eval_episode_rewards.append(eval_episode_reward)
                                eval_episode_rewards_history.append(eval_episode_reward)
                                eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)
