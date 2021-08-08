# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests trained models on transfer environments to generate videos and scores.

Note that this code assumes it will be provided with a .csv file indicating
which checkpoints it should load based on finding the best hyperparameters
for a given metric, such as 'SolvedPathLength_last20%'. It assumes this csv will
have columns labeled 'metric', 'exp_id', 'best_seeds', and 'settings'. Such a
csv can be created using the function utils.save_best_work_units_csv()
"""
import ast
import datetime
import os
import pdb
import pickle
import sys

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf  # tf

from tf_agents.environments import tf_py_environment
# from tf_agents.google.utils import mp4_video_recorder
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.trajectories import trajectory

# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid

from social_rl.adversarial_env import adversarial_env
from social_rl.adversarial_env import utils
from social_rl.multiagent_tfagents import multiagent_gym_suite
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import plotly.express as px


flags.DEFINE_string(
    'root_dir', None,
    'Directory where videos and transfer results will be saved')
flags.mark_flag_as_required('root_dir')
# flags.DEFINE_string(
#     'hparam_csv', None,
#     'Required to determine which checkpoints to load.')
# flags.mark_flag_as_required('hparam_csv')
flags.DEFINE_string(
    'transfer_csv', None,
    'If provided, will load this csv and continue saving to it')
flags.DEFINE_boolean(
    'test_on_test', False,
    'If True, will also test on the test environments')
flags.DEFINE_boolean(
    'test_mini', False,
    'If True, will test on the mini environments instead')
flags.DEFINE_boolean(
    'fill_in_missing', False,
    'If True, will test load all existing transfer dfs and try to fill in the '
    'missing data')
flags.DEFINE_boolean(
    'reverse_order', False,
    'If True, will iterate through experiments in reverse order.')
flags.DEFINE_string(
    'metric', 'SolvedPathLength_best_ever',
    'Metric to use for selecting which seeds to test.')
flags.DEFINE_boolean(
    'debug', False,
    'If True, will only load 1 seed for each experiment for speed')
flags.DEFINE_integer(
    'num_trials', 10,
    'Number of trials to do for each seed in each transfer environment')
flags.DEFINE_boolean(
    'save_video_matrices', False,
    'If True, will also save matrix encodings of the environment state used to'
    'make rendered videos')
flags.DEFINE_string(
    'name', 'Test transfer',
    'Informative name to output to explain what process is running.')
FLAGS = flags.FLAGS


VAL_ENVS = [
    'MultiGrid-TwoRooms-Minigrid-v0',
    'MultiGrid-Cluttered40-Minigrid-v0',
    'MultiGrid-Cluttered10-Minigrid-v0',
    'MultiGrid-SixteenRooms-v0',
    'MultiGrid-Maze2-v0',
    'MultiGrid-Maze3-v0',
    'MultiGrid-Labyrinth2-v0',
]
TEST_ENVS = [
    'MultiGrid-FourRooms-Minigrid-v0',
    'MultiGrid-Cluttered50-Minigrid-v0',
    'MultiGrid-Cluttered5-Minigrid-v0',
    'MultiGrid-Empty-Random-15x15-Minigrid-v0',
    'MultiGrid-SixteenRoomsFewerDoors-v0',
    'MultiGrid-Maze-v0',
    'MultiGrid-Labyrinth-v0',
]
MINI_VAL_ENVS = [
    'MultiGrid-MiniTwoRooms-Minigrid-v0',
    'MultiGrid-Empty-Random-6x6-Minigrid-v0',
    'MultiGrid-MiniCluttered6-Minigrid-v0',
    'MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    'MultiGrid-MiniMaze-v0'
]
MINI_TEST_ENVS = [
    'MultiGrid-MiniFourRooms-Minigrid-v0',
    'MultiGrid-MiniCluttered7-Minigrid-v0',
    'MultiGrid-MiniCluttered1-Minigrid-v0'
]


def load_environment( env_name):
    if 'Adversarial' in env_name:
        py_env = adversarial_env.load(env_name)
        tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)
    else:
        py_env = multiagent_gym_suite.load(env_name)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return py_env, tf_env


def display_vid(frames):
    img = None
    for f in frames:
        if img is None:
            img = plt.imshow(f)
        else:
            img.set_data(f)
        plt.pause(.1)
        plt.draw()


def run_agent(policy, py_env, tf_env,env_name=False):
    """Run an agent's policy in a particular environment. Possibly record."""

    encoded_images = []

    # Add blank frames to make it easier to distinguish between runs/agents
    # for _ in range(5):
    #     encoded_images.append(py_env.render())

    rewards = 0
    policy_state = policy.get_initial_state(1)


    # elif 'Adversarial' in env_name:
    #     time_step = tf_env.reset_agent()
    # else:
    time_step = tf_env.reset()

    encoded_images.append(py_env.render())  # pylint:disable=protected-access

    num_steps = tf.constant(0.0)
    while True:
        policy_step = policy.action(time_step, policy_state=policy_state)

        policy_state = policy_step.state
        next_time_step = tf_env.step(policy_step.action)

        traj = trajectory.from_transition(time_step, policy_step, next_time_step)
        time_step = next_time_step

        num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

        rewards += time_step.reward

        encoded_images.append(py_env.render())  # pylint:disable=protected-access

        if traj.is_last():
            break

    return rewards.numpy().sum(), encoded_images


def run_experiments_on_env(policies, env_name):
    exp_names = []
    original_reward = []
    new_reward = []
    for i in range(15):
        exp_name = env_name + "_" + str(i)
        exp_names.append(exp_name)
        py_env, tf_env = load_environment(env_name)
        for j in range(i):
            tf_env.reset()

        reward, images = run_agent(policies['original'], py_env, tf_env)
        reward = np.round(reward, 3)
        original_reward.append(reward)
        
        # display_vid(images)


        py_env, tf_env = load_environment(env_name)
        for j in range(i):
            tf_env.reset()

        reward, images = run_agent(policies['entropy'], py_env, tf_env)
        reward = np.round(reward, 3)
        new_reward.append(reward)
        # display_vid(images)
        if original_reward[-1] == 0 and new_reward[-1] == 0:
            original_reward = original_reward[:-1]
            new_reward = new_reward[:-1]
        else:
            print('------------------')
            print(f"original reward: {original_reward[-1]} on env: {exp_name}")
            print(f"entropy reward: {new_reward[-1]} on env: {exp_name}")
    return exp_names, original_reward, new_reward


def main(_):



# MINI_VAL_ENVS
# MINI_TEST_ENVS 
    all_env_names = MINI_TEST_ENVS + TEST_ENVS + MINI_VAL_ENVS + MINI_TEST_ENVS
    env_idx = 1

    agent_names = ['original', 'entropy']
    policies = {}
    for a_n in agent_names:
        path = f"/home/nitsan/Downloads/{a_n}_policy_000499950"
        policy = tf.compat.v2.saved_model.load(path)
        policies[a_n] = policy

    all_names, all_orig_rewards, all_new_rewards = [],[],[]
    
    for env_name in all_env_names:
        exp_names, original_reward, new_reward = run_experiments_on_env(policies, env_name)
        all_names += exp_names
        all_orig_rewards += original_reward
        all_new_rewards += new_reward
    
    df = pd.DataFrame({'env_names': all_names, 'orig_reward':all_orig_rewards , 'new_reward':all_new_rewards })
    fig = px.line(df, x="env_names", y=['orig_reward', 'new_reward'], title='Reward compare')
    fig.show()

if __name__ == '__main__':
    app.run(main)
