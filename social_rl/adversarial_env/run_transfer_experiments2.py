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
from numpy.core.defchararray import title
from numpy.core.fromnumeric import sort
import pandas as pd
import tensorflow as tf  # tf

from tf_agents.environments import tf_py_environment
# from tf_agents.google.utils import mp4_video_recorder
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.trajectories import trajectory
from collections import defaultdict

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots



flags.DEFINE_string(
    'root_dir', None,
    'Directory where videos and transfer results will be saved')
flags.mark_flag_as_required('root_dir')

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
ALL_ENVS = MINI_VAL_ENVS+ VAL_ENVS + TEST_ENVS + MINI_TEST_ENVS 

BLACK_LIST_ENV_IDX = defaultdict(list)

def update_black_list():
    BLACK_LIST_ENV_IDX["MultiGrid-Cluttered50-Minigrid-v0"] += [2,4, ]
    BLACK_LIST_ENV_IDX["MultiGrid-MiniCluttered6-Minigrid-v0"] += [0,1,2 ]
    BLACK_LIST_ENV_IDX["MultiGrid-MiniCluttered-Lava-Minigrid-v0"] +=[1]
    BLACK_LIST_ENV_IDX["MultiGrid-MiniFourRooms-Minigrid-v0"] += [3, 4]
    BLACK_LIST_ENV_IDX["MultiGrid-MiniCluttered7-Minigrid-v0"] +=[0,3,4 ]
update_black_list()

def load_environment( env_name):
    if 'Adversarial' in env_name:
        py_env = adversarial_env.load(env_name)
        tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)
    else:
        py_env = multiagent_gym_suite.load(env_name)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return py_env, tf_env


def run_agent(policy, py_env, tf_env,env_name=False):
    """Run an agent's policy in a particular environment. Possibly record."""

    encoded_images = []

    # Add blank frames to make it easier to distinguish between runs/agents
    for _ in range(5):
        encoded_images.append(py_env.render())

    rewards = 0
    policy_state = policy.get_initial_state(1)


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


def display_vid(frames, title):
    img = None
    plt.clf()

    plt.title(title)
    for f in frames:
        if img is None:
            img = plt.imshow(f)
        else:
            img.set_data(f)
        
        plt.pause(.1)
        plt.draw()



def run_policy(i, env_name, policy):
    py_env, tf_env = load_environment(env_name)
    for j in range(i):
        tf_env.reset()

    return run_agent(policy, py_env, tf_env)

def run_experiments_on_env(policies, env_name,all_rewards, num__iters=5, display_vid=False):
    exp_names = []
    all_vids = {agent_name:[] for agent_name in policies}
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        exp_name = env_name + "_" + str(i)
        exp_names.append(exp_name)


        for agent_name in policies:
            reward, images = run_policy(i, env_name, policies[agent_name])
            reward = np.round(reward, 3)
            all_rewards[agent_name].append(reward)
            all_vids[agent_name].append(images)

        # remove 0 for-all reward results
        num_empty_rewards = 0
        for a_n , r_l in all_rewards.items():
            if r_l[-1] == 0:
                num_empty_rewards +=1

        if num_empty_rewards == len(all_rewards.keys()):
            exp_names = exp_names[:-1]
            for agent_name in policies:
                all_rewards[agent_name] = all_rewards[agent_name][:-1]
                all_vids[agent_name] = all_vids[agent_name][:-1]

        else:
            print('------------------')
            for agent_name in policies:

                if display_vid:
                    if all_rewards[agent_name] != 0:
                        display_vid(all_vids[agent_name], agent_name)

                print(f"{agent_name} reward: {all_rewards[agent_name][-1]} on env: {exp_name}")

    return exp_names, all_rewards

def compare_all_envs_train_rewards(all_env_names):
    env_idx = 1

    agent_names = ['original', 'entropy', 'history']
    policies = {}
    for a_n in agent_names:
        path = f"/home/nitsan/Downloads/saved_policies/{a_n}_policy_latest"
        policy = tf.compat.v2.saved_model.load(path)
        policies[a_n] = policy

    all_names= []
    all_rewards = {agent_name:[] for agent_name in policies}

    for env_name in all_env_names:
        exp_names, all_rewards = run_experiments_on_env(policies, env_name, all_rewards)
        all_names += exp_names
    
    pd_dict = {a_n: all_rewards[a_n] for a_n in all_rewards}
    pd_dict['env_names'] = all_names
    y = [a_n for a_n in all_rewards]
    
    df = pd.DataFrame(pd_dict)
    fig = px.line(df, x="env_names", y=y, title='Reward compare')
    fig.show()

def run_experiments_on_env_avg_reward(policies, env_name, all_rewards, num__iters=5, display_vid=False):
    all_vids = {agent_name:[] for agent_name in policies}
    num_envs_trained = 0
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        num_envs_trained +=1

        exp_name = env_name + "_" + str(i)


        for agent_name in policies:
            reward, images = run_policy(i, env_name, policies[agent_name])
            all_rewards[env_name][agent_name] +=reward
            all_vids[agent_name].append(images)
    
    for agent_name in policies:
        all_rewards[env_name][agent_name] = np.round(all_rewards[env_name][agent_name] /num_envs_trained, 3)

    return all_rewards



def compare_final_weights_avg_reward(all_env_names, agent_names = ['original', 'entropy', 'history']):

    policies = {}
    for a_n in agent_names:
        path = f"/home/nitsan/Downloads/saved_policies/{a_n}_policy_latest"
        policy = tf.compat.v2.saved_model.load(path)
        policies[a_n] = policy

    all_names= []
    all_rewards = {env_name:{} for env_name in all_env_names}
    for agent_name in policies:
        for env_name in all_env_names:
            all_rewards[env_name][agent_name] = 0

    for env_name in all_env_names:
        all_rewards = run_experiments_on_env_avg_reward(policies, env_name, all_rewards)
        all_names.append(env_name)
    
    titles = [env.replace("MultiGrid-", "").replace("Minigrid-", "") for env in ALL_ENVS]

    color10_16 = ['blue', 'cyan', 'red', "yellow",  "green",  "orange"]
    fig = make_subplots(rows=1, cols=len(all_env_names), subplot_titles=titles)
    for i,env_name in enumerate(all_env_names):
        fig.add_trace(
            go.Bar(x=[agent_name for agent_name in policies], y=[all_rewards[env_name][agent_name] for agent_name in policies],name=env_name,marker_color=color10_16),row=((i//4)+1), col=((i%4)+1))


    fig.update_layout(height=600, width=1600, title_text="Reward Comparation", showlegend=False, legend=dict(
    yanchor="bottom",
    y=-0.5,
    xanchor="right",
    x=1
))
    fig.show()


def run_experiments_on_env_curriculum_avg_reward(policy, env_name, num__iters=5, display_vid=False):
    num_envs_trained = 0
    avg_reward = 0
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        num_envs_trained +=1

        reward, images = run_policy(i, env_name, policy)
        avg_reward +=reward
    
        avg_reward = np.round(avg_reward /num_envs_trained, 3)

    return avg_reward



def compare_curriculum_weights_avg_reward(all_env_names, agent_names= ['original', 'entropy', 'history']):
    path = "/home/nitsan/Downloads/saved_policies/"
    
    
    all_rewards = {a_n:[] for a_n in agent_names}
    x_axis = {a_n:[]  for a_n in agent_names}
    for a_n in agent_names:
        agent_weight_dirs = [ os.path.join(path, name) for name in os.listdir(path) if (os.path.isdir(os.path.join(path, name)) and a_n in name) ]
        agent_weight_dirs = np.sort(agent_weight_dirs)
        print(agent_weight_dirs)
        for env_name in all_env_names:
            for a_w in agent_weight_dirs:
                policy = tf.compat.v2.saved_model.load(a_w)
                policy_iter = a_w.split("_")[-1]
                x_axis[a_n].append(policy_iter)
                full_name = a_n + "_" + policy_iter
                avg_reward = run_experiments_on_env_curriculum_avg_reward(policy, env_name)
                all_rewards[a_n].append(avg_reward)
    

    color10_16 = ['blue', 'cyan', 'red', "yellow",  "green",  "orange"]
    fig = make_subplots(rows=len(agent_names), cols=len(all_env_names), subplot_titles=all_env_names)
    for i,env_name in enumerate(all_env_names):
        for j,agent_name in enumerate(agent_names):
            fig.add_trace(
                go.Line(x=x_axis[agent_name], y=all_rewards[agent_name],name=agent_name,marker_color=color10_16),row=(j+1), col=(i+1))


    fig.update_layout(height=600, width=1600, title_text="Reward Comparation")

    fig.show()






def main(_):
    compare_curriculum_weights_avg_reward([ALL_ENVS[1]], agent_names = ['original', 'entropy'])
    compare_final_weights_avg_reward(ALL_ENVS, agent_names = ['original', 'entropy'])

    # compare_train_rewards()
    # for env_name in ALL_ENVS:
    #     py_env, tf_env = load_environment(env_name)
    #     for i in range(5):
    #         for j in range(i):
    #             tf_env.reset()
    #         plt.imshow(py_env.render())
    #         plt.title(env_name + "_"+str(i))
    #         plt.show()

if __name__ == '__main__':
    app.run(main)
