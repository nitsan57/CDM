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

"""Runs the adversary to create the environment, then the agents to play it.

Implements episode collection for the PAIRED algorithm, a minimax adversary, and
domain randomization. First runs the adversary to generate the environment, then
runs the main agent, and (if running PAIRED) the antagonist agent. The scores
of both agents are used to compute the regret, which is used to train the
adversary and the agents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import cv2

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from social_rl.custom_printer import custom_printer
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
import os
import copy

from social_rl.adversarial_env.curriculum_env_rating import EnvCurriculum


class AdversarialDriver(object):
    """Runs the environment adversary and agents to collect episodes."""

    def __init__(self,
                 env,
                 agent,
                 adversary_agent,
                 adversary_env,
                 env_metrics=None,
                 collect=True,
                 disable_tf_function=False,
                 debug=False,
                 combined_population=False,
                 flexible_protagonist=False):
        """Runs the environment adversary and agents to collect episodes.

        Args:
          env: A tf_environment.Base environment.
          agent: An AgentTrainPackage for the main learner agent.
          adversary_agent: An AgentTrainPackage for the second agent, the
            adversary's ally. This can be None if using an unconstrained adversary
            environment.
          adversary_env: An AgentTrainPackage for the agent that controls the
            environment, learning to set parameters of the environment to decrease
            the agent's score relative to the adversary_agent. Can be None if using
            domain randomization.
          env_metrics: Global environment metrics to track (such as path length).
          collect: True if collecting episodes for training, otherwise eval.
          disable_tf_function: If True the use of tf.function for the run method is
            disabled.
          debug: If True, outputs informative logging statements.
          combined_population: If True, the entire population of protagonists plays
            each generated environment, and regret is the calc'd as the difference
            between the max of the population and the average (there are no explicit
            antagonists).
          flexible_protagonist: Which agent plays the role of protagonist in
            calculating the regret depends on which has the lowest score.
        """
        common.check_tf1_allowed()
        self.debug = debug
        self.total_episodes_collected = 0

        if not disable_tf_function:
            self.run = common.function(self.run, autograph=True)
            self.run_agent = common.function(self.run_agent, autograph=True)

        self.env_metrics = env_metrics
        self.collect = collect
        self.env = env
        self.agent = agent
        self.adversary_agent = adversary_agent
        self.adversary_env = adversary_env
        self.combined_population = combined_population
        self.flexible_protagonist = flexible_protagonist

    def run(self, random_episodes=False, search_based=False):
        """Runs 3 policies in same environment: environment, agent 1, agent 2."""
        if search_based:
            agent_r_max, train_idxs = self.adversarial_episode_search()  # self.adversarial_episode()
        if random_episodes:
            # Generates a random environment for both protagonist and antagonist
            # to play.
            agent_r_max, train_idxs = self.randomized_episode()
        elif self.adversary_env is not None:
            # Generates an environment using an adversary.
            if self.combined_population:
                agent_r_max, train_idxs = self.combined_population_adversarial_episode()
            else:
                if os.environ["mode"] == "original":
                    agent_r_max, train_idxs = self.adversarial_episode() 
                else:
                    agent_r_max, train_idxs = self.adversarial_episode_heuristic() #self.adversarial_episode() 

        else:
            # Only one agent plays a randomly generated environment.
            agent_r_max, train_idxs = self.domain_randomization_episode()

        self.total_episodes_collected += agent_r_max.shape[0]

        self.log_environment_metrics(agent_r_max)

        return train_idxs

    def adversarial_episode_search(self):
        """Episode in which adversary constructs environment and agents play it."""
        # Build environment with adversary.

        ###NEW LOGIC###
        from social_rl.adversarial_env.adversarial_env import AdversarialTFPyEnvironment
        orig_data = self.env.data_PyEnvironment
        custom_printer(f"ENVIRONEMNT NUMBER: {self.total_episodes_collected}")
        # create some env copies
        train_idxs = {}
        # import pdb
        # pdb.set_trace()
        env_curriculum = EnvCurriculum()
        if self.collect:
            num_envs = 20
            orig_env_list = [AdversarialTFPyEnvironment(orig_data) for i in range(num_envs)]
            filled_base_env_list = []
            trajectories_list = []
            for i in range(len(orig_env_list)):
                _, _, env_idx, trajectories = self.run_agent(
                    orig_env_list[i], self.adversary_env, self.env.reset, self.env.step_adversary, gen_env_mode=True)
                trajectories_list.append(trajectories)
                filled_base_env_list.append(orig_env_list[i])

            agent_idx = np.random.choice(len(self.agent))
            agent = self.agent[agent_idx]
            # custom_printer(f"AGNET_RUNNING ON SAMPLED STATE:{agent.name}")
            policy = agent.collect_policy

            policy_state = policy.get_initial_state(self.env.batch_size)
            idx = env_curriculum.choose_best_env_idx(filled_base_env_list, policy, policy_state)
            self.env = orig_env_list[idx]
            # for trajectories in trajectories_list:
            for traj in trajectories_list[idx]:
                for obs in self.adversary_env[0].observers:
                    obs(traj)

            train_idxs = {'adversary_env': [agent_idx]}
            ####
        else:
            # Build environment with adversary.
            _, _, env_idx = self.run_agent(
                self.env, self.adversary_env, self.env.reset, self.env.step_adversary)
            train_idxs = {'adversary_env': [env_idx]}

        # Run protagonist in generated environment.
        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
            self.env, self.agent, self.env.reset_agent, self.env.step)
        train_idxs['agent'] = [agent_idx]
        # print("end of run:", agent_r_avg)

        # Run antagonist in generated environment.
        if self.adversary_agent:
            adv_agent_r_avg, adv_agent_r_max, antag_idx = self.run_agent(
                self.env, self.adversary_agent, self.env.reset_agent, self.env.step)
            train_idxs['adversary_agent'] = [antag_idx]
        # print("end of run:", adv_agent_r_avg)

        # Use agents' reward to compute and set regret-based rewards for PAIRED.
        # By default, regret = max(antagonist) - mean(protagonist).
        if self.adversary_agent:
            self.adversary_agent[antag_idx].enemy_max = agent_r_max
            self.agent[agent_idx].enemy_max = adv_agent_r_max
            if self.flexible_protagonist:
                # In flexible protagonist case, we find the best-performing agent
                # and compute regret = max(best) - mean(other).
                protagonist_better = tf.cast(tf.math.greater(agent_r_max, adv_agent_r_max), tf.float32)
                env_reward = protagonist_better * (agent_r_max - adv_agent_r_avg) + (1 - protagonist_better) * (adv_agent_r_max - agent_r_avg)
                adv_agent_r_max = protagonist_better * agent_r_max + (1 - protagonist_better) * adv_agent_r_max
            elif self.adversary_env[env_idx].non_negative_regret:
                # Clip regret signal so that it can't go below zero.
                env_reward = tf.math.maximum(adv_agent_r_max - agent_r_avg, 0)
            else:
                # Regret = max(antagonist) - mean(protagonist)
                env_reward = adv_agent_r_max - agent_r_avg

            # Add adversary block budget.
            env_reward += self.compute_adversary_block_budget(
                adv_agent_r_max, env_idx)

        # Minimax adversary reward.
        else:
            env_reward = -agent_r_avg

        self.adversary_env[env_idx].final_reward = env_reward

        # Log metrics to tensorboard.
        if self.collect:
            self.adversary_env[env_idx].env_train_metric(env_reward)
        else:
            self.adversary_env[env_idx].env_eval_metric(env_reward)

        # Log metrics to console.
        if self.debug:
            custom_printer(f'Agent reward: avg = {tf.reduce_mean(agent_r_avg).numpy()}, max = {tf.reduce_mean(agent_r_max).numpy()}')
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(agent_r_avg).numpy(),
                         tf.reduce_mean(agent_r_max).numpy())
            logging.info('Environment score: %f',
                         tf.reduce_mean(env_reward).numpy())
            if self.adversary_agent:
                custom_printer(f'Adversary Agent reward: avg = {tf.reduce_mean(adv_agent_r_avg).numpy()}, max = {tf.reduce_mean(adv_agent_r_max).numpy()}')
                logging.info('Adversary agent reward: avg = %f, max = %f',
                             tf.reduce_mean(adv_agent_r_avg).numpy(),
                             tf.reduce_mean(adv_agent_r_max).numpy())

        return agent_r_max, train_idxs

    def adversarial_episode_heuristic(self):
        """Episode in which adversary constructs environment and agents play it."""
        # Build environment with adversary.

        ###NEW LOGIC###
        from social_rl.adversarial_env.adversarial_env import AdversarialTFPyEnvironment
        orig_data = self.env.data_PyEnvironment
        # create some env copies
        train_idxs = {}
        custom_printer(f"ENVIRONEMNT NUMBER: {self.total_episodes_collected}")

        env_curriculum = EnvCurriculum()
        if self.collect:
            num_envs = 20
            orig_env_list = [AdversarialTFPyEnvironment(orig_data) for i in range(num_envs)]
            filled_base_env_list = []
            trajectories_list = []
            for i in range(len(orig_env_list)):
                _, _, env_idx, trajectories = self.run_agent(
                    orig_env_list[i], self.adversary_env, self.env.reset, self.env.step_adversary, gen_env_mode=True)
                trajectories_list.append(trajectories)
                filled_base_env_list.append(orig_env_list[i])

            agent_idx = np.random.choice(len(self.agent))
            agent = self.agent[agent_idx]
            # custom_printer(f"AGNET_RUNNING ON SAMPLED STATE:{agent.name}")
            policy = agent.collect_policy

            policy_state = policy.get_initial_state(self.env.batch_size)
            if os.environ["mode"] == "entropy":
                idx = env_curriculum.choose_best_env_idx_by_entropy(filled_base_env_list, policy, policy_state)
            elif os.environ["mode"] == "history":
                idx = env_curriculum.choose_best_env_idx_by_history(filled_base_env_list)
            else:
                custom_print("HEURSITIC MODE NOT SUPPORTED!!! Will exit now..")
                exit()

            self.env = orig_env_list[idx]
            # for trajectories in trajectories_list:
            for traj in trajectories_list[idx]:
                for obs in self.adversary_env[0].observers:
                    obs(traj)

            train_idxs = {'adversary_env': [agent_idx]}
            ####
        else:
            # Build environment with adversary.
            _, _, env_idx = self.run_agent(
                self.env, self.adversary_env, self.env.reset, self.env.step_adversary)
            train_idxs = {'adversary_env': [env_idx]}

        # Run protagonist in generated environment.
        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
            self.env, self.agent, self.env.reset_agent, self.env.step)
        train_idxs['agent'] = [agent_idx]
        # print("end of run:", agent_r_avg)

        # Run antagonist in generated environment.
        if self.adversary_agent:
            adv_agent_r_avg, adv_agent_r_max, antag_idx = self.run_agent(
                self.env, self.adversary_agent, self.env.reset_agent, self.env.step)
            train_idxs['adversary_agent'] = [antag_idx]
        # print("end of run:", adv_agent_r_avg)

        # Use agents' reward to compute and set regret-based rewards for PAIRED.
        # By default, regret = max(antagonist) - mean(protagonist).
        if self.adversary_agent:
            self.adversary_agent[antag_idx].enemy_max = agent_r_max
            self.agent[agent_idx].enemy_max = adv_agent_r_max
            if self.flexible_protagonist:
                # In flexible protagonist case, we find the best-performing agent
                # and compute regret = max(best) - mean(other).
                protagonist_better = tf.cast(tf.math.greater(agent_r_max, adv_agent_r_max), tf.float32)
                env_reward = protagonist_better * (agent_r_max - adv_agent_r_avg) + (1 - protagonist_better) * (adv_agent_r_max - agent_r_avg)
                adv_agent_r_max = protagonist_better * agent_r_max + (1 - protagonist_better) * adv_agent_r_max
            elif self.adversary_env[env_idx].non_negative_regret:
                # Clip regret signal so that it can't go below zero.
                env_reward = tf.math.maximum(adv_agent_r_max - agent_r_avg, 0)
            else:
                # Regret = max(antagonist) - mean(protagonist)
                env_reward = adv_agent_r_max - agent_r_avg

            # Add adversary block budget.
            env_reward += self.compute_adversary_block_budget(
                adv_agent_r_max, env_idx)

        # Minimax adversary reward.
        else:
            env_reward = -agent_r_avg

        self.adversary_env[env_idx].final_reward = env_reward

        # Log metrics to tensorboard.
        if self.collect:
            self.adversary_env[env_idx].env_train_metric(env_reward)
        else:
            self.adversary_env[env_idx].env_eval_metric(env_reward)
        
        if os.environ["mode"] == "history":
            self.env.reset_agnet()
            env = self.env.render('rgb_array')
            agent_reward = tf.reduce_mean(agent_r_avg).numpy()
            env_curriculum.History[env] = agent_reward

        # Log metrics to console.
        if self.debug:
            custom_printer(f'Agent reward: avg = {tf.reduce_mean(agent_r_avg).numpy()}, max = {tf.reduce_mean(agent_r_max).numpy()}')
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(agent_r_avg).numpy(),
                         tf.reduce_mean(agent_r_max).numpy())
            logging.info('Environment score: %f',
                         tf.reduce_mean(env_reward).numpy())
            if self.adversary_agent:
                custom_printer(f'Adversary Agent reward: avg = {tf.reduce_mean(adv_agent_r_avg).numpy()}, max = {tf.reduce_mean(adv_agent_r_max).numpy()}')
                logging.info('Adversary agent reward: avg = %f, max = %f',
                             tf.reduce_mean(adv_agent_r_avg).numpy(),
                             tf.reduce_mean(adv_agent_r_max).numpy())

        return agent_r_max, train_idxs

    def adversarial_episode(self):
        """Episode in which adversary constructs environment and agents play it."""
        # Build environment with adversary.
        _, _, env_idx = self.run_agent(
            self.env, self.adversary_env, self.env.reset, self.env.step_adversary)
        train_idxs = {'adversary_env': [env_idx]}
        custom_printer(f"ENVIRONEMNT NUMBER: {self.total_episodes_collected}")
        ########################################################
        # DEBUG ENV
        # x = self.env._envs[-1].render()
        # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        # print("EPISODE NUMBER", self.total_episodes_collected)
        # h, w, c = x.shape
        # ratio = w / h
        # new_h = 400
        # new_w = int(new_h * ratio)
        # x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(r"temp/images/" + str(self.total_episodes_collected) + ".png", x)
        ########################################################

        # Run protagonist in generated environment.
        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
            self.env, self.agent, self.env.reset_agent, self.env.step)
        train_idxs['agent'] = [agent_idx]
        # print("end of run:", agent_r_avg)

        # Run antagonist in generated environment.

        if self.adversary_agent:
            adv_agent_r_avg, adv_agent_r_max, antag_idx = self.run_agent(
                self.env, self.adversary_agent, self.env.reset_agent, self.env.step)
            train_idxs['adversary_agent'] = [antag_idx]
        # print("end of run:", adv_agent_r_avg)

        # Use agents' reward to compute and set regret-based rewards for PAIRED.
        # By default, regret = max(antagonist) - mean(protagonist).
        if self.adversary_agent:
            self.adversary_agent[antag_idx].enemy_max = agent_r_max
            self.agent[agent_idx].enemy_max = adv_agent_r_max
            if self.flexible_protagonist:
                # In flexible protagonist case, we find the best-performing agent
                # and compute regret = max(best) - mean(other).
                protagonist_better = tf.cast(tf.math.greater(agent_r_max, adv_agent_r_max), tf.float32)
                env_reward = protagonist_better * (agent_r_max - adv_agent_r_avg) + (1 - protagonist_better) * (adv_agent_r_max - agent_r_avg)
                adv_agent_r_max = protagonist_better * agent_r_max + (1 - protagonist_better) * adv_agent_r_max
            elif self.adversary_env[env_idx].non_negative_regret:
                # Clip regret signal so that it can't go below zero.
                env_reward = tf.math.maximum(adv_agent_r_max - agent_r_avg, 0)
            else:
                # Regret = max(antagonist) - mean(protagonist)
                env_reward = adv_agent_r_max - agent_r_avg

            # Add adversary block budget.
            env_reward += self.compute_adversary_block_budget(
                adv_agent_r_max, env_idx)

        # Minimax adversary reward.
        else:
            env_reward = -agent_r_avg

        self.adversary_env[env_idx].final_reward = env_reward

        # Log metrics to tensorboard.
        if self.collect:
            self.adversary_env[env_idx].env_train_metric(env_reward)
        else:
            self.adversary_env[env_idx].env_eval_metric(env_reward)

        # Log metrics to console.
        if self.debug:
            custom_printer(f'Agent reward: avg = {tf.reduce_mean(agent_r_avg).numpy()}, max = {tf.reduce_mean(agent_r_max).numpy()}')
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(agent_r_avg).numpy(),
                         tf.reduce_mean(agent_r_max).numpy())
            logging.info('Environment score: %f',
                         tf.reduce_mean(env_reward).numpy())
            if self.adversary_agent:
                custom_printer(f'Adversary Agent reward: avg = {tf.reduce_mean(adv_agent_r_avg).numpy()}, max = {tf.reduce_mean(adv_agent_r_max).numpy()}')
                logging.info('Adversary agent reward: avg = %f, max = %f',
                             tf.reduce_mean(adv_agent_r_avg).numpy(),
                             tf.reduce_mean(adv_agent_r_max).numpy())

        return agent_r_max, train_idxs

    def combined_population_adversarial_episode(self):
        """Episode in which adversary constructs environment and agents play it."""
        # Build environment with adversary.
        _, _, env_idx = self.run_agent(
            self.env, self.adversary_env, self.env.reset, self.env.step_adversary)
        train_idxs = {'adversary_env': [env_idx], 'agent': []}

        # Run all protagonist agents in generated environment.
        means = []
        maxs = []
        for agent_idx in range(len(self.agent)):
            agent_r_avg, agent_r_max, agent_idx_selected = self.run_agent(
                self.env, self.agent, self.env.reset_agent, self.env.step,
                agent_idx=agent_idx)
            assert agent_idx == agent_idx_selected
            means.append(agent_r_avg)
            maxs.append(agent_r_max)
            train_idxs['agent'].append(agent_idx)

        # Stack into shape: [num agents in population, batch]
        means = tf.stack(means)
        maxs = tf.stack(maxs)

        # Compute and set regret-based rewards for PAIRED.
        population_max = tf.reduce_max(maxs, axis=0)
        population_avg = tf.reduce_mean(means, axis=0)
        regret = population_max - population_avg
        if self.adversary_env[env_idx].non_negative_regret:
            regret = tf.math.maximum(regret, 0)

        for agent_idx in range(len(self.agent)):
            self.agent[agent_idx].enemy_max = population_max

        adv_r = regret + self.compute_adversary_block_budget(
            population_max, env_idx)

        self.adversary_env[env_idx].final_reward = adv_r

        # Log metrics to tensorboard.
        if self.collect:
            self.adversary_env[env_idx].env_train_metric(adv_r)
        else:
            self.adversary_env[env_idx].env_eval_metric(adv_r)

        # Log metrics to console.
        if self.debug:
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(population_avg).numpy(),
                         tf.reduce_max(population_max).numpy())
            logging.info('Environment regret: %f',
                         tf.reduce_mean(regret).numpy())

        return population_max, train_idxs

    def log_environment_metrics(self, agent_r_max):
        """Log extra environment metrics."""
        distance_to_goal = self.env.get_distance_to_goal()
        num_blocks = self.env.get_num_blocks()
        deliberate_placement = self.env.get_deliberate_placement()
        env_episodes = [tf.convert_to_tensor(
            self.total_episodes_collected, dtype=tf.float32)]
        goal_x = self.env.get_goal_x()
        goal_y = self.env.get_goal_y()
        passable = self.env.get_passable()
        shortest_path = self.env.get_shortest_path_length()
        shortest_passable_path = passable * shortest_path
        solved = tf.cast(agent_r_max > 0, tf.float32)
        solved_path_length = solved * shortest_path
        for i, m in enumerate([distance_to_goal, num_blocks,
                               deliberate_placement, env_episodes, goal_x, goal_y,
                               passable, shortest_path, shortest_passable_path,
                               solved_path_length]):
            self.env_metrics[i](m)

        if self.debug:
            logging.info('Driver times invoked %d', self.total_episodes_collected)
            logging.info('Num blocks: %f', tf.reduce_mean(num_blocks).numpy())
            logging.info('Distance to goal: %f',
                         tf.reduce_mean(distance_to_goal).numpy())
            logging.info('Deliberate agent placement: %f',
                         tf.reduce_mean(deliberate_placement).numpy())
            logging.info('Goal (X, Y): (%f, %f)', tf.reduce_mean(goal_x).numpy(),
                         tf.reduce_mean(goal_y).numpy())
            logging.info('Possible to finish environment?: %f',
                         tf.reduce_mean(passable).numpy())
            logging.info('Shortest path length to goal: %f',
                         tf.reduce_mean(shortest_path).numpy())
            logging.info('Solved path length: %f',
                         tf.reduce_mean(solved_path_length).numpy())

            # print("!!!!!!!", gym_env, type(gym_env))
            # print(self.env._env)
            # print(self.env.render())

    def domain_randomization_episode(self):
        """Use random reset function to create a randomized environment."""
        # Randomly generate environment.
        self.env.reset_random()

        # Run single agent.
        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
            self.env, self.agent, self.env.reset_agent, self.env.step)
        train_idxs = {'agent': [agent_idx]}

        if self.debug:
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(agent_r_avg).numpy(),
                         tf.reduce_mean(agent_r_max).numpy())

        return agent_r_max, train_idxs

    def randomized_episode(self):
        """Both agent and adversary_agent play a randomized environment."""
        # Randomly generate environment.
        self.env.reset_random()

        # Run protagonist agent.
        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
            self.env, self.agent, self.env.reset_agent, self.env.step)
        train_idxs = {'agent': [agent_idx]}

        # Run antagonist agent.
        if self.adversary_agent:
            adv_agent_r_avg, adv_agent_r_max, antag_idx = self.run_agent(
                self.env, self.adversary_agent, self.env.reset_agent, self.env.step)
            train_idxs['adversary_agent'] = [antag_idx]

        # Use agents' reward to compute and set regret-based rewards for PAIRED.
        if self.adversary_agent:
            self.adversary_agent[antag_idx].enemy_max = agent_r_max
            self.agent[agent_idx].enemy_max = adv_agent_r_max
        else:
            self.agent[agent_idx].enemy_max = agent_r_max

        if self.debug:
            logging.info('Agent reward: avg = %f, max = %f',
                         tf.reduce_mean(agent_r_avg).numpy(),
                         tf.reduce_mean(agent_r_max).numpy())
            if self.adversary_agent:
                logging.info('Adversary agent reward: avg = %f, max = %f',
                             tf.reduce_mean(adv_agent_r_avg).numpy(),
                             tf.reduce_mean(adv_agent_r_max).numpy())

        return agent_r_max, train_idxs

#        agent_r_avg, agent_r_max, agent_idx = self.run_agent(
        # self.env, self.agent, self.env.reset_agent, self.env.step)

    def run_agent(self, env, agent_list, reset_func, step_func, agent_idx=None, gen_env_mode=False):
        """Runs an agent in an environment given a step and reset function.

        Args:
          env: A TF-agents TF environment.
          agent_list: A list of TrainAgentPackages, each of which contains an agent
            that can be run in the environment. The agent to run will be randomly
            selected from the list (to handle population based training).
          reset_func: Callable function used to reset the environment.
          step_func: Callable function used to step the environment.
          agent_idx: The integer population index of the agent to run.

        Returns:
          The average reward achieved, the maximum reward, and the index of the
            agent selected.
        """
        if gen_env_mode:
            traj_list = []

        if agent_idx is None:
            agent_idx = np.random.choice(len(agent_list))
        agent = agent_list[agent_idx]
        # custom_printer(f"AGNET_RUNNING:{agent.name}", )

        if self.collect:
            policy = agent.collect_policy
            observers = agent.observers
        else:
            policy = agent.eval_policy
            observers = agent.eval_metrics

        time_step = reset_func()
        policy_state = policy.get_initial_state(env.batch_size)

        num_steps = tf.constant(0.0)
        num_episodes = tf.zeros_like(time_step.reward)

        avg_reward = tf.zeros_like(time_step.reward)
        max_reward = tf.zeros_like(time_step.reward)

        h, w, c = self.env._envs[-1].render().shape

        ratio = w / h
        new_h = 400
        new_w = int(new_h * ratio)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        debug_dir = os.environ['debug_dir']
        im_dir = debug_dir + '/images/steps/'
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)

        writer = cv2.VideoWriter(im_dir + str(self.total_episodes_collected) + "_" + agent.name + ".mp4", fourcc, 15, (new_w, new_h))

        while num_steps < agent.max_steps:
            action_step = policy.action(time_step, policy_state)
            next_time_step = step_func(action_step.action)

            if False:  # for debug only
                image = next_time_step.observation['image'][0]
                image = image.numpy().astype(np.float32)
                # image = image / 10
                # image = image * 255
                image = image.astype(np.uint8)
                custom_printer(f"STEP NUMBER:{num_steps.numpy()}")
                x = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h, w, c = x.shape
                ratio = w / h
                new_h = 400
                new_w = int(new_h * ratio)
                x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(debug_dir + "/agent_view_" + str(int(num_steps.numpy())) + ".png", x)

                x = self.env._envs[-1].render()
                x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                writer.write(x)
                cv2.imwrite(debug_dir + "/real_view_" + str(int(num_steps.numpy())) + ".png", x)

            x = self.env._envs[-1].render()
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            x = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            writer.write(x)

            # Replace with terminal timestep to manually end episode (enables
            # artificially decreasing number of steps for one of the agents).

            if agent.name == 'agent' and num_steps >= agent.max_steps - 1:
                outer_dims = nest_utils.get_outer_array_shape(
                    next_time_step.reward, env.reward_spec())
                next_time_step = ts_lib.termination(
                    next_time_step.observation, next_time_step.reward,
                    outer_dims=outer_dims)
            num_steps += 1

            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            num_episodes += tf.cast(traj.is_last(), tf.float32)

            avg_reward += next_time_step.reward
            # print("REALLY REWARD PROPEGATED HERE:", next_time_step, tf.cast(traj.is_last(), tf.float32))
            # print("REALLY REWARD PROPEGATED HERE:", next_time_step.reward, num_episodes, tf.cast(traj.is_last(), tf.float32))
            max_reward = tf.math.maximum(max_reward, next_time_step.reward)

            if gen_env_mode == False:
                for observer in observers:
                    observer(traj)
            else:
                traj_list.append(traj)

            time_step = next_time_step
            policy_state = action_step.state

        writer.release()

        avg_reward = avg_reward / num_episodes
        if gen_env_mode == False:
            return avg_reward, max_reward, agent_idx
        else:
            return avg_reward, max_reward, agent_idx, traj_list

        return avg_reward, max_reward, agent_idx

    def compute_adversary_block_budget(self, antag_r_max, env_idx,
                                       use_shortest_path=True):
        """Compute block budget reward based on antagonist score."""
        # If block_budget_weight is 0, will return 0.
        if use_shortest_path:
            budget = self.env.get_shortest_path_length()
        else:
            budget = self.env.get_num_blocks()
        weighted_budget = budget * self.adversary_env[env_idx].block_budget_weight
        antag_didnt_score = tf.cast(tf.math.equal(antag_r_max, 0), tf.float32)

        # Number of blocks gives a negative penalty if the antagonist didn't score,
        # else becomes a positive reward.
        block_budget_reward = antag_didnt_score * -weighted_budget + \
            (1 - antag_didnt_score) * weighted_budget

        logging.info('Environment block budget reward: %f',
                     tf.reduce_mean(block_budget_reward).numpy())
        return block_budget_reward
