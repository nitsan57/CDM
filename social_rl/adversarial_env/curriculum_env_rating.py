from __future__ import print_function
from operator import le
import numpy as np
from scipy.stats import entropy
from social_rl.custom_printer import custom_printer
import os
import pickle
import cv2
import tensorflow as tf

class EnvCurriculum(object):
    def __init__(self, root_dir, mode, agent_type) -> None:

        self.History = dict()
        self.curr_d_param = 1 # param change rate
        self.params_vector = []
        self.mode = mode
        self.agent_type = agent_type
        
        if mode == "history":
            f_name = os.path.join(root_dir,"history.pickle")
            self.f_name = f_name
            if os.path.exists(f_name):
                with open(f_name, 'rb') as handle:
                    self.History = pickle.load(handle)

        if mode == "search":
            f_name = os.path.join(root_dir,"search.pickle")
            self.f_name = f_name
            if os.path.exists(f_name):
                with open(f_name, 'rb') as handle:
                    search_parmas = pickle.load(handle)
                    self.params_vector = search_parmas['vector']
                    self.curr_d_param = search_parmas['d_param']
            else:
                custom_printer("SEARCH FILE DOESNT EXISTS!!!!")

    def eval_env_entropy(self, env, agent):
        # TODO: CALC ~10% of num states
        # entropy(p, base=4) #p = prob vector
        num_to_sample = 25
        total_agnet_entropy = 0

        policy = agent.collect_policy
        policy_state = policy.get_initial_state(env.batch_size)

        for i in range(num_to_sample):
            # TODO: FIND OUT MULTIPLE ENVS ISSUES
            time_step = env.reset_agent()
            time_step = env.sample_random_state()
            probs = []
            if self.agent_type == "dqn":            
                q_values = agent.tf_agent._q_network(time_step.observation, time_step.step_type)[0]
                q_probs = tf.nn.softmax(q_values)
                probs = q_probs
            else:
                action_step = policy.distribution(time_step, policy_state)
                for i in range(num_actions):
                    probs.append(action_step.action.prob(i).numpy())

            num_actions = policy.policy_step_spec.action.maximum - policy.policy_step_spec.action.minimum + 1

            probs = probs[0]
            agnet_entropy = entropy(probs, base=num_actions)

            total_agnet_entropy += agnet_entropy
            # return to orig state
            env.reset_agent()

        return total_agnet_entropy / num_to_sample

    def choose_best_env_idx_by_entropy(self, env_list,agent, return_seq=False):
        scores = []
        for env in env_list:
            scores.append(self.eval_env_entropy(env, agent))
        # get env with the closest score of 0.5 # not too hard not too easy
        scores = np.array(scores)
        idx = np.argsort(scores)[len(scores) // 2]
        # idx = (np.abs(scores - 0.5)).argmin()
        custom_printer(f"DEBUG entropy sampled: {scores[idx]}")
        if return_seq is False:
            return idx
        else:
            return idx, scores


    def eval_env_history_dist(self, env):
        env_param= env._envs[-1].param_vector
        min_dist = np.sum(np.ones_like(env_param)*np.max(env_param))
        for seen_env in self.History:
            seen_env = np.array(seen_env)
            dist = np.sum(np.abs(seen_env - env_param))
            if dist < min_dist:
                min_dist = dist
        return min_dist

        # return total_agnet_entropy / num_to_sample

    def save_history(self):
        f_name = self.f_name
        with open(f_name, 'wb') as handle:
            pickle.dump(self.History, handle)

    def choose_best_env_idx_by_history(self, env_list):
        new_envs_list = []

        for env in env_list:
            if env not in self.History:
                new_envs_list.append(env)

        scores = []
        for new_env in new_envs_list:
            dist = self.eval_env_history_dist(new_env)
            scores.append(dist)
        if len(scores) != 0:
            # We have new unseen environments
            scores = np.array(scores)
            idx = np.argsort(scores)[len(scores) // 2]
            return idx
        else:
            # return the lowest rewarded from history
            min_reward = None
            min_reward_env_idx = 0
            for i, env in enumerate(env_list):
                reward = self.History(env)
                if i == 0:  # init a value
                    min_reward = reward
                if reward < min_reward:
                    min_reward_env_idx = i
            return min_reward_env_idx



    def create_env_greedy(self, env_list, agent):
        policy = agent.collect_policy
        policy_state = policy.get_initial_state(env.batch_size)

        last_param_vector = self.params_vector
        adversary_action_dim = env_list[0]._envs[-1].adversary_action_dim
        max_length = env_list[0]._envs[-1].adversary_max_steps

        if self.params_vector == []:
            self.params_vector = np.zeros(max_length)




        max_var_bound = (adversary_action_dim**2) / 4
        all_params_vectors = []
        for i,env in enumerate(env_list):
            env.reset()
            changed_params = 0
            current_param_vector = np.copy(self.params_vector)
            while changed_params < self.curr_d_param:
                chosen_idx = np.random.choice(max_length)
                chosen_d_param = np.random.choice(adversary_action_dim)
                old_param = current_param_vector[chosen_idx]
                current_param_vector[chosen_idx] = chosen_d_param

                changed_params +=1

            all_params_vectors.append(current_param_vector)

            for j in range(len(current_param_vector)):
                adversary_action = tf.convert_to_tensor(np.array([current_param_vector[j]]))
                env.step_adversary(adversary_action)

        # import matplotlib.pyplot as plt
        # for i,e in enumerate(env_list):
        #     print(all_params_vectors[i])
        #     plt.imshow(e._envs[-1].render())
        #     plt.show()

        idx, scores = self.choose_best_env_idx_by_entropy(env_list, agent, return_seq=True)
        curr_step_variance = np.var(scores)
        if (curr_step_variance / max_var_bound) < 0.2 and self.curr_d_param < (adversary_action_dim*max_length // 2):
            #increase parameter change by 1 if variance too small
            self.curr_d_param +=1
        self.params_vector = all_params_vectors[idx]       


        f_name = self.f_name
        search_parmas = {}
        with open(f_name, 'wb') as handle:
            search_parmas['vector'] = self.params_vector
            search_parmas['d_param'] = self.curr_d_param
            pickle.dump(search_parmas, handle)


        return idx