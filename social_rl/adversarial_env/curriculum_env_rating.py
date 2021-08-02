import numpy as np
from scipy.stats import entropy
from social_rl.custom_printer import custom_printer


class EnvCurriculum(object):
    def __init__(self) -> None:
        self.History = dict()

    def eval_env_entropy(self, env, policy, policy_state):
        # TODO: CALC ~10% of num states
        # entropy(p, base=4) #p = prob vector
        num_to_sample = 15
        total_agnet_entropy = 0
        for i in range(num_to_sample):
            # TODO: FIND OUT MULTIPLE ENVS ISSUES
            time_step = env.reset_agent()
            time_step = env.sample_random_state()

            action_step = policy.distribution(time_step, policy_state)
            num_actions = len(action_step.action.logits_parameter()[0])
            probs = []
            for i in range(num_actions):
                probs.append(action_step.action.prob(i).numpy())
            probs = np.array(probs).reshape(len(probs))
            agnet_entropy = entropy(probs, base=num_actions)

            total_agnet_entropy += agnet_entropy
            # return to orig state
            env.reset_agent()

        return total_agnet_entropy / num_to_sample

    def choose_best_env_idx_by_entropy(self, env_list, policy, policy_state):
        scores = []
        for env in env_list:
            scores.append(self.eval_env_entropy(env, policy, policy_state))
        # get env with the closest score of 0.5 # not too hard not too easy
        scores = np.array(scores)
        idx = np.argsort(scores)[len(scores) // 2]
        # idx = (np.abs(scores - 0.5)).argmin()
        custom_printer(f"DEBUG score list: {scores}, {idx}")
        return idx

    def eval_env_history_dist(self, env):
        env = np.array(env)
        min_dist = np.sum(np.ones_like(env))
        for seen_env in self.History:
            seen_env = np.array(seen_env)
            dist = np.sum(np.abs(seen_env - env))
            if dist < min_dist:
                min_dist = dist
        return min_dist

        # return total_agnet_entropy / num_to_sample

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

        return idx
