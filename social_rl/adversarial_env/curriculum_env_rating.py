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
            probs = np.array(probs)
            agnet_entropy = entropy(probs, base=num_actions)
            total_agnet_entropy += agnet_entropy
            # return to orig state
            env.reset_agent()

        return total_agnet_entropy / num_to_sample

    def choose_best_env_idx_by_entropy(self, env_list, policy, policy_state, method):
        scores = []
        for env in env_list:
            scores.append(self.eval_env_entropy(env, policy, policy_state))
        # get env with the closest score of 0.5 # not too hard not too easy
        scores = np.array(scores)
        idx = np.argsort(scores)[len(scores) // 2]
        # idx = (np.abs(scores - 0.5)).argmin()
        custom_printer(f"DEBUG score list: {scores}")
        return idx

    def choose_best_env_idx_by_history(self, env_list, policy, policy_state, method):
        scores = []
        for env in env_list:
            scores.append(self.eval_env_entropy(env, policy, policy_state))
        # get env with the closest score of 0.5 # not too hard not too easy
        scores = np.array(scores)
        idx = np.argsort(scores)[len(scores) // 2]
        # idx = (np.abs(scores - 0.5)).argmin()
        custom_printer(f"DEBUG score list: {scores}")
        return idx
