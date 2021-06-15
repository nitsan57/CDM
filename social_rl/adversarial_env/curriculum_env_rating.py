import numpy as np
from scipy.stats import entropy


def get_env_rating(env, agnet_act_function):
    # TODO: CALC ~10% of num states
    # entropy(p, base=4) #p = prob vector
    num_to_sample = 15
    total_agnet_entropy = 0
    for i in range(num_to_sample):
        random_state = env.sample_random_state()
        actions_probability = agnet_act_function(random_state)
        agnet_entropy = entropy(actions_probability, base=len(actions_probability))
        total_agnet_entropy += agnet_entropy

    return total_agnet_entropy / num_to_sample


def choose_best_env(env_list):
    scores = []
    for env in env_list:
        scores.append(get_env_rating(env))
    # get env with the closest score of 0.5 # not too hard not too easy

    idx = (np.abs(scores - 0.5)).argmin()
    return env[idx]
