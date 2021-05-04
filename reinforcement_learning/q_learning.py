import numpy as np


def get_random_action(env):
    """
    Returns a random action for the specific environment.
    :param env: OpenAI Gym environment
    :return: random action
    """
    return env.action_space.sample()


def get_best_action(q_table, state):
    """
    Returns the best action for the current state given the q table.
    :param q_table: q table
    :param state: current state
    :return: best action
    """
    return np.argmax(q_table[state])


def get_action(env, q_table, state, epsilon):
    """
    Returns the best action following epsilon greedy policy for the current state given the q table.
    :param env: OpenAI Gym environment
    :param q_table: q table
    :param state: current state
    :param epsilon: exploration rate
    :return:
    """
    num_actions = env.action_space.n
    probability = np.random.random() + epsilon / num_actions
    if probability < epsilon:
        return get_random_action(env)
    else:
        return get_best_action(q_table, state)


def random_q_table(min_val, max_val, size):
    """
    Returns randomly initialized n-dimensional q table.
    :param min_val: lower bound of values
    :param max_val: upper bound of values
    :param size: size of the q table
    :return: n-dimensional q table
    """
    return np.random.uniform(low=min_val, high=max_val, size=size)


def calculate_new_q_value(q_table, old_state, new_state, action, reward, lr=0.1, discount_factor=0.99):
    """
    Calculates new q value for the current state given the new state, action and reward.
    :param q_table: n-dimensional q table
    :param old_state: old (current) state
    :param new_state: new (next) state
    :param action: action to be taken at state old_state
    :param reward: reward received for performing action
    :param lr: learning rate
    :param discount_factor: discount factor
    :return: new q value for old_state and action
    """
    max_future_q = np.max(q_table[new_state])
    if isinstance(old_state, tuple):
        current_q = q_table[old_state + (action,)]
    else:
        current_q = q_table[old_state, action]
    return (1 - lr) * current_q + lr * (reward + discount_factor * max_future_q)
