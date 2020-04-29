import sys
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import pickle

import gym
from gym import spaces
from gym.utils import seeding

"""
One step actor-critic method in mountain car environment using discrete actions.
"""

#Simulation parameters
NUM_EPISODES = 20000
MAX_T = 200
ALPHA_W = 0.01
ALPHA_THETA = 0.01
GAMMA = 0.9
EPSILON = 0.2
TEMP = 10

#Test flgas
DEBUG = False
RENDER_POLICY = True
MEAN_RANGE = 10
NUM_EPISODES_PLOT = 500


def get_state(env):
    """
    It calculates the vector representation of the current state. In this case,
    it is used a state aggregation representation.
    """
    segmentation_factor = 100 # number of partition on each feature
    pos_segment = (env.high[0] - env.low[0]) / segmentation_factor
    vel_segment = (env.high[1] - env.low[1]) / segmentation_factor
    state = env.state
    coarse_state = np.zeros(2*segmentation_factor)

    coarse_state[int((state[0] - env.low[0])/ pos_segment)] = 1

    coarse_state[int((state[1] - env.low[1])/ vel_segment) + segmentation_factor] = 1

    return coarse_state

def value_approx(state, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights.
    """
    return np.dot(state, weights)

def value_approx_grad(state, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights.
    NOTE: In this case (linear case), this function is useless. Modify it if you
    change the value_approx fuction.
    """
    return state

def calculate_preference(env,s,theta):
    """
    It calculates the preference function which indicate how much is good to
    execute each action from s.
    """
    preferences = np.zeros(env.action_space.n)
    preferences_grad = []

    for a in range(env.action_space.n):
        action_one_hot_vector = np.zeros(env.action_space.n)
        action_one_hot_vector[a] = 1
        s_a = np.zeros(len(theta))

        w_i = 0
        for s_i in s:
            for a_i in action_one_hot_vector:
                s_a[w_i] = s_i*a_i
                w_i = w_i + 1

        preferences[a] = np.dot(s_a, theta)
        preferences_grad.append(s_a)

    return preferences, preferences_grad

def policy(env,s,theta):
    """
    It calculates the policy function and return the probability of executing
    each action. In this case it is used a softmax policy.
    """
    preferences, _ = calculate_preference(env,s,theta)
    pref_exp = np.exp(preferences/TEMP)
    return pref_exp / np.sum(pref_exp)

def policy_grad(env,a,s,theta):
    """
    It calculates the gradient of the softmax policy.
    """
    preferences, preferences_grad = calculate_preference(env,s,theta)

    x_s = preferences_grad
    h_s_a_w = x_s * theta

    f = np.exp(h_s_a_w[a])
    f_prime = np.exp(h_s_a_w[a]) * x_s[a]
    g = np.sum(np.exp(h_s_a_w), axis = 0)
    g_prime = np.sum(np.exp(h_s_a_w) * x_s, axis = 0)

    numerator = f_prime * g - f * g_prime
    denominator = g ** 2
    return numerator / denominator

def training(env, w, theta, rewards):
    """
    It executes NUM_EPISODES episodes of training and returns the total rewards
    and the weights learned (weights w for the value approximation function and
    theta for the policy function).
    """
    for episode in range(NUM_EPISODES):
        time_start = time.time()
        env.reset()
        total_reward = 0

        for t in range(MAX_T):
            s = get_state(env)
            policy_prob = policy(env,s,theta)
            a = np.random.choice(range(env.action_space.n), p = policy_prob)
            _, reward, done, _ = env.step(a)
            s_next = get_state(env)
            delta = reward + value_approx(s_next,w) - value_approx(s,w)
            w = w + ALPHA_W * delta * value_approx_grad(s,w)
            theta = theta + ALPHA_THETA * delta * policy_grad(env,a,s,theta)

            s = s_next
            total_reward = total_reward + reward

            if done:
                break

        rewards[episode] = total_reward
        print("episode time: ", time.time()-time_start)

        # plot
        if (episode+1) % NUM_EPISODES_PLOT == 0:
            plt.plot(range(episode+1), rewards[:episode+1], "b")
            plt.axis([0, episode, np.min(rewards[:episode+1]), np.max(rewards[:episode+1])])
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.pause(0.1)

            if RENDER_POLICY:
                render_policy(env, theta)

    return w, theta, rewards

def render_policy(env, theta):
    """
    It shows the current learned behaviour on the GUI
    """
    env.reset()
    env.render()

    for t in range(MAX_T):
        s = get_state(env)
        policy_prob = policy(env,s,theta)
        a = np.random.choice(range(env.action_space.n), p = policy_prob)
        _, reward, done, _ = env.step(a)
        env.render()

        if done:
            print("I've reached the goal!")
            break

    print("Policy executed.")

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    env.reset()
    env_dim = len(get_state(env))
    rewards = np.zeros(NUM_EPISODES)

    w = np.zeros(env_dim)
    theta = np.zeros(env_dim*env.action_space.n)

    w, theta, rewards = training(env, w, theta, rewards)

    print("Execute final policy...")
    render_policy(env, theta)
    print("Everything is done!")

    env.close()
    plt.show()
