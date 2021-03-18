#! /usr/bin/env python3

import numpy as np

from cleaning_robot_env import CleaningRobotEnv
from robot_enums import Action
from robot_enums import State

# Model answer from lab 2


class EpsilonGreedyAgent(object):
    def __init__(self, nS, nA, epsilon=0.1):
        # epsilon: probability for exploration in epsilon-greedy algorithm
        self.epsilon = epsilon
        self.action = [Action.SEARCH,Action.WAIT,Action.RECHARGE]
        self.state = [State.LOW,State.MEDIUM,State.HIGH]
        self.nS = nS
        self.nA = nA

        # estimation for each action
        self.q_estimation = np.zeros([self.nS, self.nA])

        # 1/n this is from the lecture 3
        self.n = np.zeros([self.nS, self.nA])

    def act(self, state, explore=True):
        if (np.random.rand() < self.epsilon) and explore:
            # Random (exploration) action
            return np.random.choice(self.nA)
        else:
            # Pick greedy optimal solution
            q_best = np.max(self.q_estimation[state])
            action = np.random.choice(np.where(self.q_estimation[state] == q_best)[0])
            return action

    def train(self, state, action, reward):
        # 1/n this is from the lecture 3
        self.n[state][action] += 1
        self.step_size = 1 / self.n[state][action]

        self.q_estimation[state][action] += \
            self.step_size * (reward
                              - self.q_estimation[state][action])

    def get_greedy_policy(self):
        policy = np.zeros([self.nS, self.nA])
        for state in range(self.nS):
            q_best = np.max(self.q_estimation[state])
            action_idx = np.random.choice(np.where(self.q_estimation[state] == q_best)[0])
            policy[state][action_idx] = 1.0
        return policy


if __name__ == '__main__':
    robot_env = CleaningRobotEnv()
    agent = EpsilonGreedyAgent(robot_env.nS, robot_env.nA)
    state = robot_env.reset()
    # Training
    training_steps = 100000
    for i in range(training_steps):
        action = agent.act(state, explore=True)
        state, reward, done = robot_env.step(action)
        agent.train(state, action, reward)
        if done:
            robot_env.reset()

    print("Testing greedy policy")
    cumulative_reward = 0
    state = robot_env.reset()
    previous_state = state
    for i in range(10):
        action = agent.act(state,explore=False)
        state, reward, done = robot_env.step(action)
        print("s,s',reward::",previous_state,state,action,reward)
        previous_state = state
        cumulative_reward += reward
        if done:
            state = robot_env.reset()

    print(cumulative_reward)
