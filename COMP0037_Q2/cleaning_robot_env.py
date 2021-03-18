#! /usr/bin/env python3

import random
import numpy as np

from robot_enums import Action
from robot_enums import State


# Transition dynamics for Q2 Coursework 1
# Fill out this class for Q2c. extended from Lab 2. The methods have been commented thoroughly.
# The BatterySimulator class simulates transitions of the robot battery from activities it the robot takes.

"""
It is recommended you use Enums to make your code more readable. Simply, Rather than using an integer to represent a 
state or action, we use the Enum. For example:
Rather than having:
if state == 0:
...

you can write
if state == State.HIGH

"""



class CleaningRobotEnv(object):
    """
    @info: Initialise BatterySimulatorEnv object with probabilities for different transitions.
    @param: Alpha, beta, gamma and delta (transition probabilities)
    """

    def __init__(self, alpha=0.4, beta=0.1, gamma=0.1, delta=0.9):
        # State and action definitions
        self.nS = len(State)
        self.nA = len(Action)

        # Transition dynamics p(s'|s,a)
        # key: State and action
        # value: Possible transitions to other states with associated probabilities
        self.transitions = {
            # Search action
            (State.HIGH, Action.SEARCH): {State.HIGH: 1 - alpha, State.MEDIUM: 2 * alpha / 3, State.LOW: alpha / 3},
            (State.MEDIUM, Action.SEARCH): {State.MEDIUM: 1 - beta, State.LOW: beta},
            (State.LOW, Action.SEARCH): {State.LOW: 1 - gamma, State.FLAT: gamma},

            # Wait action
            (State.HIGH, Action.WAIT): {State.HIGH: 1},
            (State.MEDIUM, Action.WAIT): {State.MEDIUM: 1},
            (State.LOW, Action.WAIT): {State.LOW: 1},

            # Recharge action
            (State.HIGH, Action.RECHARGE): {State.HIGH: 1},
            (State.MEDIUM, Action.RECHARGE): {State.HIGH: delta, State.MEDIUM: 1 - delta},
            (State.LOW, Action.RECHARGE):  {State.MEDIUM: delta, State.LOW: 1 - delta},

            # Include FLAT State, no transitions away
            (State.FLAT, Action.SEARCH): {State.FLAT: 1},
            (State.FLAT, Action.RECHARGE): {State.FLAT: 1},
            (State.FLAT, Action.WAIT): {State.FLAT: 1}
        }

        # Reward definition is dictionary with key action and value reward
        self.rewards = {
            Action.SEARCH: 10,
            Action.WAIT: 5,
            Action.RECHARGE: 0,
            State.FLAT: -10  # discharge
        }

        # Initialise starting state to high
        self.state = State.HIGH

    """
    @info: Compute reward calculates the reward based on the given action
    @param: action: Tuple of current position in x, position in y and heading.
    @return: reward value.
    """
    def compute_reward(self, state, action):
        if state == State.FLAT:
            return self.rewards[state]
        else:
            return self.rewards[action]

    """
    @info: Checks if the state is a terminal state.
    @param: state: Tuple of current position in x, position in y and heading.
    @return: True (this is a terminal state) or False (this is not a terminal state).
    """
    def is_terminal(self, state):
        return state == State.FLAT

    """
    @info: Compute transition from current state and action.
    @param: Current state.
            Action.
    @return: next_state
    """
    def reset(self, random_state=False):
        if random_state:
            state = np.random.choice(self.nS)
        else:
            # Start off with high battery
            state = State.HIGH
        self.state = state
        return state

    """
    @info: Step function performs one step in MDP.
    @param: Action.
    @return: next_state, reward, terminal
    """
    def step(self, action):
        # With dynamics function compute all possible state_prob, new_state, reward, terminal
        state_probs, next_states, rewards, terminals = self.dynamics(self.state, action)
        # Sample a new_state, reward and terminal
        i = np.random.choice(np.size(state_probs), p=state_probs)
        self.state = next_states[i]
        return next_states[i], rewards[i], terminals[i]

    """
    @info: Compute transition from current state and action.
    @param: Current state.
            Action.
    @return: next_state
    """
    def dynamics(self, state, action):
        # Get transition probabilities at current state
        transition_probs = self.transitions[(state, action)]

        # Iterate and store through possible states and associated probabilities
        next_states = []
        state_probs = []
        rewards = []
        terminals = []
        for (s, s_prob) in transition_probs.items():
            next_states.append(s)
            state_probs.append(s_prob)
            rewards.append(self.compute_reward(s, action))
            terminals.append(self.is_terminal(s))
        return state_probs, next_states, rewards, terminals

    def render(self):
        print("State: ", self.state)

    def check_probability(self, action,test_size, state):
        next_state_counter_list = [0,0,0,0]
        for i in range(0,test_size):
            state_probs, next_states, rewards, terminals = self.dynamics(action,state)
            j = np.random.choice(np.size(state_probs), p=state_probs)
            next_state = next_states[j]
            if next_state == state.HIGH:
                next_state_counter_list[state.HIGH] += 1
            if next_state == state.MEDIUM:
                next_state_counter_list[state.MEDIUM] += 1
            if next_state == state.LOW:
                next_state_counter_list[state.LOW] += 1
            if next_state == state.FLAT:
                next_state_counter_list[state.FLAT] += 1
        result_list = [0,0,0,0]
        for i in range(0,4):
            result_list[i] = next_state_counter_list[i] / test_size
        return result_list


def print_result_list(result_list):
    print("High Prob: ", result_list[state.HIGH])
    print("Medium Prob: ", result_list[state.MEDIUM])
    print("Low Prob: ", result_list[state.LOW])
    print("Flat Prob: ", result_list[state.FLAT])

# Simple test
if __name__ == '__main__':
    # Instantiate the cleaning robot gym environment.
    robot_env = CleaningRobotEnv()

    # Initialise our environment and get the initial starting battery state.
    state = robot_env.reset()
    # Initialise the previous state to current state to start.
    next_state = state
    # Loop 10 times.
    for i in range(10):
        # Randomly sample an action.
        action = Action.draw_random_action()
        # Step through the environment
        next_state, reward, done = robot_env.step(action)
        # Print out previous state, action, new state, reward and done.
        print('s: ', state)
        print('action: ', action)
        print("s': ", next_state)
        print('reward: ', reward)
        print('done: ', done)
        print('\n')

        # Set old state to current state for next loop.
        state = next_state
        # If termination state reached, reset our environment.
        if done:
            robot_env.reset()
    # Search Test
    print_result_list(robot_env.check_probability(action.SEARCH,100000,state.HIGH))
    print_result_list(robot_env.check_probability(action.SEARCH,100000,state.MEDIUM))
    print_result_list(robot_env.check_probability(action.SEARCH,100000,state.LOW))
    # Wait Test
    print_result_list(robot_env.check_probability(action.WAIT,100000,state.HIGH))
    print_result_list(robot_env.check_probability(action.WAIT,100000,state.MEDIUM))
    print_result_list(robot_env.check_probability(action.WAIT,100000,state.LOW))
    # Recharge Test
    print_result_list(robot_env.check_probability(action.RECHARGE,100000,state.HIGH))
    print_result_list(robot_env.check_probability(action.RECHARGE,100000,state.MEDIUM))
    print_result_list(robot_env.check_probability(action.RECHARGE,100000,state.LOW))


