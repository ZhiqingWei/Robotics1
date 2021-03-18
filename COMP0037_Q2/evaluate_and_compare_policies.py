#!/usr/bin/env python3

import numpy as np
import re
from cleaning_robot_env import CleaningRobotEnv
from robot_enums import Action
from robot_enums import State
from greedy_agent import EpsilonGreedyAgent
from policies import uniformly_random_policy
from policies import tuned_random_policy



def simulate_policy(environment, policy, max_t):
    state = environment.reset()
    cumulative_reward = 0
    # Loop 10 times.
    for i in range(max_t):
        # Follow policy
        action_prob = policy[state]
        action = np.random.choice(environment.nA, p=action_prob)
        # Step through the environment
        next_state, reward, done = robot_env.step(action)
        cumulative_reward += reward

        # If termination state reached, reset our environment.
        # Otherwise, set the old state to the current state
        # for the next loop
        if done:
            state = robot_env.reset()
        else:
            state = next_state 

    return cumulative_reward



# Iterative Policy evaluation
# Recall Chapter 4 pg. 75 algorithm for policy evaluation.
# Inputs: policy (pi), environment (transition dynamics), discount factor, theta (threshold)
def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    # Initialise value function for each state as zero, V(terminal) = 0, others can be arbitrary
    V = np.zeros(environment.nS)
    # While our value function is worse than the threshold theta
    for iter_counter in range(int(max_iterations)):
        # Keep track of the update done in value function
        delta = 0
        # For each state, look ahead one step at each possible action and next state
        for state in range(environment.nS):
            if environment.is_terminal(state):
                v += rewards
                break
            v = 0
            # The possible next actions, policy[s]:[a,action_prob]
            for action, action_prob in enumerate(policy[state]):
                # For each action, look at the possible next states,
                state_probs, next_states, rewards, terminals = environment.dynamics(state,action)
                for i in range(len(next_states)):
                    v += action_prob * state_probs[i] * (rewards[i] + discount_factor * V[int(next_states[i])])
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        # Stop evaluating once our value function update is below a threshold
        if delta < theta:
            break
    return V



if __name__ == '__main__':
    # Initialise robot environment
    robot_env = CleaningRobotEnv()

    # Evaluate random policy
    # Create a random policy
    random_policy = uniformly_random_policy(robot_env)
    random_reward = simulate_policy(robot_env, random_policy, 100)
    print(random_reward)
    V_random = policy_evaluation(random_policy, robot_env)
    print(V_random)

    # Evaluate the tuned manual policy
    manual_policy = tuned_random_policy(robot_env)
    manual_reward = simulate_policy(robot_env, manual_policy, 100)
    print(manual_reward)
    V_manual = policy_evaluation(manual_policy, robot_env)
    print(V_manual)

    # Evaluate a greedy policy
    # Train
    epsilon_greedy_agent = EpsilonGreedyAgent(robot_env.nS, robot_env.nA)
    state = robot_env.reset()
    # Training
    training_steps = 10000
    for i in range(training_steps):
        action = epsilon_greedy_agent.act(state, explore=True)
        state, reward, done = robot_env.step(action)
        epsilon_greedy_agent.train(state, action, reward)
        if done:
            robot_env.reset()

    greedy_policy = epsilon_greedy_agent.get_greedy_policy()
    print(greedy_policy)

    # Evaluate a manual policy
    greedy_reward = simulate_policy(robot_env, greedy_policy, 100)
    print(greedy_reward)
    V_greedy = policy_evaluation(greedy_policy, robot_env)
    print(V_greedy)
    print("finish printing")
