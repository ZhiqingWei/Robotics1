#! /usr/bin/env python3

# Import statements
import numpy as np
from simple_grid_env import SimpleGridEnv
from clean_airport_env import CleanAirportEnv
from robot_enums import Heading
from robot_enums import Action
from plotting import ValueFunctionPlotter
from plotting import PolicyPlotter

class GeneralPolicyIteration(object):
    """
    @info: Class constructor.
    @param: Environment (clean airport or simple grid).
    """

    def __init__(self, environment, discount_factor):
        self.env = environment
        self.discount_factor = discount_factor

    """
    @info: Evaluate the given policy.
    @param: policy: state to action mapping.
            theta: Stopping condition.
            max_iteration: Maximum number of iterations before automatically returned.
    @return: V: state-value function.
    """

    def policy_evaluation(self, policy, theta=1e-9, max_iterations=1e3):
        # Initialise value function for each state as zero, V(terminal) = 0, others can be arbitrary
        V = np.zeros([self.env.grid_rows, self.env.grid_cols, 4])
        iteration = 0
        state_number = self.env.grid_rows * self.env.grid_cols * 4

        while True:
            iteration += 1
            delta = 0
            current_state = (0, 0, Heading.NORTH)
            row = current_state[0]
            col = current_state[1]
            act = current_state[2]
            
            for i in range(state_number):
                cur_action = policy[row, col, act]
                new_state, reward, terminal = self.env.dynamics(current_state, action=Heading(cur_action))
                v = V[row, col, act]

                if self.env.is_terminal(current_state):
                    V[row, col, act] = reward
                else:
                    V[row, col, act] = reward + self.discount_factor * V[new_state[0], new_state[1], new_state[2]]

                delta = max(delta, abs(v-V[row, col, act]))

                if act < 3:
                    act += 1
                else:
                    act = 0
                    if col < (self.env.grid_cols - 1):
                        col += 1
                    else:
                        col = 0
                        if row < (self.env.grid_rows - 1):
                            row += 1
                current_state = ([row,col,act])

            if delta < theta:
                return V
            if iteration == max_iterations:
                return V

        return V

    """
    @info:  Improve the given policy.
    @param: policy: state to action mapping.
            V:  State-value function.
    @return: policy: Improved policy.
    """

    def policy_improvement(self, policy, V):

        stable_policy = True
        state_number = self.env.grid_rows * self.env.grid_cols*4
        value_action = np.zeros((state_number, self.env.nA+1))
        current_state = (0, 0, Heading.NORTH)
        row = current_state[0]
        col = current_state[1]
        act = current_state[2]

        for i in range(state_number):
            old_action = policy[row, col, act]

            for j in range(self.env.nA):
                if self.env.is_terminal(current_state):
                    new_state, reward, terminal = self.env.dynamics(current_state, action=Heading(j))
                    value_action[i,j] = reward
                    max_act = 3
                else:
                    new_state, reward, terminal = self.env.dynamics(current_state, action=Heading(j))
                    value_action[i,j] = reward + self.discount_factor * V[new_state[0],new_state[1],new_state[2]]

            if(self.env.is_terminal(current_state) == False):
                max_act = np.argmax(value_action[i, :])
                if(max_act == 3):
                    max_act = old_action

            policy[row, col, act] = max_act

            if old_action != policy[row, col, act]:
                stable_policy = False
            if act < 3:
                act += 1
            else:
                act = 0
                if row < (self.env.grid_rows - 1):
                    row += 1
                else:
                    row = 0
                    if col < (self.env.grid_cols - 1):
                        col += 1

            current_state = ([row, col, act])

        if stable_policy:
            return V, policy
        else:
            self.policy_evaluation(policy)

    """
    @info:  Policy iteration (using iterative policy evaluation).
            max_iteration: Maximum number of iterations before automatically returned.
    @return: policy: State to action mapping.
             V: State-value function.
    """

    def policy_iteration(self, max_iterations=1e3):
        # Initialize a random policy
        policy = np.random.randint(0, self.env.nA, size=(self.env.grid_rows, self.env.grid_cols, len(Heading)))
        V = np.zeros([self.env.grid_rows, self.env.grid_cols, len(Heading)])
        
        # Create functions for plotting

        for i in range(int(max_iterations)):
            print(i)
            p_test = policy.copy()
            V = self.policy_evaluation(policy)
            self.policy_improvement(policy,V)
            if((p_test == policy).all()):
                break
            
        value_function_plotter = ValueFunctionPlotter()
        value_function_plotter.plot(V)
        policy_plotter = PolicyPlotter()
        policy_plotter.plot(policy)
        return policy, V

    """
    @info:  Value iteration
            theta: Stopping condition.
            max_iteration: Maximum number of iterations before automatically returned.
    @return: policy: State to action mapping.
             V: State-value function.
    """

    def value_iteration(self, theta=1e-9, max_iterations=1e3):
        # Initialize a random policy
        policy = np.random.randint(0, self.env.nA, size=(self.env.grid_rows, self.env.grid_cols, len(Heading)))
        V = np.zeros([self.env.grid_rows, self.env.grid_cols, len(Heading)])
        final_action =np.zeros([self.env.grid_rows, self.env.grid_cols, len(Heading)])
        iteration = 0

        while True:
            iteration += 1
            print(iteration)
            delta = 0
            current_state = (0, 0, Heading.NORTH)
            row = current_state[0]
            col = current_state[1]
            act = current_state[2]
            state_number = self.env.grid_rows * self.env.grid_cols*4
            
            for j in range(state_number):
                v = V[row, col, act]
                value_action_ = np.zeros(self.env.nA)
                
                for a in range(self.env.nA):
                    new_state, reward, terminal = self.env.dynamics(current_state, action=Heading(a))
                    if self.env.is_terminal(current_state):
                        action_value = reward
                        value_action_[a] = action_value
                    else:
                        action_value = reward + self.discount_factor * V[new_state[0],new_state[1],new_state[2]]
                        value_action_[a] = action_value
                                
                V[row, col, act] = max(value_action_)
                final_action[row, col, act] = np.argmax(value_action_)
                delta = max(delta, abs(v-V[row, col, act]))

                if act < 3:
                    act += 1
                else:
                    act = 0
                    if col < (self.env.grid_cols - 1):
                        col += 1
                    else:
                        col = 0
                        if row < (self.env.grid_rows - 1):
                            row += 1
                current_state = ([row,col,act])
                
            if(delta < theta):
                break
            if iteration == max_iterations:
                break
                
        cur_state = (0, 0, 0)
        row = cur_state[0]
        col = cur_state[1]
        act = cur_state[2]
        for state in range (state_number):
            
            if(self.env.is_terminal(cur_state)):
                policy[row,col,act] = 3
            else:
                policy[row,col,act] = final_action[row, col, act]
            
            if act < 3:
                    act += 1
            else:
                act = 0
                if col < (self.env.grid_cols - 1):
                    col += 1
                else:
                    col = 0
                    if row < (self.env.grid_rows - 1):
                        row += 1

            cur_state = ([row,col,act])

        # Create functions for plotting
        value_function_plotter = ValueFunctionPlotter()
        value_function_plotter.plot(V)
        policy_plotter = PolicyPlotter()
        policy_plotter.plot(policy)

        return policy, V


if __name__ == '__main__':

    # Which type of environment?
    use_simple_grid = False

    # Which algorithms? Both can be enabled at the same time
    policy_itr = True
    value_itr  = True

    if use_simple_grid:
        # Simple Grid Env
        grid_rows = 3
        grid_cols = 4
        start_state = (0, 0, Heading.NORTH)

        # Simple example
        win_states = [(0, 3)]
        lose_states = [(1, 3)]

        # Simple gridworld
        robot = SimpleGridEnv(grid_rows, grid_cols, start_state, win_states, lose_states)

    else:
        
        # Creating an object of type CleanAirportEnv
        # Locations matching the grid airport
        grid_rows = 7
        grid_cols = 9
        start_state = (0, 0, Heading.EAST)
        cleaning_locations = [(0, 5), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (4, 0), (4, 7)]
        traffic_locations = [(1, 2), (5, 1), (5, 3), (5, 5), (5, 7)]
        trapdoor_location = [(2, 8, Heading.EAST), (4, 8, Heading.EAST)]
        customs_barrier_row = 3
        # Create object with init method requirements
        robot = CleanAirportEnv(grid_rows, grid_cols, start_state, cleaning_locations, traffic_locations,
                                trapdoor_location, customs_barrier_row)
        
    # Run the requested algorithms
    gpi = GeneralPolicyIteration(robot, discount_factor=1.0)
    if policy_itr:
        # Policy iteration
        policys, values = gpi.policy_iteration()
        input("Press Enter to continue...")
    if value_itr:
        # Value iteration
        policy, value = gpi.value_iteration()
        input("Press Enter to continue...")
