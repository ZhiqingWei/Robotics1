#! /usr/bin/env python3

# Import statements and heading enumeration
import numpy as np
import matplotlib.pyplot as plt
from robot_enums import Action
from robot_enums import Heading


# @class CleanAirportEnv
# @info The CleanAirportEnv class provides the environment in which this problem can be
# represented.
class CleanAirportEnv(object):
    """
     @info: Init function of the class
     @param  grid_rows: Number of rows of the grid map
             grid_cols: Number of cols of the grid map
             cleaning_locs: Cleaning locations (x,y) positions
             traffic_locs: Traffic locations (x,y) positions
             trapdoor_loc: Trapdoor location [(x,y), (x,y)]
             blocked_row: Row of barrier and customs
    """

    def __init__(self, grid_rows, grid_cols, start_state, cleaning_locations, traffic_locations,
                 trapdoor_states, blocked_row):
        # Size of map
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # The number of non-terminal actions; the terminal action is always the
        # last one
        self.nA = len(Action) - 1
        # Where rewards are on map
        self.reward_map = np.zeros([grid_rows, grid_cols])
        # Create the grid map to track robot movement
        self.robot_grid = np.zeros([grid_rows, grid_cols])
        # Initialise state as start_state
        self.start_state = start_state
        self.state = np.asarray(self.start_state)
        # Indicate '1' for robot location on grid
        self.robot_grid[tuple(self.state[0:2])] = 1
        # Save win_states and lose_states for reward calculation
        # Cleaning locations, traffic locations, charge locations
        self.cleaning_locations = cleaning_locations
        self.traffic_locations = traffic_locations
        self.trapdoor_states = trapdoor_states
        self.blocked_row = blocked_row
        for cleaning_location in cleaning_locations:
            self.reward_map[cleaning_location] = 10
        for traffic_location in traffic_locations:
            self.reward_map[traffic_location] = -10
        # Movement table combines action selection and current heading to determine
        # the change in grid position.
        # Eg. Choose to action=Forward and current heading is NORTH, position change is
        # (0, +1).
        # Eg. Choose action=CW and current heading is NORTH, no position change (0,0)
        self.movement_table = \
            {
                (Action.FORWARD, Heading.NORTH): (1, 0),
                (Action.FORWARD, Heading.EAST): (0, 1),
                (Action.FORWARD, Heading.SOUTH): (-1, 0),
                (Action.FORWARD, Heading.WEST): (0, -1),
                # Rotations don't change position
            }

    """
    @info: Compute reward calculates the reward based on the given state. Because only
    considered about position, locations are positions with any heading.
    @param: state: Tuple of current position in x, position in y and heading.
    @return: reward value.
    """

    def compute_reward(self, state, action):
        reward = 0
        # Taking any non-terminal actions result in -1 reward
        if action in [Action.ROTATE_CW, Action.ROTATE_CCW, Action.FORWARD]:
            reward += -1
        position = state[0:2]

        # Taking the terminal action
        if action == Action.TERMINATE:
            if tuple(position) in self.cleaning_locations:
                reward += 10
            if tuple(position) in self.traffic_locations:
                reward += -10
        return reward

    """
    @info: Checks if the state is a terminal state.
    @param: state: Tuple of current position in x, position in y and heading.
    @return: True (this is a terminal state) or False (this is not a terminal state).
    """

    def is_terminal(self, state):
        position = state[0:2]
        if (tuple(position) in self.cleaning_locations) or (tuple(position) in self.traffic_locations):
            return True
        else:
            return False

    """
    info: The step function simulates the gridworld robot by 1 step given an action.
    @param: actions: actions are integers represented by the class actions enum data type.
            # ACTION.forward = 0, move forward
            # ACTION.backward = 1, move backward
            # ACTION.rotate_cw = 2, rotate in place clockwise
            # ACTION.rotate_ccw = 3, rotate in place clockwise
    @return: next state: The resulting state of the robot after the action.
             reward: The associated reward with the action and new state
             terminal: Is the new state a terminal state (True) or not (False)
    """

    def step(self, action):
        # Use dynamics function to determine new state, reward and terminal
        new_state, reward, terminal = self.dynamics(self.state, action)
        # Set state as new state
        self.state = new_state
        # Update the grid position
        self.robot_grid[tuple(self.state[0:2])] = 1
        return new_state, reward, terminal

    """
    @info: Determine the new state, reward and if terminal with a given state and action.
    @param: state: Tuple of current position in x, position in y and heading.
            actions: actions are integers represented by the class actions enum data type.
            # ACTION.forward = 0, move forward
            # ACTION.backward = 1, move backward
            # ACTION.rotate_cw = 2, rotate in place clockwise
            # ACTION.rotate_ccw = 3, rotate in place clockwise
    @return: next state: The resulting state of the robot after the action.
             reward: The associated reward with the action and new state
             terminal: Is the new state a terminal state (True) or not (False)
    """

    def reset(self, random=True):
        # Return a randomly position on the map that is not occupied
        if random:
            valid = False
            # Keep sampling until valid grid state found
            sampled_state = np.random.randint([0, 0, 0], [6, 8, 3])
            while not valid:
                if (tuple(sampled_state[0:2]) in self.cleaning_locations) \
                        or (tuple(sampled_state[0:2]) in self.traffic_locations):
                    valid = False
                    sampled_state = np.random.randint([0, 0, 0], [6, 8, 3])
                elif tuple(sampled_state) in self.trapdoor_states:
                    valid = False
                    sampled_state = np.random.randint([0, 0, 0], [6, 8, 3])
                elif sampled_state[0] == self.blocked_row:
                    valid = False
                    sampled_state = np.random.randint([0, 0, 0], [6, 8, 3])
                else:
                    valid = True
            self.state = sampled_state
            self.robot_grid = np.zeros([self.grid_rows, self.grid_cols])
            self.robot_grid[tuple(self.state[0:2])] = 1
        else:
            # Return robot to starting position
            self.state = self.start_state
        return self.state

    def dynamics(self, state, action):
        # initialise new_state to be zeros
        new_state = np.zeros_like(state)
        # Check if a terminal state, if so take terminal action and compute rewards
        if self.is_terminal(state):
            # Take terminal action and return reward
            reward = self.compute_reward(state, Action.TERMINATE)
            new_state = state
            terminal = True
            return new_state, reward, terminal

        # If action is a forward movement
        elif action == Action.FORWARD:
            # if we're teleporting through trapdoor
            if tuple(state) in self.trapdoor_states:
                if tuple(state) == self.trapdoor_states[0]:
                    new_state = np.asarray(self.trapdoor_states[1])
                    new_state[2] = Heading.WEST
                elif tuple(state) == self.trapdoor_states[1]:
                    new_state = np.asarray(self.trapdoor_states[0])
                    new_state[2] = Heading.WEST
            # Regular non-trapdoor forward movement
            else:
                # Determine the change in position from action and current heading
                movement = self.movement_table[(action, state[2])]
                # Sum current position to change in position
                new_state[0:2] = state[0:2] + np.asarray(movement)
                # Same heading as previous state
                new_state[2] = state[2]
                # Ensure within grid limits
                new_state[0] = np.minimum(new_state[0], self.grid_rows - 1)
                new_state[0] = np.maximum(new_state[0], 0)
                new_state[1] = np.minimum(new_state[1], self.grid_cols - 1)
                new_state[1] = np.maximum(new_state[1], 0)
                # Check for blocked row, if next state is blocked, return to last state
                if new_state[0] == self.blocked_row:
                    new_state = state
        # If action is a rotation
        elif action in [Action.ROTATE_CW, Action.ROTATE_CCW]:
            # If the action is rotating, determine the new heading
            # counter-clockwise rotation
            # Using the Heading Enum, wrap the orientation. Adding moves heading CW
            # subtracting moves heading CCW
            # If at NORTH (0) or WEST (3), use modulus operator to wrap
            # ie. CCW - NORTH (0) - 1 = WEST (3)
            # ie. CW - WEST (3) + 1 = NORTH (0)
            new_state[0:2] = state[0:2]
            if action == Action.ROTATE_CW:
                new_state[2] = np.mod(state[2] + 1, 4)
            # counter-clockwise rotation
            elif action == Action.ROTATE_CCW:
                new_state[2] = np.mod(state[2] - 1, 4)
        # Compute reward
        reward = self.compute_reward(new_state, action)
        # Check terminal state
        terminal = self.is_terminal(new_state)
        return new_state, reward, terminal

    """
    @info: Render a basic visualisation of the robot in the grid world
    """

    def render(self):
        fig, ax = plt.subplots()
        # Initialise an empty grid map to fill
        render_terminal_map = np.zeros([self.grid_rows, self.grid_cols]) - 1
        # Fill in values for win and lose states
        for cleaning_loc in self.cleaning_locations:
            render_terminal_map[cleaning_loc] = 10
        for traffic_loc in self.traffic_locations:
            render_terminal_map[traffic_loc] = -10
        im = ax.imshow(render_terminal_map, origin="lower", cmap="copper")

        # Loop over data dimensions and create text annotations based on heading
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if np.array_equal(np.array([i, j]), self.state[0:2]):
                    if self.state[2] == Heading.NORTH:
                        _ = ax.text(j, i, '^', ha="center", va="center", color="w")
                    if self.state[2] == Heading.EAST:
                        _ = ax.text(j, i, '>', ha="center", va="center", color="w")
                    if self.state[2] == Heading.SOUTH:
                        _ = ax.text(j, i, 'v', ha="center", va="center", color="w")
                    if self.state[2] == Heading.WEST:
                        _ = ax.text(j, i, '<', ha="center", va="center", color="w")
                elif self.robot_grid[i, j] == 1:
                    _ = ax.text(j, i, '*', ha="center", va="center", color="w")
                else:
                    if i is not self.blocked_row:
                        _ = ax.text(j, i, render_terminal_map[i, j], ha="center", va="center", color="w")

        ax.set_xticks(np.arange(self.grid_cols))
        ax.set_yticks(np.arange(self.grid_rows))
        ax.set_title('Position: ' + str((self.state[0], self.state[1])) + ' with Heading: ' + str(self.state[2]))
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        fig.tight_layout()
        plt.show()


# Simple tests
if __name__ == '__main__':
    # Creating an object of type CleanAirportEnv
    # Locations matching the grid airport
    grid_rows = 7
    grid_cols = 9
    start_state = (0, 0, Heading.EAST)
    cleaning_locations = [(2, 0), (2, 1), (2, 2), (1, 4), (1, 5), (0, 5), (4, 0), (4, 7)]
    traffic_locations = [(1, 2), (5, 1), (5, 3), (5, 5), (5, 7)]
    trapdoor_location = [(2, 8, Heading.EAST), (4, 8, Heading.EAST)]
    customs_barrier_row = 3
    # Create object with init method requirements
    robot = CleanAirportEnv(grid_rows, grid_cols, start_state, cleaning_locations, traffic_locations,
                            trapdoor_location, customs_barrier_row)

    # Render the current robot state
    robot.render()

    # Tests:
    # 1. Movement and traffic locations
    print("Testing movement...")
    robot.step(Action.FORWARD)
    robot.step(Action.ROTATE_CCW)
    s, r, d = robot.step(Action.FORWARD)
    print(s, r, d)

    # 2. Test cleaning locations
    print("Testing cleaning reward...")
    robot.state = (1, 0, Heading.NORTH)
    s, r, d = robot.step(Action.FORWARD)
    # Extra step to trigger TERMINAL action
    s, r, d = robot.step(Action.FORWARD)
    print(s, r, d)

    # 3. Test cleaning locations
    print("Testing traffic reward...")
    robot.state = (5, 0, Heading.EAST)
    s, r, d = robot.step(Action.FORWARD)
    # Extra step to trigger TERMINAL action
    s, r, d = robot.step(Action.FORWARD)
    print(s, r, d)

    # 4. Test barriers
    print("Testing barriers...")
    robot.state = (4, 4, Heading.SOUTH)
    s, r, d = robot.step(Action.FORWARD)
    print(s)

    # Test trapdoor
    print("Testing trapdoor...")
    robot.state = (4, 8, Heading.EAST)
    s, r, d = robot.step(Action.FORWARD)
    print(s)
    robot.state = (2, 8, Heading.EAST)
    s, r, d = robot.step(Action.FORWARD)
    print(s)
