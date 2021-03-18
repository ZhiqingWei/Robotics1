# Declare common types

from enum import Enum
from enum import IntEnum


# The actions the robot can do
class Action(IntEnum):
    FORWARD = 0
    ROTATE_CW = 1
    ROTATE_CCW = 2
    TERMINATE = 3


# The robot's heading
class Heading(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# Helper function to turn an action into a short string for plotting
def action_string(action):
    if action == Action.FORWARD:
        return 'F'
    elif action == Action.ROTATE_CW:
        return 'C'
    elif action == Action.ROTATE_CCW:
        return 'CC'
    else:
        return 'T'
