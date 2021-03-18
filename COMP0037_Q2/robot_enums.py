# Declare common types
from enum import IntEnum

import random

# The actions the robot can do
class Action(IntEnum):
    SEARCH = 0
    WAIT = 1
    RECHARGE = 2

    @classmethod
    def draw_random_action(cls):
        return random.choice([Action.SEARCH, Action.WAIT, Action.RECHARGE])

# The states the battery can take
class State(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    FLAT = 3

