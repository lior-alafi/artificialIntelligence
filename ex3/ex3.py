import numpy as np
import pickle
id = ["000000000"]


class Policy:
    """This class is a controller for a pressure plate game."""

    def __init__(self, policy_file='my_policy_file', train=False):
        """Initialize controller for given game model.
        """
        if not train:
            self.load_policy(policy_file)
        else:
            # TODO: you can add input to __init__ for policy generation
            self.save_policy(policy_file)

    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        return np.random.choice(["U", "R", "D", "L"])

    # TODO: you do NOT have to use pickle and is suggested to find an alternative
    def save_policy(self, file):
        with open(file+'.pkl', 'wb') as f:
            pickle.dump(self, f)

    # TODO: you do NOT have to use pickle and is suggested to find an alternative
    def load_policy(self, file):
        with open(file+'.pkl', 'rb') as f:
            obj = pickle.load(f)


