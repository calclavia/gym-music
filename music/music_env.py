import random
from gym import Env, spaces
from .util import *

class MusicEnv(Env):
    """
    An abstract music environment for music composition.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Discrete(NUM_CLASSES)
        self.action_space = spaces.Discrete(NUM_CLASSES)
        # Total number of notes
        self.num_notes = 32

    def _step(self, action):
        """
        Args:
            action: An integer that represents the note chosen
        """
        self.composition.append(action)
        self.beat += 1
        return action, 0, self.beat == self.num_notes, {}

    def _reset(self):
        # Start with a random note (except end composition).
        state = random.randint(MIN_CLASS, NUM_CLASSES  -1)
        # Composition is a list of notes composed
        self.composition = [state]
        self.beat = 0
        return state

    def _render(self, mode='human', close=False):
        pass
