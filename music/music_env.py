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

    def _step(self, action):
        """
        Args:
            action: An integer that represents the note chosen
        """
        self.composition.append(action)
        self.beat += 1
        return action, 0, False, {}

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = []
        self.beat = 0
        return random.randint(0, NUM_CLASSES)

    def _render(self, mode='human', close=False):
        pass
