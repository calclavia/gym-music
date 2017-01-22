import numpy as np
import random
from .music_env import MusicEnv

label_compositions = []

class MusicCloneEnv(MusicEnv):
    """
    Music environment that attempts to clone an existing melody.
    """
    def _step(self, action):
        state, reward, done, info = super()._step(action)

        # Award for action matching example composition
        if action == self.label_composition[self.beat - 1]:
            reward += self.reward_amount

        if self.beat == len(self.label_composition):
            done = True

        return state, reward, done, info

    def _reset(self):
        # TODO: Avoid globals
        # A global list of compositions used as example for training
        global label_compositions
        self.label_composition = random.choice(label_compositions)
        self.num_notes = len(self.label_composition)
        self.reward_amount = 1. / self.num_notes

        self.composition = [self.label_composition[0]]
        self.beat = 0
        return self._current_state()
