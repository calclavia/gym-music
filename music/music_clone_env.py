import numpy as np
from .music_env import MusicEnv

label_compositions = []

class MusicCloneEnv(MusicEnv):
    """
    Music environment that attempts to clone an existing composition
    """
    def _step(self, action):
        print('a', action)
        action = np.array(action)
        action, reward, done, info = super()._step(action)

        # Award for action matching example composition
        diff = np.abs(self.label_composition[self.beat] - action)
        diff_ratio = sum(diff) / len(diff)
        reward += self.reward_amount * (1 - diff_ratio)

        return action, reward, done, info

    def _reset(self):
        # TODO: Avoid globals
        # A global list of compositions used as example for training
        global label_compositions
        self.label_composition = np.random.choice(label_compositions)
        self.num_notes = len(self.label_composition)
        self.reward_amount = 1 / self.num_notes
        return super()._reset()
