import numpy as np
from .music_env import MusicEnv
from .util import *

class MusicTheoryEnv(MusicEnv):
    """
    Award based on music theory.
    Source:
    https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner.py
    """

    def _step(self, action):
        super.step(action)

        # Compute total rewards
        reward = 0
        reward += reward_key(action)

        return action, reward, False, {}

    def reward_key(self, action_note, penalty_amount=-1.0, key=C_MAJOR_KEY):
        """
        Applies a penalty for playing notes not in a specific key.
        Args:
          action: One-hot encoding of the chosen action.
          penalty_amount: The amount the model will be penalized if it plays
            a note outside the key.
          key: The numeric values of notes belonging to this key. Defaults to
            C-major if not provided.
        Returns:
          Float reward value.
        """
        return penalty_amount if action_note not in key else 0
