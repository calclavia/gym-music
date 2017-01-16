import numpy as np
from .music_env import MusicEnv

class MusicTheoryEnv(MusicEnv):
    """
    Music environment that attempts to clone an existing composition
    """
    def __init__(self):
        super().__init__()
