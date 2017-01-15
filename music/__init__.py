from gym.envs.registration import register
from .music_theory_env import MusicTheoryEnv
from .util import *

register(
    id='music-theory-v0',
    entry_point='music:MusicTheoryEnv'
)
