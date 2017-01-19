from gym.envs.registration import register
from .music_theory_env import *
from .music_clone_env import *
from .util import *

register(
    id='music-theory-v0',
    entry_point='music:MusicTheoryEnv'
)

register(
    id='music-clone-v0',
    entry_point='music:MusicCloneEnv'
)
