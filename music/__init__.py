from gym.envs.registration import register
from .music_theory_env import MusicTheoryEnv

register(
    id='music-theory-v0',
    entry_point='music:MusicTheoryEnv'
)
