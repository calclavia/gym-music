from gym.envs.registration import register
from .music_env import MusicEnv

register(
    id='music-v0',
    entry_point='music:MusicEnv'
)
