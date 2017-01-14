import gym

class MusicEnv(gym.Env):
    """
    An abstract music environment for music composition.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def _step(self, action):
        pass

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = []

    def _render(self, mode='human', close=False):
        pass
