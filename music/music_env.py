import gym

class MusicEnv(gym.Env):
    """
    An abstract music environment for music composition.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_notes):
        self.observation_space = spaces.Discrete(num_notes)
        self.action_space = spaces.Discrete(num_notes)

    def _step(self, action):
        """
        Parameters:
            - action: An integer that represents the note chosen
        """
        self.composition.append(action)
        return action

    def _reset(self):
        # Composition is a list of notes composed
        self.composition = []

    def _render(self, mode='human', close=False):
        pass
