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
        super()._step(action)

        # Compute total rewards
        reward = 0
        reward += self.reward_key(action)
        reward += self.reward_non_repeating(action)

        return action, reward, False, {}

    def reward_key(self, action_note, penalty_amount=-1, key=C_MAJOR_KEY):
        """
        Applies a penalty for playing notes not in a specific key.
        Args:
          action_note: Integer of chosen note
          penalty_amount: The amount the model will be penalized if it plays
            a note outside the key.
          key: The numeric values of notes belonging to this key. Defaults to
            C-major if not provided.
        Returns:
          Float reward value.
        """
        return penalty_amount if action_note not in key else 0

    def reward_non_repeating(self, action_note):
        """
        Rewards the model for not playing the same note over and over.
        Penalizes the model for playing the same note repeatedly, although more
        repeititions are allowed if it occasionally holds the note or rests in
        between. Reward is uniform when there is no penalty.
        Args:
            action_note: Integer of chosen note
        Returns:
            Float reward value.
        """
        if not self.detect_repeating_notes(action_note):
            return 0.1
        return 0

    def detect_repeating_notes(self, action_note):
        """
        Detects whether the note played is repeating previous notes excessively.
        Args:
          action_note: An integer representing the note just played.
        Returns:
          True if the note just played is excessively repeated, False otherwise.
        """
        num_repeated = 0
        contains_held_notes = False
        contains_breaks = False

        # Note that the current action yas not yet been added to the composition
        for i in range(len(self.composition) - 1, -1, -1):
          if self.composition[i] == action_note:
            num_repeated += 1
          elif self.composition[i] == NOTE_OFF:
            contains_breaks = True
          elif self.composition[i] == NO_EVENT:
            contains_held_notes = True
          else:
            break

        if action_note == NOTE_OFF and num_repeated > 1:
          return True
        elif not contains_held_notes and not contains_breaks:
          if num_repeated > 4:
            return True
        elif contains_held_notes or contains_breaks:
          if num_repeated > 6:
            return True
        else:
          if num_repeated > 8:
            return True

        return False

    def reward_tonic(self, action_note, tonic_note=C_MAJOR_TONIC, reward_amount=3):
        """
        Rewards for playing the tonic note at the right times.
        Rewards for playing the tonic as the first note of the first bar, and the
        first note of the final bar.
        Args:
          action: Integer of chosen note
          tonic_note: The tonic/1st note of the desired key.
          reward_amount: The amount the model will be awarded if it plays the
            tonic note at the right time.
        Returns:
          Float reward value.
        """
        # TODO: Complete this
        first_note_of_final_bar = self.num_notes_in_melody - 4

        if self.beat == 0 or self.beat == first_note_of_final_bar:
          if action_note == tonic_note:
            return reward_amount
        elif self.beat == first_note_of_final_bar + 1:
          if action_note == NO_EVENT:
            return reward_amount
        elif self.beat > first_note_of_final_bar + 1:
          if action_note == NO_EVENT or action_note == NOTE_OFF:
            return reward_amount
        return 0.0
