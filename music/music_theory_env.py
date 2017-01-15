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
        state, reward, done, info = super()._step(action)

        # Compute total rewards
        reward += self.reward_tonic(action)
        reward += self.reward_key(action)
        reward += self.reward_non_repeating(action)
        reward += self.reward_motif(action)
        reward += self.reward_repeated_motif(action)

        # Finished
        """
        done = action == END_COMPOSITION

        if done:
            reward += 1
        """
        return state, reward, done, info

    def reward_key(self, action, penalty_amount=-1, key=C_MAJOR_KEY):
        """
        Applies a penalty for playing notes not in a specific key.
        Args:
          action: Integer of chosen note
          penalty_amount: The amount the model will be penalized if it plays
            a note outside the key.
          key: The numeric values of notes belonging to this key. Defaults to
            C-major if not provided.
        Returns:
          Float reward value.
        """
        return penalty_amount if action not in key else 0

    def reward_non_repeating(self, action):
        """
        Rewards the model for not playing the same note over and over.
        Penalizes the model for playing the same note repeatedly, although more
        repeititions are allowed if it occasionally holds the note or rests in
        between. Reward is uniform when there is no penalty.
        Args:
            action: Integer of chosen note
        Returns:
            Float reward value.
        """
        if not self.detect_repeating_notes(action):
            return 0.1
        return 0

    def detect_repeating_notes(self, action):
        """
        Detects whether the note played is repeating previous notes excessively.
        Args:
          action: An integer representing the note just played.
        Returns:
          True if the note just played is excessively repeated, False otherwise.
        """
        num_repeated = 0
        contains_held_notes = False
        contains_breaks = False

        # Note that the current action is discounted
        for i in range(len(self.composition) - 2, -1, -1):
            if self.composition[i] == action:
                num_repeated += 1
            elif self.composition[i] == NOTE_OFF:
                contains_breaks = True
            elif self.composition[i] == NO_EVENT:
                contains_held_notes = True
            else:
                break

        if action == NOTE_OFF and num_repeated > 1:
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

    def reward_tonic(self, action, tonic_note=C_MAJOR_TONIC, reward_amount=3):
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
        first_note_of_final_bar = self.num_notes - 4

        if self.beat == 0 or self.beat == first_note_of_final_bar:
            if action == tonic_note:
                return reward_amount
        elif self.beat == first_note_of_final_bar + 1:
            if action == NO_EVENT:
                return reward_amount
        elif self.beat > first_note_of_final_bar + 1:
            if action == NO_EVENT or action == NOTE_OFF:
                return reward_amount
        return 0.0

    def reward_motif(self, action, reward_amount=3.0):
        """
        Rewards the model for playing any motif.

        Motif must have at least three distinct notes in the course of one bar.
        There is a bonus for playing more complex motifs; that is, ones that
        involve a greater number of notes.

        Args:
            action: Integer of chosen action
            reward_amount: The amount that will be returned if the last note belongs
            to a motif.
        Returns:
            Float reward value.
        """
        motif, num_notes_in_motif = self.detect_last_motif(self.composition)
        if motif is not None:
            motif_complexity_bonus = max((num_notes_in_motif - 3) * .3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_last_motif(self, composition, bar_length=BEATS_PER_BAR):
        """
        Detects if a motif was just played and if so, returns it.
        A motif should contain at least three distinct notes that are not note_on
        or note_off, and occur within the course of one bar.
        Args:
          composition: The composition in which the function will look for a
            recent motif. Defaults to the model's composition.
          bar_length: The number of notes in one bar.
        Returns:
          None if there is no motif, otherwise the motif in the same format as the
          composition.
        """
        if len(composition) < bar_length:
            return None, 0

        last_bar = composition[-bar_length:]

        actual_notes = [a for a in last_bar if a != NO_EVENT and a != NOTE_OFF]
        num_unique_notes = len(set(actual_notes))
        if num_unique_notes >= 3:
            return last_bar, num_unique_notes
        else:
            return None, num_unique_notes

    def reward_repeated_motif(self,
                              action,
                              bar_length=BEATS_PER_BAR,
                              reward_amount=4.0):
        """
        Adds a big bonus to previous reward if the model plays a repeated motif.
        Checks if the model has just played a motif that repeats an ealier motif in
        the composition.
        There is also a bonus for repeating more complex motifs.
        Args:
          action: One-hot encoding of the chosen action.
          bar_length: The number of notes in one bar.
          reward_amount: The amount that will be added to the reward if the last
            note belongs to a repeated motif.
        Returns:
          Float reward value.
        """
        is_repeated, motif = self.detect_repeated_motif(action, bar_length)
        if is_repeated:
            actual_notes = [a for a in motif if a !=
                            NO_EVENT and a != NOTE_OFF]
            num_notes_in_motif = len(set(actual_notes))
            motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_repeated_motif(self, action, bar_length=8):
        """
        Detects whether the last motif played repeats an earlier motif played.

        Args:
          action: One-hot encoding of the chosen action.
          bar_length: The number of beats in one bar. This determines how many beats
            the model has in which to play the motif.
        Returns:
          True if the note just played belongs to a motif that is repeated. False
          otherwise.
        """
        if len(self.composition) < bar_length:
            return False, None

        motif, _ = self.detect_last_motif(self.composition, bar_length)
        if motif is None:
            return False, None

        prev_composition = self.composition[:-(bar_length - 1) - 1]

        # Check if the motif is in the previous composition.
        for i in range(len(prev_composition) - len(motif) + 1):
            for j in range(len(motif)):
                if prev_composition[i + j] != motif[j]:
                    break
            else:
                return True, motif
        return False, None
