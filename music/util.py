import numpy as np
"""
Source:
https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py
"""
# Note values of special actions.
NOTE_OFF = 0
NO_EVENT = 1

# Number of output note classes.
MIN_CLASS = 2  # First note class
NUM_CLASSES = MIN_CLASS + 128

# Number of beats in a bar
BEATS_PER_BAR = 4
# The quickest note is a half-note
NOTES_PER_BAR = 2 * BEATS_PER_BAR

# Number of octaves
NUM_OCTAVES = 10

# Music theory constants used in defining reward functions.
# Actions that are in C major
C_MAJOR_KEY = [0, 1]

for o in range(NUM_OCTAVES):
    C_MAJOR_KEY += [MIN_CLASS + i for i in [0, 2, 4, 5, 7, 9, 11]]

C_MAJOR_TONIC = MIN_CLASS + 48
A_MINOR_TONIC = MIN_CLASS + 57

# The number of half-steps in musical intervals, in order of dissonance
OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5

# Indicate melody direction
ASCENDING = 1
DESCENDING = -1

# Indicate whether a melodic leap has been resolved or if another leap was made
LEAP_RESOLVED = 1
LEAP_DOUBLED = -1


def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,), dtype=int)
    arr[i] = 1
    return arr
