"""
Source:
https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py
"""
# Note values of special actions.
NOTE_OFF = 0
NO_EVENT = 1

# Number of output note classes.
NUM_CLASSES = 38
MIN_CLASS = 2 # First note class

MIN_NOTE = 48  # Inclusive
MAX_NOTE = 84  # Exclusive
BEATS_PER_BAR = 8

# Music theory constants used in defining reward functions.
# Note that action 2 = midi note 48.
C_MAJOR_SCALE = [2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 25, 26]
C_MAJOR_KEY = [0, 1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21, 23, 25, 26, 28,
               30, 31, 33, 35, 37]

C_MAJOR_TONIC = 14
A_MINOR_TONIC = 23
