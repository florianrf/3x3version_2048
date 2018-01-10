"""
Game 2048 3x3

Following the implementation of Georg Wiese: 
https://github.com/georgwiese/2048-rl

Editing the game setting to a 3x3 format with the purpose
to reduce the number of states. Game states are represented as shape (3, 3) numpy arrays 
whose entries are 0 for empty fields and log2(value) for any tiles.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import Numby library for array calculations
import numpy as np 


# Define Actions
ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game(object):

  def __init__(self, state=None, initial_reward=0): 
    """ first step of game
    arguments:
        state: (3,3) numby array to initialize the state with. If None,
        the state will be initialized with with two random tiles (as done in the original game).
        initial_reward: reward to initialize the Game with.
    """
    self._reward = initial_reward

    if state is None:  # if start of the game -> state =None 
      self._state = np.zeros((3, 3), dtype=np.int) 
      # edit (3,3), start of the game -> empty arrays (zeros)
      self.add_random_tile() # add two random tiles at the beginning
      self.add_random_tile()
    else:
      self._state = state # if not begin of the game...
         

  def copy(self):
    """Return a copy of self."""
    return Game(np.copy(self._state))

## define game over, available actions, do actions, add tiles 

  def game_over(self):
    """Whether the game is over."""
    for action in range(4): # four possible actions
      if self.is_action_available(action):
        return False
      else:
        self._reward = -100
    return True

  def available_actions(self):
    """Computes the set of actions that are available."""
    return [action for action in range(4) if self.is_action_available(action)]

  def is_action_available(self, action):
    """Determines whether action is available.
    That is, executing it would change the state.
    """

    temp_state = np.rot90(self._state, action) # rotate array by 90 degrees
    return self._is_action_available_left(temp_state)

  def _is_action_available_left(self, state):
    """Determines whether action 'Left' is available."""

    # True if any field is 0 (empty) on the left of a tile or two tiles can
    # be merged.
    for row in range(3):
      has_empty = False
      for col in range(3):
        has_empty |= state[row, col] == 0
        if state[row, col] != 0 and has_empty:
          return True # left is possible 
        if (state[row, col] != 0 and col > 0 and
            state[row, col] == state[row, col - 1]):
          return True # left is possible

    return False # else, left is impossible


  def do_action(self, action):
    """Execute action, add a new tile, update & return the reward."""

    temp_state = np.rot90(self._state, action)
    reward = self._do_action_left(temp_state)
    self._state = np.rot90(temp_state, -action)
    self._reward = reward 

    self.add_random_tile()

    return reward

  def _do_action_left(self, state):
    """Exectures action 'Left'."""

    reward = -1

    for row in range(3): 
      # Always the rightmost tile in the current row that was already moved
      merge_candidate = -1 # tile one to the left of rightmost tile
      merged = np.zeros((3,), dtype=np.bool) # bool: binary array (e.g. true/flase
      

      for col in range(3): 
        if state[row, col] == 0:
          continue

        if (merge_candidate != -1 and
            not merged[merge_candidate] and
            state[row, merge_candidate] == state[row, col]):
          # Merge tile with merge_candidate
          state[row, col] = 0
          merged[merge_candidate] = True
          state[row, merge_candidate] += 1 # add value to variable
          if (state[row, merge_candidate] == 6):
              reward = 100
              break

        else:
          # Move tile to the left
          merge_candidate += 1
          if col != merge_candidate:
            state[row, merge_candidate] = state[row, col]
            state[row, col] = 0

    return reward

  def add_random_tile(self):
    """Adds a random tile to the grid. Assumes that it has empty fields."""

    x_pos, y_pos = np.where(self._state == 0)
    assert len(x_pos) != 0
    empty_index = np.random.choice(len(x_pos))
    value = np.random.choice([1, 2], p=[0.9, 0.1]) #add new tile 90% chance add 1, 10% chance add 2

    self._state[x_pos[empty_index], y_pos[empty_index]] = value

  def print_state(self):
    """Prints the current state."""

    def tile_string(value):
      """Concert value to string."""
      if value > 0:
        return '% 5d' % (2 ** value,)
      return "     "

    print ("-" * 25)
    for row in range(3): 
      print ("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
      print ("-" * 25)

  def state(self):
    """Return current state."""
    return self._state

  def reward(self):
    """Return current reward."""
    return self._reward



    


