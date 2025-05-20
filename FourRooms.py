import gym
from gym import spaces
import numpy as np
from MiniGrid import GridWorldEnv, Grid


class FourRoomsEnv(GridWorldEnv):

    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. The agents always start in the top left corner of the grid.

    ## Mission Space

    No mission. 

    ## Action Space

    | Num | Action       |
    |-----|--------------|
    | 0   | Move right   |
    | 1   | Move up      |
    | 2   | Move down    |
    | 3   | Move left    |
    | 4   | Stay still   |


    ## Termination

    The episode ends if any one of the following condition is met:

    Timeout (see `max_steps`).

    """

    def __init__(self, max_steps=100, **kwargs):
        # self._agent_default_pos = agent_pos

        self.size = 11

        super().__init__(
            max_steps=max_steps,
            size=self.size,
            **kwargs,
        )

    
    def _gen_grid(self):
        # Create the grid
        self.grid = Grid(self.size, self.size)

        room_w = self.size // 2 # each room width
        room_h = self.size // 2 # each room height

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door 
                if i == 0: 
                    self.grid.vert_wall(xR, yT, room_h + j)
                    pos = (xR, int((yT + j + yB)/2))
                    self.grid.set(*pos, None)
                    self.grid.wall_cells = [ x for x in self.grid.wall_cells if not (x==np.array([xR, int((yT + j+ yB)/2)])).all()]

                # Bottom wall and door
                if j == 0:
                    self.grid.horz_wall(xL, yB, room_w + i)
                    pos = (int((xL + i + xR)/2), yB)
                    self.grid.set(*pos, None)
                    self.grid.wall_cells = [ x for x in self.grid.wall_cells if not (x==np.array([int((xL + i+ xR)/2), yB])).all()]

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """
        return self.np_random.integers(low, high)


