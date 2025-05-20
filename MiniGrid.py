import gym
from gym import spaces
import numpy as np
from abc import abstractmethod
	
	
class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width: int, height: int):

        self.width: int = width
        self.height: int = height

        self.grid = [None] * (self.width * self.height)
        self.wall_cells = []

    def set(self, i, j, v):
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        if i >= 0 and i < self.width:
            if j >= 0 and j < self.height:
                return self.grid[j * self.width + i]
            else:
                return "Wall"
        else:
            return "Wall"

    def horz_wall(self, x, y, length, v="Wall"):
        for i in range(0, length):
            self.set(x + i, y, v)
            self.wall_cells.append(np.array([x+i,y]))

    def vert_wall(self, x, y, length, v="Wall"):
        for j in range(0, length):
            self.set(x, y + j, v)
            self.wall_cells.append(np.array([x,y+j]))

    def target_marginal(self):
        """Build the target marginal measure when using the marginal matching reward
        """
        rho = np.zeros(self.width * self.height)
        # for each row of rooms
        l = 6

        for i in range(1,4):
            for j in range(1,4):
                rho[j * self.width + i] = 1
        rho[2 * self.width + 2] = 0

        for i in range(1,4):
            for j in range(1+l,4+l):
                rho[j * self.width + i] = 1
        rho[(2+l) * (self.width) + 2] = 0

        for i in range(1+l,4+l):
            for j in range(1,4):
                rho[j * self.height + i] = 1
        rho[2*self.width + 2+l] = 0

        for i in range(1+l,4+l):
            for j in range(1+l,4+l):
                rho[j * self.width + i] = 1
        rho[(2+l)*self.width + 2+l] = 0

        return rho/(8*4)

    def constrained_states(self):
        """Build the constrained states when using the constrained mdp reward
        """
        cons_states = []
        size=11
        for i in range(2):
            for j in range(2):
                for row in range(1,5):
                    if row == 1:
                        cons_states.append((3+j*(5-1)) * size + row + i*8)
                    elif row == 2:
                        cons_states.append((3+j*(5-1)) * size + row + i*6)
                        cons_states.append((2 + j*(5 + 1)) * size + row + i*6)
                    elif row == 3:
                        cons_states.append((3+j*(5-1)) * size + row + i*4)
                        cons_states.append((2 + j*(5 + 1)) * size + row +  i*4)
                        cons_states.append((1 + j*(5 + 3)) * size + row +  i*4)
                        cons_states.append((4 + j*2) * size + row +  i*4)
                    elif row == 4:
                        cons_states.append((3+j*(5-1)) * size + row + i*2)
                        cons_states.append((4 + j*2) * size + row +  i*2)

        return cons_states
                


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11, max_steps=100, noise_prob = 0.05, noise_type='central'):
        self.size = size  # The size of the square grid
        self.max_steps = max_steps

        # Observations are dictionaries with the agent's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "neutral"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        # Initialize grid
        self.grid = Grid(self.size, self.size)

        # Random noise probability vector
        if noise_type == 'up':
            self.p = np.zeros(5) 
            self.p[2] = noise_prob
        elif noise_type == 'central':
            if noise_prob > 0.2:
                noise_prob = 0.2 
            self.p = np.ones(5) * noise_prob
            self.p[-1] = 1-noise_prob * 4

    def _get_obs(self):
        return {"agent": self._agent_location}

    def obs_to_state(self, obs):
        """
        Transform observation to state
        """
        pos = obs["agent"]
        return pos[1] * self.size + pos[0]

    def reset(self, dist, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Generate a new random grid at the start of each episode
        self._gen_grid()

        # Choose the agent's location sampled from initial dist.
        state = np.random.choice((self.size * self.size), p=dist)
        i = state % self.size
        j = state // self.size
        self._agent_location = np.array([i,j])

        # Step count since episode start
        self.step_count = 0

        observation = self._get_obs()

        return observation

    @abstractmethod
    def _gen_grid(self):
        pass

    def P(self, noise_p):
        """
        Build grid probability transition for a given noise vector 
        """
        S = self.size * self.size
        A = self.action_space.n
        P = np.zeros((S, A, S))
        for state in range(S):
            i = state % self.size
            j = state // self.size
            pos = np.array([i,j])
            for a in range(A):
                direction = self._action_to_direction[a]
                next_pos = pos + direction
                for noise in range(A):
                    direction_noise = self._action_to_direction[noise]
                    next_pos_noise = np.clip(
                        next_pos + direction_noise, 0, self.size - 1
                            )
                    fwd_obs = self.grid.get(*next_pos_noise)
                    if fwd_obs == None:
                        next_state = next_pos_noise[1] * self.size + next_pos_noise[0]
                        P[state, a, next_state] += noise_p[noise]
                    else:
                        # increase probability of staying in the same place
                        P[state, a, state] += noise_p[noise]
        return P
            
          
    def initial_state_dist(self):
        mu = np.zeros((self.size * self.size))
        mu[0] = 1
        return mu
    
    def step(self, action):
        self.step_count += 1

        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Sample random noise (element of {0,1,2,3,4}) using probability vector
        epsilon = np.random.choice(5, p=self.p)
        epsilon_direction = self._action_to_direction[epsilon]

        # We use `np.clip` to make sure we don't leave the grid
        fwd_pos = np.clip(
            self._agent_location + direction + epsilon_direction, 0, self.size - 1
        )

        # We make sure we don't pass through a wall
        fwd_position = self.grid.get(*fwd_pos)
        if fwd_position == None:
            self._agent_location = fwd_pos

        # An episode is done iff we reach the max number of steps
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
        reward = 0

        observation = self._get_obs()

        return observation, reward, False, truncated, epsilon

    
