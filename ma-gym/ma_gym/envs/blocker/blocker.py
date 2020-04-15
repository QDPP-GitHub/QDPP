import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_cell_outline, draw_circle, write_cell_text

logger = logging.getLogger(__name__)


class Blocker(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable=False, step_cost=-1., n_agents=3, max_steps=40):
        self._grid_shape = (4, 7)
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self._total_episode_reward = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])  # l,r,t,d,noop

        mid_h = int(self._grid_shape[0] / 2)
        mid_v = int(self._grid_shape[1] / 2)
        self.state_size = n_agents
        init_pos = np.random.choice(7, 3, replace=False)
        self.init_agent_pos = {i: [0, init_pos[i]]  for i in range(self.n_agents)}
        self.final_agent_pos = {0: [3, 0], 
                                1: [3, 3],
                                2: [3, 6], 
                                }  # they have to go in opposite direction
        self.blocker_position = (3, 3)

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self.__init_full_obs()
        self.viewer = None

        self.full_observable = full_observable
        # agent pos (2)
        self._obs_high = np.array([1., 1.])
        self._obs_low = np.array([0., 0.])
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def _update_grid(self):
        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self.__draw_base_img()
        # self.__init_full_obs()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR)

        # for agent_i, pos in list(self.final_agent_pos.items())[:self.n_agents]:
        #     row, col = pos[0], pos[1]
        #     draw_cell_outline(self._base_img, (row, col), cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])

    def __create_grid(self):
        _grid = np.zeros(self._grid_shape)  # all are walls
        # _grid[self._grid_shape[0] // 2, :] = 0  # road in the middle
        for i in range(7):
            if i != self.blocker_position[1]:
                _grid[3, i] = -1
            else:
                _grid[3, i] = 0
        return _grid
        

    def __init_full_obs(self):
        self.agent_pos = copy.copy(self.init_agent_pos)
        self._full_obs = self.__create_grid()
        for agent_i, pos in self.agent_pos.items():
            self.__update_agent_view(agent_i)
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(0, self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [round(pos[0] / (self._grid_shape[0] - 1), 2),
                            round(pos[1] / (self._grid_shape[1] - 1), 2)]
            # _agent_i_obs += [self._step_count / self._max_steps]  # add current step count (for time reference)
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]

        return _obs

    def reset(self):
        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return self._base_grid[row, col] == -1

    def _is_cell_vacant(self, pos):
        is_valid = (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])
        return is_valid and (self._full_obs[pos[0], pos[1]] == 0)

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0], curr_pos[1]] = 0
            self.__update_agent_view(agent_i)
        else:
            pass

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0], self.agent_pos[agent_i][1]] = agent_i + 1

    def __is_agent_done(self, agent_i):
        return self.agent_pos[agent_i] == self.final_agent_pos[agent_i]

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]
        for agent_i, action in enumerate(agents_action):
            self.__update_agent_pos(agent_i, action)

        agent_pos_tuple = tuple(tuple(pos) for pos in self.agent_pos.values())
        # print(agent_pos_tuple)
        if self.blocker_position in agent_pos_tuple:
            for agent_i in range(self.n_agents):
                self._agent_dones[agent_i] = True
                if self._agent_dones[agent_i]:
                    rewards[agent_i] = 1.
        else:
            if self.blocker_position == (3, 3):
                if (2, 3) in agent_pos_tuple:
                    
                    if (2, 0) not in agent_pos_tuple:
                        # print('case 1')
                        self.blocker_position = (3, 0)
                    else:
                        # print('case 2')
                        self.blocker_position = (3, 6)
            elif self.blocker_position == (3, 0) or self.blocker_position == (3, 6):
                if (2, 0) in agent_pos_tuple or (2, 6) in agent_pos_tuple:
                    if (3, 3) not in agent_pos_tuple:
                        # print('case 3')
                        self.blocker_position = (3, 3)
            self._update_grid()

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        info = {'state_num': self._grid_shape[0] * self._grid_shape[1]}
        pos_idxs = []
        for i in range(self.n_agents):
            x, y = self.agent_pos[0], self.agent_pos[1]
            idx =  x * self._grid_shape[1] + y
            pos_idxs.append(idx)
        info['pos_idxs'] = self.get_state()

        return self.get_agent_obs(), rewards, self._agent_dones, info

    def get_state(self):
        pos_idxs = []
        for i in range(self.n_agents):
            x, y = self.agent_pos[i][0], self.agent_pos[i][1]
            idx =  x * self._grid_shape[1] + y
            pos_idxs.append(idx)
        return np.array(pos_idxs)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i], radius=0.3)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange'
}

CELL_SIZE = 30

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}
