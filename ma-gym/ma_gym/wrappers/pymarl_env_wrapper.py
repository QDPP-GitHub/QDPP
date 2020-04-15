from .pymarl_env_base import MultiAgentEnv
import numpy as np
import gym
import itertools
from copy import deepcopy

class PyMARLEnvWrapper(MultiAgentEnv):
    def __init__(self,  **kwargs):
        args = kwargs.get('env_args', kwargs)
        self.base_env = env = gym.make(args['game_name'])
        self.episode_limit = self.base_env._max_steps
        self.reward_normalizer = args.get('reward_normalizer', 1.)
        self.episode_steps = 0
        self.n_actions = 5
        self.env_args = args
        self.n_agents = self.base_env.n_agents

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
    
    def step(self, actions):
        obs, reward, terminated, info =  self.base_env.step(actions)
        self.episode_steps += 1
        return np.mean(reward) * self.reward_normalizer, np.all(terminated), info

    def get_obs(self):
        """ Returns all agent observations in a list """
        # print('get_obs', self.base_env.get_agent_obs())
        return self.base_env.get_agent_obs()

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        # print('get_obs_agent', agent_id, self.get_obs()[agent_id])
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        # return int
        # print('get_obs_size', len(self.get_obs_agent(0)))
        return len(self.get_obs_agent(0))

    def get_state(self):
        """Returns the global state."""
        has_state_fn = getattr(self.base_env, "get_state", None)
        # print('get_state', self.base_env.get_state())
        # return self.base_env.get_state()
        # return self.get_obs_agent(0)
        if has_state_fn:
            # print('returned state')
            return self.base_env.get_state()
        else:
            return self.get_obs_agent(0)

    def get_state_size(self):
        """ Returns the shape of the state"""
        # print('get_state_size', self.n_agents)
        # return self.n_agents
        # print('get_state_size', self.get_obs_size())
        # return self.get_obs_size()
        has_state_fn = getattr(self.base_env, "get_state", None)
        if has_state_fn:
            return self.base_env.state_size
        else:
            return self.get_obs_size()
        # return self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # print('get_total_actions', self.n_actions)
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self.episode_steps = 0
        self.base_env.reset()
        return self.get_obs(), self.get_state()

    def get_stats(self):
        return None

    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()

    def seed(self):
        self.base_env.seed()

    def save_replay(self):
        pass
    

    def all_obs(self):
        def make_obs(pos, shape):
            pos = [round(pos[0] / (shape[0] - 1), 2), round(pos[1] / (shape[1] - 1), self.get_obs_size())]
            if self.get_obs_size() > 2:
                others = [0.] * (self.get_obs_size() - 2)
                pos = pos + others
            return pos
        x = np.arange(self.base_env._grid_shape[0])
        y = np.arange(self.base_env._grid_shape[1])
        obs = np.asarray(list(map(lambda pos: make_obs(pos, self.base_env._grid_shape), itertools.product(x,y))))
        obs_n = [obs] * self.n_agents
        obs_n = np.concatenate(obs_n, axis=1).reshape(-1, self.n_agents, self.get_obs_size())
        return obs_n

    def get_env_info(self):
        state_num = None
        if getattr(self.base_env, "state_num", None):
            state_num = self.base_env.state_num
        else:
            state_num = self.base_env._grid_shape[0] * self.base_env._grid_shape[1]
        all_obs = None
        if getattr(self.base_env, "all_obs", None):
            all_obs = self.base_env.all_obs()
        else:
            all_obs = self.all_obs()

        env_info = {
                    "state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.base_env.n_agents,
                    "episode_limit": self.episode_limit,
                    'state_num': state_num,
                    "all_obs": all_obs
                    }
        return env_info