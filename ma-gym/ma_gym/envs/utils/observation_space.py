import gym
import numpy as np


def flat_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Dict):
        return np.sum([flat_dim(v) for v in space.spaces.values()])
    else:
        return np.sum([flat_dim(x) for x in space.spaces])


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True

    def get_agent_obs_size(self, agent_id):
        return flat_dim(self._agents_observation_space[agent_id])

    def get_obs_size(self):
        return np.prod(list(map(flat_dim, self._agents_observation_space)))