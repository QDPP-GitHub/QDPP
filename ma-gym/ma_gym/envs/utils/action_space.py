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


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

    def get_agent_action_size(self, agent_id):
        return flat_dim(self._agents_action_space[agent_id])

    def get_action_size(self):
        return np.prod(list(map(flat_dim, self._agents_action_space)))
