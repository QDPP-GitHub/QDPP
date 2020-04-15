import logging

from gym import envs
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register openai's environments as multi agent
# This should be done before registering new environments
env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
for spec in env_specs:
    register(
        id='ma_' + spec.id,
        entry_point='ma_gym.envs.openai:MultiAgentWrapper',
        kwargs={'name': spec.id, **spec._kwargs}
    )

# add new environments : iterate over full observability
for i, observability in enumerate([False, True]):
    register(
        id='CrossOver-v' + str(i),
        entry_point='ma_gym.envs.crossover:CrossOver',
        kwargs={'full_observable': observability, 'step_cost': -0.5}
    )

    register(
        id='Checkers-v' + str(i),
        entry_point='ma_gym.envs.checkers:Checkers',
        kwargs={'full_observable': observability}
    )

    register(
        id='Switch2-v' + str(i),
        entry_point='ma_gym.envs.switch:Switch',
        kwargs={'n_agents': 2, 'full_observable': observability, 'step_cost': -0.1}
    )
    register(
        id='Switch4-v' + str(i),
        entry_point='ma_gym.envs.switch:Switch',
        kwargs={'n_agents': 4, 'full_observable': observability, 'step_cost': -0.1}
    )

    register(
        id='TrafficJunction-v' + str(i),
        entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
        kwargs={'full_observable': observability}
    )


register(
    id='Combat-v0',
    entry_point='ma_gym.envs.combat:Combat',
)
register(
    id='PongDuel-v0',
    entry_point='ma_gym.envs.pong_duel:PongDuel',
)

register(
    id='Spread-v0',
    entry_point='ma_gym.envs.spread:Spread',
)

register(
    id='Spread-v1',
    entry_point='ma_gym.envs.spread:Spread',
    kwargs={
            'reward_type': 'neg'
        }
)

register(
    id='Spread7x7-v0',
    entry_point='ma_gym.envs.spread:Spread',
    kwargs={
            'grid_shape': (7, 7)
        }
)

register(
    id='Spread8x8-v0',
    entry_point='ma_gym.envs.spread:Spread',
    kwargs={
            'grid_shape': (8, 8)
        }
)

register(
    id='Spread9x9-v0',
    entry_point='ma_gym.envs.spread:Spread',
    kwargs={
            'grid_shape': (9, 9)
        }
)

register(
    id='Blocker-v0',
    entry_point='ma_gym.envs.blocker:Blocker',
)

for game_info in [[(5, 5), 2, 1], [(7, 7), 4, 2]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents, n_preys = game_info
    _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
    register(
        id='{}-v0'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
        }
    )
    # fully -observable ( each agent sees observation of other agents)
    register(
        id='{}-v1'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True
        }
    )

    # prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v2'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )

    # full observability + prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v3'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )

    register(
        id='{}-v01'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'pos_obs': True
        }
    )
    # fully -observable ( each agent sees observation of other agents)
    register(
        id='{}-v11'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True, 'pos_obs': True
        }
    )

    # prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v21'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
            'prey_move_probs': [0, 0, 0, 0, 1], 'pos_obs': True
        }
    )

    # full observability + prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v31'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
            'prey_move_probs': [0, 0, 0, 0, 1], 'pos_obs': True
        }
    )
