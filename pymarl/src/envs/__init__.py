from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
# sys.path.append("/home/lhchen/nas/diverse_dpp_marl/ma-gym/")
import os
import gym
from ma_gym.wrappers import Monitor
from ma_gym.wrappers.pymarl_env_wrapper import PyMARLEnvWrapper
from .matrix_game.nstep_matrix_game import NStepMatrixGame

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["grid"] = partial(env_fn, env=PyMARLEnvWrapper)
REGISTRY["nstep_matrix"] = partial(env_fn, env=NStepMatrixGame)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
