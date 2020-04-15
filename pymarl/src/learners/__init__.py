from .q_learner import QLearner
from .noise_q_learner import QLearner as NoiseQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .qdppq_learner import QDPPQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qdpp_q_learner"] = QDPPQLearner
REGISTRY["noise_q_learner"] = NoiseQLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner