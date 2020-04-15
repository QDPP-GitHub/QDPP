import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from utils.timehelper import time_spent
import time
from components.epsilon_schedules import DecayThenFlatSchedule
EPS = 1e-8

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = th.eye(num_classes) 
    return y[labels] 


class QDPPMixer(nn.Module):
    def __init__(self, args):
        super(QDPPMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        print(args)
        if getattr(args, "state_num", None) == None:
            if hasattr(self.args, 'continuous_state') and self.args.continuous_state:
                args.state_num = self.state_dim
            else:
                args.state_num = args.env_args["state_num"]
        self._state_num = args.state_num

        if hasattr(self.args, 'continuous_state') and self.args.continuous_state:
            bias_base = self.n_actions
        else:
            bias_base = self._state_num * self.n_actions
        if self.args.device == 'cuda':
            self.state_bias = th.from_numpy(bias_base * np.arange(0, self.n_agents)).cuda().int()
        else:
            self.state_bias = th.from_numpy(bias_base * np.arange(0, self.n_agents)).int()
        
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, 'logdef_coef', None) is not None:
            self.logdet_coef = args.logdet_coef
        else:
            self.logdet_coef = 1.0
        self.noise_coef = 0.1

        self.B = nn.Embedding(self.n_agents * bias_base,
                              self.embed_dim,
                              max_norm=1.)  # make sure feature vectors are unit vectors
        
        if args.embedding_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.B.weight.data)
        elif args.embedding_init == 'xavier_normal':
            nn.init.xavier_normal_(self.B.weight.data)
        elif args.embedding_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.B.weight.data)
        elif args.embedding_init == 'kaiming_normal':
            nn.init.kaiming_normal_(self.B.weight.data)
        elif args.embedding_init == 'orthogonal':
            nn.init.orthogonal_(self.B.weight.data)
        elif args.embedding_init == 'uniform':
            nn.init.uniform_(self.B.weight.data)
        elif args.embedding_init == 'normal': 
            nn.init.normal_(self.B.weight.data)
        
        self.B.weight.data = F.normalize(self.B.weight.data, p=2, dim=1)
        if self.args.v_baseline:
            self.V_qdpp = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.embed_dim, 1))
        if hasattr(self.args, 'continuous_state') and self.args.continuous_state:
            self.state_encoders = [nn.Sequential(
                nn.Linear(self._state_num + self.n_actions, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            ) for _ in range(self.n_agents)]

        if hasattr(self.args, 'continuous_state') and self.args.continuous_state:
            self.state_encoders = []
            for i in range(self.n_agents):
                _state_encoder = nn.Sequential(
                    nn.Linear(self._state_num + self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
                self.add_module('state_encoder_%i' % i, _state_encoder)
                self.state_encoders.append(_state_encoder)

            self.state_encoders_l = []
            for i in range(self.n_agents):
                _state_encoder_l = nn.Sequential(
                    nn.Linear(self._state_num + self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, 1),
                    nn.Sigmoid()
                )
                self.add_module('state_encoder_l_%i' % i, _state_encoder_l)
                self.state_encoders_l.append(_state_encoder_l)

        self.V_qdpp = nn.Sequential(nn.Linear(self._state_num, self.embed_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.embed_dim, self.embed_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions, t_env):
        n1, n2, n3 = agent_qs.shape
        bs = agent_qs.size(0)
        # the D^1/2 matrix, sqrt(exp(Q^i)) 
        agent_qs = agent_qs.clamp(self.args.q_min, self.args.q_max).view(-1, self.n_agents)
        agent_qs_diag = th.diag_embed(th.exp(agent_qs*0.5))

        actions = actions.reshape(-1, self.n_agents)

        # Get matrix B
        if hasattr(self.args, 'continuous_state') and self.args.continuous_state:
            states = states.reshape(-1, self._state_num)
            idx_bias = self.state_bias.repeat(states.size(0), 1)
            states = states.reshape(-1, 1, states.size(-1))  # [bs, 1, k]
            embeding_idx = idx_bias + actions.int()
            B = self.B(embeding_idx.long())
            _Bs = []
            _Bs_l = []
            for i in range(self.n_agents):
                encoder_input = th.cat((B[:, i:i+1].float(), states.float()), dim=-1)
                _Bs.append(self.state_encoders[i](encoder_input))
                _Bs_l.append(self.state_encoders_l[i](encoder_input))
            B = th.cat(_Bs, 1)  # [bs*step, n_agents, emb_dim
            B_l = th.cat(_Bs_l, 1)  # [bs*step, n_agents, 1]
            B = F.normalize(B, p=2, dim=2) * B_l
            # B = B * B_l

        else:
            states = states.int()
            states = states.reshape(-1, self.state_dim)
            idx_bias = self.state_bias.repeat(states.size(0), 1)
            actions = actions.reshape(-1, self.n_agents)
            embeding_idx = states * self.n_actions + actions + idx_bias
            B = self.B(embeding_idx.long())
            if self.args.embedding_normalization:
                B = F.normalize(B, p=2, dim=2)

        # B = B + (th.rand(size=B.shape) - 0.5) * 2 * 0.1  # TEST add uniform noise from [-0.01, 0.01] or [-0.1, 0.1]

        # Calculate matrix L = D^1/2B^TBD^1/2
        # x = th.bmm(agent_qs_diag, B)  # TEST
        # x = th.bmm(x, th.transpose(B, -1, -2))  # TEST
        # x = th.bmm(x, agent_qs_diag)  # TEST
        BBT = th.bmm(B, th.transpose(B, -1, -2))  # TEST

        BBT = BBT + th.diag_embed(th.rand(size=(B.size(0), B.size(1)))) * self.noise_coef  # TEST add psd noise

        # Calculate logdet
        BBT = F.relu(BBT) + EPS  # make sure elements in L > 0, PSD condition
        self.BBT = BBT
        BBT.retain_grad()
        det_BBT = th.det(BBT)
        self.det_BBT = det_BBT
        logdet_BBT = th.log(det_BBT + EPS)  # make sure det(L) 0, prevent log NaN
        q = agent_qs.sum(dim=-1) + self.logdet_coef * logdet_BBT  # TEST
        self.q = q
        self.q_sum = agent_qs.sum(dim=-1)
        q.retain_grad()

        # A state baseline
        q_qdpp = q.view(n1, n2, 1)
        if self.args.v_baseline:
            v = self.V_qdpp(states.float() / self._state_num).view(n1, n2, 1)
            q_qdpp = q_qdpp + v

        return q_qdpp
