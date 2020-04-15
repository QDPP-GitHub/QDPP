import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qdpp import QDPPMixer
import torch as th
from torch.optim import RMSprop
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from components.projection_selector import project_sample
import matplotlib.pyplot as plt
import time
from utils.timehelper import time_spent


class QDPPQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qdpp":
                self.mixer = QDPPMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.mac.mixer = self.mixer

        if getattr(args, "all_obs", None) is not None:
            shape = self.args.all_obs.shape
            if self.args.device == 'cuda':
                self.logger_batch = {
                    'obs': th.from_numpy(self.args.all_obs.reshape(shape[0], 1, shape[1], shape[2])).float().cuda(),
                    'avail_actions': th.ones(shape[0], 1, shape[1], self.args.n_actions).float().cuda()
                }
            else:
                self.logger_batch = {
                    'obs': th.from_numpy(self.args.all_obs.reshape(shape[0], 1, shape[1], shape[2])).float(),
                    'avail_actions': th.ones(shape[0], 1, shape[1], self.args.n_actions).float()
                }

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps, weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        time_stamp = time.time()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_ind_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_ind_qvals = th.clamp(chosen_action_ind_qvals, self.args.q_min, self.args.q_max)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        target_mac_out = th.clamp(target_mac_out, self.args.q_min, self.args.q_max)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        temperature = self.mac.schedule.eval(t_env) / 2.
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999

            cur_max_actions = project_sample(batch["state"].int(), mac_out_detach, self.mixer, temperature=temperature,
                                             avail_actions=avail_actions, greedy=True)[:,1:]
            target_max_ind_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        else:
            target_states = batch["state"][:,1:].int()
            cur_max_actions = project_sample(target_states, target_mac_out, self.target_mixer, temperature=temperature,
                                             avail_actions=avail_actions, greedy=True)
            target_max_ind_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_ind_qvals, batch["state"][:, :-1], actions, t_env)  #
            target_max_qvals = self.target_mixer(target_max_ind_qvals, batch["state"][:, 1:], cur_max_actions, t_env) #
        chosen_action_qvals = th.clamp(chosen_action_qvals, self.args.q_min, self.args.q_max)
        target_max_qvals = th.clamp(target_max_qvals, self.args.q_min, self.args.q_max)

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if (not getattr(self.args, 'continuous_state', None) or not self.args.continuous_state) \
                and self.args.beta_balance:
            _, sv, _ = th.svd(self.mixer.B.weight)
            sv_p = []
            partition_l = self.args.state_num * self.args.n_actions
            beta = np.sqrt(self.args.n_agents)
            for i in range(self.args.n_agents):
                B_i = self.mixer.B.weight[i*partition_l:(i+1)*partition_l]
                _, sv_i, _ = th.svd(B_i)
                sv_p.append(sv_i)
            sv_p = th.stack(sv_p, dim=0)
            sv = sv.repeat(self.args.n_agents).view(self.args.n_agents, -1)
            raw_beta_balance = (sv / beta - sv_p)
            beta_balance_mask = raw_beta_balance > 0.
            beta_balance = (raw_beta_balance * beta_balance_mask.float()).sum()
            loss += self.args.beta_balance_rate * beta_balance

        # Optimize
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # grad_norm = 0  # TEST
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("temperature", temperature, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            if self.mixer is not None:
                BBT_grad = self.mixer.BBT.grad
                q_grad = self.mixer.q.grad
                det_grad_raw = (BBT_grad /
                                (q_grad.reshape(-1, 1, 1)+(1-mask).reshape(-1, 1, 1))  # avoid nan in masked places
                                ).reshape(-1, self.args.n_agents * self.args.n_agents) * mask.reshape(-1, 1)
                det_grad_norm = det_grad_raw.norm(dim=-1).sum() / th.sum(mask == 1)
                BBT_grad_norm = (BBT_grad.reshape(-1, self.args.n_agents * self.args.n_agents) * mask.reshape(-1, 1))\
                                    .norm(dim=-1).sum() / th.sum(mask == 1)
                q_grad_norm = ((q_grad * mask.reshape(-1,)).abs()).sum() / th.sum(mask == 1)
                det_norm = ((self.mixer.det_BBT * mask.reshape(-1, )).abs()).sum() / th.sum(mask == 1)
                self.logger.log_stat("det_grad_norm", float(det_grad_norm), t_env)
                self.logger.log_stat("det_norm", float(det_norm), t_env)
                self.logger.log_stat("BBT_grad_norm", float(BBT_grad_norm), t_env)
                self.logger.log_stat("q_grad_norm", float(q_grad_norm), t_env)
                self.logger.log_stat("logdet", float((self.mixer.q-self.mixer.q_sum).mean()), t_env)
                self.logger.log_stat("q_sum", float(self.mixer.q_sum.mean()), t_env)
            if getattr(self.args, "all_obs", None) is not None and self.args.log_all_obs:
                qvals = self.mac.forward(self.logger_batch, t=0)
                if self.args.device == 'cuda':
                    self.logger.log_stat('qvals', qvals.detach().cpu().numpy(), t_env)
                else: 
                    self.logger.log_stat('qvals', qvals.detach().numpy(), t_env)
            if self.args.device == 'cuda':
                self.logger.log_stat('B', self.mixer.B.weight.data.cpu().numpy(), t_env)
            else:
                bb = jointq_ind_q = chosen_action_qvals - chosen_action_ind_qvals.sum(dim=-1).unsqueeze(2)
                target_jointq_ind_q = target_max_qvals - target_max_ind_qvals.sum(dim=-1).unsqueeze(2)

                chosen_action_ind_qvals_sum = chosen_action_ind_qvals.sum(dim=-1).unsqueeze(2)
                self.logger.log_stat('chosen_action_ind_qvals_sum', chosen_action_ind_qvals_sum.mean().detach().numpy().astype(np.float64).item() , t_env)
                
                bb_q = bb / chosen_action_ind_qvals_sum
                self.logger.log_stat('bb_q_mean', bb_q.mean().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('bb_q_max', bb_q.max().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('bb_q_min', bb_q.max().detach().numpy().astype(np.float64).item() , t_env)

                self.logger.log_stat('jointq-ind_q_mean',
                                     jointq_ind_q.mean().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('jointq-ind_q_max',
                                     jointq_ind_q.max().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('target_jointq-ind_q_max',
                                     target_jointq_ind_q.max().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('target_jointq-ind_q_mean',
                                     target_jointq_ind_q.mean().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('jointq-ind_q_min',
                                     jointq_ind_q.min().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('target_jointq-ind_q_min',
                                     target_jointq_ind_q.min().detach().numpy().astype(np.float64).item() , t_env)
                self.logger.log_stat('target_q_taken_mean',
                                     (target_max_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat('target_ind_qvals_max',
                                     target_jointq_ind_q.detach().max().numpy().astype(np.float64).item(), t_env)
                self.logger.log_stat('target_ind_qvals_min',
                                     target_jointq_ind_q.detach().min().numpy().astype(np.float64).item(), t_env)
                self.logger.log_stat('target_ind_qvals_mean',
                                     target_jointq_ind_q.detach().mean().numpy().astype(np.float64).item(), t_env)
                self.logger.log_stat('chosen_action_ind_qvals',
                                     chosen_action_ind_qvals.detach().mean().numpy().astype(np.float64).item(), t_env)
                self.logger.log_stat('B', self.mixer.B.weight.data.numpy(), t_env)
                if self.args.all_obs is not None:
                    self.logger.log_stat('states', self.args.all_obs, t_env)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
