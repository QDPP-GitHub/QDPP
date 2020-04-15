import torch as th
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import random


EPS = 1e-8

def gram_schmidt(vv):
    def projection(u, v):
        denom = (u * u).sum() 
        nom = (v * u).sum()
        sim = nom / denom 
        return sim * u

    nk = vv.size(0)
    uu = th.zeros_like(vv, device=vv.device)
    uu[0, :] = vv[0, :].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[j].clone()
            uk = uk + projection(uj, vk)
        # $\mathbf{u}_{k}=\mathbf{v}_{k}-\sum_{j=1}^{k-1} \operatorname{proj}_{\mathbf{u}_{j}}\left(\mathbf{v}_{k}\right)$
        uu[k] = vk - uk
    return uu

def batch_gram_schmidt(vv):
    def projection(u, v):
        denom = (u * u).sum(2) 
        nom = (v * u).sum(2)
        sim = nom / (denom + EPS)
        # sim = nom / denom
        # print('denom', denom)
        if th.any(th.isnan(sim)):
            print('vv', list(vv.cpu().detach().numpy()))
            print('u', list(u.cpu().detach().numpy()))
            print('v', list(v.cpu().detach().numpy()))
        sim = sim.unsqueeze(2).expand_as(u)
        proj_u = sim * u
        return proj_u

    kn= vv.shape[2]
    uu = th.zeros_like(vv, device=vv.device)
    uu[:,:, 0, :] = vv[:,:, 0, :].clone()
    for k in range(1, kn):
        vk = vv[:,:,k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:,:,j].clone()
            uk = uk + projection(uj, vk)
        # $\mathbf{u}_{k}=\mathbf{v}_{k}-\sum_{j=1}^{k-1} \operatorname{proj}_{\mathbf{u}_{j}}\left(\mathbf{v}_{k}\right)$
        uu[:,:,k] = vk - uk
    return uu[:,:,-1]


def project_sample(states, qvals, mixer, temperature=1., avail_actions=None, greedy=False, project=True):
        vs = []
        outs = []
        # batch size, episode length, number of agent, 1
        ori_bs, ep_l, n, _ = qvals.shape

        if (qvals!=qvals).sum().data > 0:
            print(qvals)
        a_idx = list(range(mixer.n_agents))
        if mixer.args.shuffle_sample:
            random.shuffle(a_idx)

        for i in a_idx:
            qvals_i = qvals[:, :, i, :]
            states_i_raw = states[:, :, i]
            qvals_i = qvals_i.view(-1, mixer.n_actions)
            states_i_raw = states_i_raw.view(-1, 1)
            states_i = states_i_raw.repeat(1, mixer.n_actions)
            bs = qvals_i.size(0)

            if mixer.args.device == 'cuda':
                action_bias = th.from_numpy(np.arange(0, mixer.n_actions)).cuda().int()
            else:
                action_bias = th.from_numpy(np.arange(0, mixer.n_actions)).int()
            action_bias = action_bias.repeat(bs, 1)

            if hasattr(mixer.args, 'continuous_state') and mixer.args.continuous_state:
                states_i_raw = states.view(-1, 1, states.size(-1))
                states_i = states_i_raw.repeat(1, mixer.n_actions, 1)
                embeding_idx = mixer.n_actions * i + action_bias
                B_i = mixer.B(embeding_idx.long())
                encoder_input = th.cat((B_i.float(), states_i.float()), dim=-1)
                B_i_raw = mixer.state_encoders[i](encoder_input)
                B_l = mixer.state_encoders_l[i](encoder_input)
                B_i = F.normalize(B_i_raw, p=2, dim=2) * B_l
                # B_i = B_i_raw * B_l
            else:
                embeding_idx = states_i * mixer.n_actions + action_bias + mixer._state_num * mixer.n_actions * i
                B_i = mixer.B(embeding_idx.long())
                if mixer.args.embedding_normalization and project:
                    B_i = F.normalize(B_i, p=2, dim=2)  # NOT normalized in default setting

            if project:
                if len(vs) > 0:
                    tt = th.stack(vs + [B_i], dim=2)
                    B_i = batch_gram_schmidt(tt)

            # Define $q_{a_{i}}:=\left(\left\|S_{a_{i}}^{i}\right\| D_{a_{i}}^{i}\right)^{2}$
            # Alg 1, line 4
            b_norm = B_i.norm(p=2, dim=2).clamp_min(EPS) ** 2
            # print('[Debug] b_norm in project_sample', b_norm)
            # print('[Debug] b_norm.shape in project_sample', b_norm.shape)  # [1, n_actions]

            qvals_i = qvals_i / temperature
            qvals_i_max, _ = th.max(qvals_i, dim=1)
            qvals_i_max = qvals_i_max.unsqueeze(1).expand_as(qvals_i)
            qvals_i = qvals_i - qvals_i_max

            # print('[Debug] qvals_i in project_sample', qvals_i)
            # print('[Debug] qvals_i.shape in project_sample', qvals_i.shape)  # [1, n_actions]

            qvals_i = th.clamp(qvals_i, -20., 20.)  # TEST remove clamp here
            qvals_i = th.exp(qvals_i) * b_norm + EPS # smoothing to prevent zero division
            avail_actions_i = avail_actions.reshape(-1, avail_actions.size(-2), avail_actions.size(-1))[:, i, :].float()

            # Sample a^i from distribution $\left\{\frac{q_{a_{i}}}{\sum_{y \in \mathcal{A}_{i} q_{y}}}\right\}$
            # Alg 1, line 5
            if not greedy:
                # TEST original project_sampler
                qvals_i = (qvals_i + EPS) * avail_actions_i  # [1, n_actions]
                qn = th.sum(qvals_i, dim=1).unsqueeze(1).repeat(1, mixer.n_actions)  # + mixer.n_actions * EPS
                probs = qvals_i / qn
                probs = probs.clamp(EPS, 1.-EPS)  # TEST remove clamp here
                pb = Categorical(probs)
                if th.any(th.isnan(pb.logits)):
                    print('Nan detected in prob dist!')  # nan
                selected_action_i = pb.sample()
                selected_action_i = selected_action_i.view(-1, 1)
                # TEST eps_greedy_sampler
                # pd = Categorical(avail_actions_i)
                # random_action_i = pd.sample().long().view(-1, 1)
                # qvals_i = qvals_i - 9999999. * (1 - avail_actions_i)  # mask unavail actions
                # maxq, selected_action_i = th.max(qvals_i, 1, keepdim=True)
                # random_number = th.rand(random_action_i.shape)
                # random_flag = (random_number < temperature).cuda()
                # selected_action_i = (1. - random_flag) * selected_action_i + random_flag * random_action_i
            else:
                qvals_i = qvals_i - 9999999. * (1-avail_actions_i)  # mask unavail actions
                maxq, selected_action_i = th.max(qvals_i, 1, keepdim=True)

            # Alg2, line 7
            if hasattr(mixer.args, 'continuous_state') and mixer.args.continuous_state:
                v_idx = mixer.n_actions * i + selected_action_i
                v = mixer.B(v_idx.long()).repeat(1, mixer.n_actions, 1)
                encoder_input = th.cat((v.float(), states_i.float()), dim=-1)
                v = mixer.state_encoders[i](encoder_input)
                v_l = mixer.state_encoders_l[i](encoder_input)
                v = F.normalize(v, p=2, dim=2)
                v = v*v_l
                vs.append(v)
            else:
                v_idx = states_i_raw * mixer.n_actions + mixer._state_num * mixer.n_actions * i + selected_action_i
                v = mixer.B(v_idx.long()).repeat(1, mixer.n_actions, 1)
                if mixer.args.embedding_normalization:
                    v = F.normalize(v, p=2, dim=2)

            vs.append(v)
            outs.append(selected_action_i)

        if mixer.args.shuffle_sample:
            outs_inorder = []
            for i in range(mixer.n_agents):
                ii = a_idx.index(i)
                outs_inorder.append(outs[ii])
            outs = outs_inorder
        outs = th.stack(outs, dim=1).view(ori_bs, ep_l, n, 1)

        return outs.detach()