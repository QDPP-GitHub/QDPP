from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from components.projection_selector import project_sample
from components.epsilon_schedules import DecayThenFlatSchedule

# This multi-agent controller shares parameters between agents
class QDPPMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.mixer = None
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, 
                                              args.epsilon_finish, 
                                              args.epsilon_anneal_time,
                                              decay=args.epsilon_decay_scheme)

        self.action_selector = action_REGISTRY['epsilon_greedy'](args) # for test phase

        self.hidden_states = None

    def set_mixer(self, mixer):
        self.mixer = mixer

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        temperature = self.schedule.eval(t_env)
        # use project and sample to select the actions during the training
        if test_mode is False:
            # project_sample method requires shape like (batch_size, episode_length, other_dims)
            qvals = agent_outputs[bs][None,:,:,:]
            state = ep_batch["state"][:, t_ep][None,:,:]
            chosen_actions = project_sample(state, qvals, self.mixer, temperature=temperature,
                                            avail_actions=avail_actions).view(1, -1).long()
        else:
            if self.args.test_project:  # in test mode, project is not used
                qvals = agent_outputs[bs][None,:,:,:]
                state = ep_batch["state"][:, t_ep][None,:,:]
                chosen_actions = project_sample(state, qvals, self.mixer, temperature=temperature,
                                                avail_actions=avail_actions, greedy=True, project=False).view(1, -1).long()
            else:
                chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs = []
        hidden_states = []
        batch_size = avail_actions.shape[0]
        agent_inputs = agent_inputs.view(batch_size, self.n_agents, -1)
        for i in range(self.n_agents):
            agent_input = agent_inputs[:,i,:]
            hidden_input = self.hidden_states[:,i,:].unsqueeze(0)[:, 0:batch_size]
            agent_out, hidden_state = self.agents[i](agent_input, hidden_input)
            agent_outs.append(agent_out)
            hidden_states.append(hidden_state)
        agent_outs = th.stack(agent_outs, dim=1)
        self.hidden_states = th.stack(hidden_states, dim=1)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        hiddens_states = []
        for i in range(self.n_agents):
            hidden_state = self.agents[i].init_hidden()
            hiddens_states.append(hidden_state)
        self.hidden_states = th.stack(hiddens_states, dim=1).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        paras = []
        for i in range(self.n_agents):
            paras += list(self.agents[i].parameters())
        return paras
        # return self.agent.parameters()

    def load_state(self, other_mac):
        for i in range(self.n_agents):
            self.agents[i].load_state_dict(other_mac.agents[i].state_dict())

    def cuda(self):
        for i in range(self.n_agents):
            self.agents[i].cuda()

    def save_models(self, path):
        for i in range(self.n_agents):
            th.save(self.agents[i].state_dict(), "{}/agent_{}.th".format(path, i))

    def load_models(self, path):
        for i in range(self.n_agents):
             self.agents[i].load_state_dict(th.load("{}/agent_{}.th".format(path, i), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        if hasattr(self.args, 'share_policy') and self.args.share_policy:
            agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
            self.agents = [agent] * self.n_agents
        else:
            self.agents = [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        obs = batch["obs"][:, t]
        bs = obs.shape[0]
        inputs = []
        inputs.append(obs)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=obs.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
