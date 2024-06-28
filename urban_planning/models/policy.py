import torch
import torch.nn as nn


class UrbanPlanningPolicy(nn.Module):
    """
    Policy network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        """
        cfg:
        {
            'policy_land_use_head_hidden_size': [32, 1], 
            'policy_road_head_hidden_size': [32, 1]
        }
        agent: UrbanPlanningAgent
        shared_net: SGNNStateEncoder
        """
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        # TODO
        """
        policy_land_use_head:
        Sequential(
            (land_use_linear_0): Linear(in_features=64, out_features=32, bias=True)
            (land_use_tanh_0): Tanh()
            (land_use_linear_1): Linear(in_features=32, out_features=1, bias=False)
            (land_use_flatten_1): Flatten(start_dim=1, end_dim=-1)
        )
        policy_road_head:
        Sequential(
            (road_linear_0): Linear(in_features=16, out_features=32, bias=True)
            (road_tanh_0): Tanh()
            (road_linear_1): Linear(in_features=32, out_features=1, bias=False)
            (road_flatten_1): Flatten(start_dim=1, end_dim=-1)
        )
        """
        self.policy_land_use_head = self.create_policy_head(
            self.shared_net.output_policy_land_use_size, cfg['policy_land_use_head_hidden_size'], 'land_use')
        self.policy_road_head = self.create_policy_head(
            self.shared_net.output_policy_road_size, cfg['policy_road_head_hidden_size'], 'road')

    def create_policy_head(self, input_size, hidden_size, name):
        """Create the policy land_use head."""

        # name: land_use
        # input_size: 4 * gcn_node_dim
        # hidden_size: list
        ##  policy_land_use_head_hidden_size: [32, 1]
        # name: road
        # input_size: gcn_node_dim
        # hidden_size: list
        ##  policy_road_head_hidden_size: [32, 1]


        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            if i < len(hidden_size) - 1:
                policy_head.add_module(
                    '{}_tanh_{}'.format(name, i),
                    nn.Tanh()
                )
            elif hidden_size[i] == 1:
                policy_head.add_module(
                    '{}_flatten_{}'.format(name, i),
                    nn.Flatten()
                )
        """
        policy_land_use_head:
        Sequential(
            (land_use_linear_0): Linear(in_features=4 * gcn_node_dim, out_features=32, bias=True)
            (land_use_tanh_0): Tanh()
            (land_use_linear_1): Linear(in_features=32, out_features=policy_land_use_head_hidden_size[-1], bias=False)
            (land_use_flatten_1): Flatten(start_dim=1, end_dim=-1)
        )
        policy_road_head:
        Sequential(
            (road_linear_0): Linear(in_features=gcn_node_dim, out_features=32, bias=True)
            (road_tanh_0): Tanh()
            (road_linear_1): Linear(in_features=32, out_features=1, bias=False)
            (road_flatten_1): Flatten(start_dim=1, end_dim=-1)
        )
        """
        return policy_head

    def forward(self, x):
        """
        state_policy_land_use: [batch, Max_Num_Edges, 4 * gcn_node_dim]
        state_policy_road: [batch, Max_Num_Nodes, gcn_node_dim]
        state_value: [batch, state_encoder_hidden_size[-1] + 3 * gcn_node_dim + 3]
        land_use_mask: [batch, Max_Num_Edges]
        road_mask: [batch, Max_Num_Nodes]
        stage: [batch, 3]
        """

        state_policy_land_use, state_policy_road, _, land_use_mask, road_mask, stage = self.shared_net(x)

        if stage[:, 0].sum() > 0:
            # stage[:, 0].bool(): [batch]
            # state_policy_land_use[stage[:, 0].bool()]: [land_batch, Max_Num_Edges, 4 * gcn_node_dim]
            # land_use_logits: [land_batch, Max_Num_Edges]
            land_use_logits = self.policy_land_use_head(state_policy_land_use[stage[:, 0].bool()])
            # [land_batch, Max_Num_Edges]
            land_use_paddings = torch.ones_like(land_use_mask[stage[:, 0].bool()], dtype=self.agent.dtype)*(-2.**32+1)
            # [land_batch, Max_Num_Edges]
            masked_land_use_logits = torch.where(land_use_mask[stage[:, 0].bool()], land_use_logits, land_use_paddings)
            land_use_dist = torch.distributions.Categorical(logits=masked_land_use_logits)

        else:
            land_use_dist = None

        if stage[:, 1].sum() > 0:
            # [road_batch, Max_Num_Nodes]
            road_logits = self.policy_road_head(state_policy_road[stage[:, 1].bool()])
            # [road_batch, Max_Num_Nodes]
            road_paddings = torch.ones_like(road_mask[stage[:, 1].bool()], dtype=self.agent.dtype)*(-2.**32 + 1)
            # [road_batch, Max_Num_Nodes]
            masked_road_logits = torch.where(road_mask[stage[:, 1].bool()], road_logits, road_paddings)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
        else:
            road_dist = None

        return land_use_dist, road_dist, stage

    def select_action(self, x, mean_action=False):
        # x == state
        # (4*NUM_TYPES,) (Max_Num_Nodes, NUM_TYPES+10) (Max_Num_Edges, 2) (NUM_TYPES+10,) (Max_Num_Nodes,) (Max_Num_Edges,) (Max_Num_Edges,) (Max_Num_Nodes,) (3,)
        land_use_dist, road_dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        # (batch, 2)
        action = torch.zeros(batch_size, 2, dtype=self.agent.dtype, device=stage.device)
        if land_use_dist is not None:
            # land_use_action: [batch,]
            if mean_action:
                land_use_action = land_use_dist.probs.argmax(dim=1).to(self.agent.dtype)
            else:
                land_use_action = land_use_dist.sample().to(self.agent.dtype)
            action[stage[:, 0].bool(), 0] = land_use_action

        if road_dist is not None:
            if mean_action:
                road_action = road_dist.probs.argmax(dim=1).to(self.agent.dtype)
            else:
                road_action = road_dist.sample().to(self.agent.dtype)
            action[stage[:, 1].bool(), 1] = road_action
        # (batch, 2)
        return action

    def get_log_prob_entropy(self, x, action):
        # x: state list (len=chunk)
        # (4*NUM_TYPES,) (Max_Num_Nodes, NUM_TYPES+10) (Max_Num_Edges, 2) (NUM_TYPES+10,) (Max_Num_Nodes,) (Max_Num_Edges,) (Max_Num_Edges,) (Max_Num_Nodes,) (3,)
        # action: [chunk, 2]
        land_use_dist, road_dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        # [chunk]
        log_prob = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        entropy = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        if land_use_dist is not None:
            land_use_action = action[stage[:, 0].bool(), 0]
            land_use_log_prob = land_use_dist.log_prob(land_use_action)
            log_prob[stage[:, 0].bool()] = land_use_log_prob
            entropy[stage[:, 0].bool()] = land_use_dist.entropy()

        if road_dist is not None:
            road_action = action[stage[:, 1].bool(), 1]
            road_log_prob = road_dist.log_prob(road_action)
            log_prob[stage[:, 1].bool()] = road_log_prob
            entropy[stage[:, 1].bool()] = road_dist.entropy()
        # [chunk, 1], [chunk, 1]
        return log_prob.unsqueeze(1), entropy.unsqueeze(1)
