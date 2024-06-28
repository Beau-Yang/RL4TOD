import torch.nn as nn


class UrbanPlanningValue(nn.Module):
    """
    Value network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        """
        cfg:
        {
            'value_head_hidden_size': [32, 32, 1]
        }
        agent: UrbanPlanningAgent
        shared_net: SGNNStateEncoder
        """
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        self.value_head = self.create_value_head(cfg)

    def create_value_head(self, cfg):
        """Create the value head."""
        value_head = nn.Sequential()
        for i in range(len(cfg['value_head_hidden_size'])):
            if i == 0:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.shared_net.output_value_size, cfg['value_head_hidden_size'][i])
                )
            else:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['value_head_hidden_size'][i - 1], cfg['value_head_hidden_size'][i])
                )
            if i < len(cfg['value_head_hidden_size']) - 1:
                value_head.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
        """
        self.shared_net.output_value_size = state_encoder_hidden_size[-1] + 3 * gcn_node_dim + 3
        Sequential(
            (linear_0): Linear(in_features=67, out_features=32, bias=True)
            (tanh_0): Tanh()
            (linear_1): Linear(in_features=32, out_features=32, bias=True)
            (tanh_1): Tanh()
            (linear_2): Linear(in_features=32, out_features=1, bias=True)
        )
        """
        return value_head

    def forward(self, x):
        # state_value: [batch, state_encoder_hidden_size[-1] + 3 * gcn_node_dim + 3]
        _, _, state_value, _, _, _ = self.shared_net(x)
        # value: [batch, 1]
        value = self.value_head(state_value)
        return value
