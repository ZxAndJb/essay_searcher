import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermidate_size = intermediate_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size,bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size,bias=False)

    def forward(self, x, tp=False):
        bs, seq_len, hidden_size = x.shape
        hidden_state = x.clone()
        if tp:
            slice = self.intermidate_size // 2
            fc1_slices = self.fc1.weight.split(slice, dim=0)
            x1 = F.linear(x, fc1_slices[0])
            x2 = F.linear(x, fc1_slices[1])
            x = F.relu(torch.concat([x1, x2], dim = -1))
            x1, x2 = x.split(slice, dim=2)
            fc2_slices = self.fc2.weight.split(slice, dim=1)
            return F.linear(x1, fc2_slices[0]) + F.linear(x2, fc2_slices[1])

        else:
            return self.fc2(F.relu(self.fc1(x)))


# %%
mlp = MLP(128, 512).to("cuda")
inputs = torch.randn((4, 10, 128)).to("cuda")
# %%
mlp(inputs, tp=True)