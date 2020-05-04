import torch
import torch.nn as nn
torch.manual_seed(1)


def Binarize(x, t=0.5):
    #### ternary
    # mask = (x > 0.5).float() - (x < -0.5).float()
    # return mask

    #### binary
    return torch.where(torch.abs(x) < t, torch.full_like(x, 0), torch.full_like(x, 1))


class BinarizeAttention(nn.Module):
    def __init__(self, inplanes):
        super(BinarizeAttention, self).__init__()
        self.weight = nn.Parameter(torch.randn(inplanes, 1, 1), requires_grad=True)

    def forward(self, x):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = Binarize(self.weight.org)
        if x.device != torch.device('cpu'):
            self.weight.data = self.weight.data.cuda()
        return torch.mul(self.weight, x)

