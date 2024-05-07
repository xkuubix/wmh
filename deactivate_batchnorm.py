import torch
from torch import nn


def deactivate_batchnorm(net):
    if isinstance(net, nn.BatchNorm2d):
        net.reset_parameters()
        net.eval()

        with torch.no_grad():
            net.weight.fill_(1.0)
            net.bias.zero_()
