import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def masked_gumbel_softmax(logits, mask, temperature=0.1, mask_fill_value=-1e32):

    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    if mask is not None:
        logits = logits.masked_fill((1 - mask).byte(), mask_fill_value)
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
