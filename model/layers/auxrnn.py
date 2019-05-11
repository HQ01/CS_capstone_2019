import torch
from torch import nn as nn


class BatchedAuxRNN(nn.Module):
    def __init__(self, beta, lr=1e-3):
        super().__init__()
        self.ratio = beta
        assert 0 < self.ratio < 1, 'beta out of range'
        self.lr = lr

    def forward(self, h, a, mask=None):
        with torch.enable_grad():
            grad = self.find_grad(h, a, mask)
        h = h - self.lr*grad
        a_next = (h.transpose(-1,-2)).matmul(a).matmul(h)

        return h, a_next

    def find_grad(self, h, a, mask):
        h_cp = h.clone().detach().requires_grad_(True)
        # print("h_cp is", h_cp.data)
        a_next = (h_cp.transpose(-1,-2)).matmul(a).matmul(h_cp)
        a_next_norm = torch.norm(a_next, p=1)
        entropy_loss = torch.distributions.Categorical(probs=h_cp).entropy()
        # print("entropy loss data is", entropy_loss.data)
        if mask is not None:
            entropy_loss = entropy_loss * mask.expand_as(entropy_loss)
        # print("entropy loss before normalized is", entropy_loss.sum(-1))
        entropy_loss = entropy_loss.sum(-1) / (a_next.size(1) * a_next.size(2))
        aux_loss = torch.mean(self.ratio * a_next_norm + (1 - self.ratio) * entropy_loss)
        # print('a_next_norm', a_next_norm)
        # print('entropy loss', entropy_loss)
        # print(aux_loss.size())
        # print(aux_loss)
        # raise NotImplementedError
        aux_loss.backward()

        return h_cp.grad