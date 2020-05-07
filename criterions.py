import torch
import torch.nn as nn


class NTXEntLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def forward(self, h_i, h_j):
        """
        Suppose that h_i=[a,b], h_j=[a',b'], i.e., h_i, h_j are (2, 2048) 2D-array.
        pos: postive pair [aa', bb'] (also [a'a, b'b])
        s_ii: [a,b] x [a,b].T = [[aa,ab],[ba,bb]]
        s_ij: [a,b] x [a',b'].T = [[aa',ab'],[ba',bb']]
        indicator: [[0,1],[1,0]] (aa, bb should be cancelled.)
        S_i: [[aa'+ab+ab'],[bb'+ba+ba']]

        """
        N, C = h_i.size()
        indicator = (torch.ones((N, N), dtype=torch.float32) - torch.eye(N, dtype=torch.float32)).cuda()
        pos = torch.exp(torch.matmul(h_i.view(N, 1, C), h_j.view(N, C, 1)).view(-1) / self.temp)
        s_ii = torch.exp(torch.matmul(h_i, h_i.T) / self.temp)
        s_ij = torch.exp(torch.matmul(h_i, h_j.T) / self.temp)
        S_i = torch.sum(s_ii * indicator + s_ij, dim=1)
        loss_i = -1 * torch.sum(torch.log(pos / S_i)) / N

        s_jj = torch.exp(torch.matmul(h_j, h_j.T) / self.temp)
        s_ji = s_ij.T
        S_j = torch.sum(s_jj * indicator + s_ji, dim=1)
        loss_j = -1 * torch.sum(torch.log(pos / S_j)) / N

        loss = (loss_i + loss_j) / 2
        return loss
