import torch.nn as nn
from ..layers import HGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act, metapath_weight):
        super(DGI, self).__init__()
        self.hgcn = HGCN(nfeat, nhid, shid, P, act, metapath_weight)
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(nhid)

    def forward(self, seq1, seq2, adjs, sparse, msk, samp_bias1, samp_bias2):
        
        att_1, att_2, h_1 = self.hgcn(seq1, adjs, sparse)

        c = self.read(h_1, msk)
        
        c = self.sigm(c)

        att_1, att_2, h_2 = self.hgcn(seq2, adjs, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return att_1, att_2, ret

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        att_1, att_2, h_1 = self.hgcn(seq, adjs, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


# import torch.nn as nn
# from ..layers import HGCN, AvgReadout, Discriminator
# from torch.nn.parallel import DataParallel

# class DGI(nn.Module):
#     def __init__(self, nfeat, nhid, shid, P, act, metapath_weight):
#         super(DGI, self).__init__()
#         self.hgcn = DataParallel(HGCN(nfeat, nhid, shid, P, act, metapath_weight))
#         self.read = AvgReadout()
#         self.sigm = nn.Sigmoid()
#         self.disc = Discriminator(nhid)

#     def forward(self, seq1, seq2, adjs, sparse, msk=None, samp_bias1=None, samp_bias2=None):
#         att_1, att_2, h_1 = self.hgcn(seq1, adjs, sparse)
#         c = self.read(h_1, msk)
#         c = self.sigm(c)
#         att_1, att_2, h_2 = self.hgcn(seq2, adjs, sparse)
#         ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
#         return att_1, att_2, ret

#     # Detach the return variables
#     def embed(self, seq, adjs, sparse, msk=None):
#         att_1, att_2, h_1 = self.hgcn(seq, adjs, sparse)
#         c = self.read(h_1, msk)
#         return h_1.detach(), c.detach()


