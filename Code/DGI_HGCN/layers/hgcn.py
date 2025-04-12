import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GCN, SemanticAttentionLayer
from . import GCN, SemanticAttentionLayer

class HGCN(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act, metapath_weight):
        """Dense version of GAT."""
        super(HGCN, self).__init__()
        self.gcn_level_embeddings = []
        self.P = P #number of meta-Path
        self.metapath_weight = metapath_weight
        for _ in range(P):
            self.gcn_level_embeddings.append(GCN(nfeat, nhid, act, bias=True))
            
        for i, gcn_embedding_path in enumerate(self.gcn_level_embeddings):
            
                self.add_module('gcn_path_{}'.format(i), gcn_embedding_path)

        self.semantic_level_attention = SemanticAttentionLayer(nhid, shid, metapath_weight)

        
    def forward(self, x, adjs, sparse):
        meta_path_x = []
        for i, adj in enumerate(adjs):
            m_x = self.gcn_level_embeddings[i](x, adj, sparse)
            meta_path_x.append(m_x)
        
        x = torch.cat([m_x for m_x in meta_path_x], dim=0)
        
        att_1, att_2, x = self.semantic_level_attention(x, self.P)
#        print(x.size())
        x = torch.unsqueeze(x, 0)   
#        print(x.size())
        return att_1, att_2, x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from layers import GCN, SemanticAttentionLayer
# from . import GCN, SemanticAttentionLayer

# class HGCN(nn.Module):
#     def __init__(self, nfeat, nhid, shid, P, act, metapath_weight):
#         """Dense version of GAT."""
#         super(HGCN, self).__init__()
#         self.gcn_level_embeddings = nn.ModuleList()
#         self.P = P  # number of meta-Path
#         self.metapath_weight = metapath_weight
#         for _ in range(P):
#             self.gcn_level_embeddings.append(GCN(nfeat, nhid, act, bias=True))

#         self.semantic_level_attention = SemanticAttentionLayer(nhid, shid, metapath_weight)

#     def forward(self, x, adjs, sparse):
#         meta_path_x = []
#         for i, adj in enumerate(adjs):
#             m_x = self.gcn_level_embeddings[i](x, adj, sparse)
#             meta_path_x.append(m_x)
        
#         x = torch.stack(meta_path_x, dim=1)
#         att_1, att_2, x = self.semantic_level_attention(x, self.P)
        
#         # x should have shape (batch_size, nhid)
#         x = x.mean(dim=1)  # take the mean over meta-paths

#         return att_1, att_2, x
