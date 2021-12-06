import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNModel(nn.Module):
    def __init__(self, layer_gnn, dim):
        super(GNNModel, self).__init__()
        self.layer_gnn = layer_gnn

        self.W_gnn = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(layer_gnn)])
    
    def gnn(self, xs, A, layer):
        gnn_median = []
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
            temp = torch.mean(xs, 0)
            temp = temp.squeeze(0)
            temp = temp.unsqueeze(0)
            gnn_median.append(temp)
        return gnn_median

    def forward(self, gnn_peptide, gnn_adjacencies):
        fingerprint_vectors = self.embed_fingerprint(gnn_peptide)
        gnn_vectors = self.gnn(fingerprint_vectors, gnn_adjacencies, self.layer_gnn)
        # self.feature1 = gnn_vectors
        return gnn_vectors


