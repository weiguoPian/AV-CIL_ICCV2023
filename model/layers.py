import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

def stable_cosine_distance(a, b, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]

def reduce_proxies(similarities, proxy_per_class):
    # shape (batch_size, n_classes * proxy_per_class)
    n_classes = similarities.shape[1] / proxy_per_class
    n_classes = int(n_classes)
    bs = similarities.shape[0]

    simi_per_class = similarities.view(bs, n_classes, proxy_per_class)
    attentions = F.softmax(simi_per_class, dim=-1)
    return (simi_per_class * attentions).sum(-1)
    

class NormedLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, scale = True):
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = scale
        self.in_features = in_features
        self.out_features = out_features
        if scale:
            self.eta = torch.nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.scale:
            self.eta.data.fill_(1)
        

    def forward(self, x):
        out = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        if self.scale:
            out = self.eta * out
        return out

class SplitNormedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features1, out_features2, scale = True):
        super(SplitNormedLinear, self).__init__()
        # in_features: d, out_features1: old classes, out_features2: new classes
        self.fc1 = NormedLinear(in_features, out_features1, False)
        self.fc2 = NormedLinear(in_features, out_features2, False)
        self.scale = scale
        self.in_features = in_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        if scale:
            self.eta = torch.nn.Parameter(torch.Tensor(1))
            self.eta.data.fill_(1)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        
        if self.scale:
            out = self.eta * out
        
        return out

class LSCLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LSCLinear, self).__init__()
        self.K = 10
        self.out_features = out_features
        self.in_features = in_features
        
        self.weight = torch.nn.Parameter(torch.Tensor(self.K*out_features, in_features))
        
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        
        self.factor = torch.nn.Parameter(torch.tensor(1.))
        
    def forward(self, x):
        # B x out x K
        raw_similarities = -stable_cosine_distance(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        similarities = reduce_proxies(raw_similarities, self.K)
        return similarities
    
class SplitLSCLinear(torch.nn.Module):
    def __init__(self, in_features, out_features1, out_features2):
        super(SplitLSCLinear, self).__init__()
        # in_features: d, out_features1: old classes, out_features2: new classes
        self.fc1 = LSCLinear(in_features, out_features1)
        self.fc2 = LSCLinear(in_features, out_features2)
        
        self.in_features = in_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        
        self.factor = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        
        return out