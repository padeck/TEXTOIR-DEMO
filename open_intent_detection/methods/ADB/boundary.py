import torch
from torch import nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    """
    Deep Open Intent Classification with Adaptive Decision Boundary.
    https://arxiv.org/pdf/2012.10209.pdf
    """
    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        
        # Automatically detect device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else
                                   "cpu")
        
        self.delta = nn.Parameter(torch.randn(num_labels).to(self.device))  # Ensure delta is on the correct device
        nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels, w=1):
        # Ensure all tensors are on the same device
        pooled_output = pooled_output.to(self.device)
        centroids = centroids.to(self.device)
        labels = labels.to(self.device)  # Ensure labels are on the same device as delta
        
        # Move delta to the correct device
        delta = self.delta.to(self.device)  # Move delta to self.device (same as pooled_output, centroids, and labels)
        delta = F.softplus(delta)

        
        c = centroids[labels]
        d = delta[labels]

        
        euc_dis = torch.norm(pooled_output - c, 2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.float32).to(self.device)
        neg_mask = (euc_dis < d).type(torch.float32).to(self.device)
        
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = w * pos_loss.mean() + neg_loss.mean()
        
        # Final debug print for the loss
        #print(f"Loss: {loss.item()}")
        
        return loss, delta
