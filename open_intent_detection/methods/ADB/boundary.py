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
        
        # Dynamically assign device based on availability of MPS, CUDA, or fallback to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize delta as a parameter on the correct device
        self.delta = nn.Parameter(torch.randn(num_labels).to(self.device))
        nn.init.normal_(self.delta)

    def forward(self, pooled_output, centroids, labels, w=1):
        delta = F.softplus(self.delta)
        
        # Ensure that centroids and labels are moved to the correct device
        centroids = centroids.to(self.device)
        labels = labels.to(self.device)
        
        # Get the corresponding centroid and delta values for the current labels
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output.to(self.device)
        
        # Compute Euclidean distance
        euc_dis = torch.norm(x - c, p=2, dim=1).view(-1)
        
        # Create masks based on distance thresholds
        pos_mask = (euc_dis > d).type(torch.float32).to(self.device)
        neg_mask = (euc_dis < d).type(torch.float32).to(self.device)
        
        # Compute positive and negative losses
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        
        # Compute final loss as weighted sum
        loss = w * pos_loss.mean() + neg_loss.mean()
        
        return loss, delta
