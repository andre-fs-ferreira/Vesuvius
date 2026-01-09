from torch import abs
import torch.nn as nn

class CriterionL1(nn.Module):
    def __init__(self, mask_weight=10.0):
        super().__init__()
        """
        Computes L1 loss.
        mask_weight: How much more we care about the masked region than the global image.
                     Default 10.0 means masked region is ~10x more important.
        """
        self.l1_loss = nn.L1Loss()
        self.mask_weight = mask_weight
    
    def forward(self, pred, target, dropout_mask):
        # 1. Calculate absolute difference
        l1_diff = abs(pred - target)
        
        # 2. Masked Loss (The "Hard" Task)
        # Apply mask (1 = hole/missing, 0 = visible)
        masked_l1 = l1_diff * dropout_mask
        
        # Normalize by the number of masked pixels
        # (Sum of L1 errors in mask / Count of masked pixels)
        loss_masked = masked_l1.sum() / (dropout_mask.sum() + 1e-8)

        # 3. Global Loss (The "Stabilizer")
        # Calculates mean over the *entire* volume (masked + visible)
        loss_global = self.l1_loss(pred, target)
        
        # 4. Combine
        # Effectively: Loss = (1.0 * Masked) + (0.1 * Global)
        total_loss = loss_masked + (loss_global / self.mask_weight)
        
        # No need to divide by 2 unless you have a specific learning rate reason
        return total_loss