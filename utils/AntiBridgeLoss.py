import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
from skimage.measure import label

class AntiBridgeLoss(nn.Module):
    def __init__(self, sigma=1.0, w0=10.0):
        super().__init__()
        self.smooth = 1e-5
        self.sigma = sigma
        self.w0 = w0

    def _get_edge_weight_map(self, gt_vol):
        """
        Generates a 3D weight map for one volume.
        gt_vol input: Numpy array (D, H, W)
        Returns: Numpy array (1, 1, D, H, W)
        """
        # Ensure clean copy to avoid side-effects on the original dataset
        gt_vol = np.array(gt_vol)
        gt_vol[gt_vol == 2] = 0 
        
        # Label 3D Instances
        instance_mask = label(gt_vol)
        
        weight_map = np.zeros_like(instance_mask, dtype=np.float32)

        # 3D Structure for Dilation (Connect across Z axis too)
        struct = generate_binary_structure(3, 2) 

        obj_ids = np.unique(instance_mask)
        obj_ids = obj_ids[obj_ids != 0]

        for obj_id in obj_ids:
            obj_mask = (instance_mask == obj_id)

            # 3D Dilation
            dilated = binary_dilation(obj_mask, structure=struct, iterations=2)
            
            # Create Halo (XOR)
            halo = dilated ^ obj_mask 
            
            # 3D Gaussian Blur
            blurred_halo = gaussian_filter(halo.astype(float), sigma=self.sigma)
            
            # Normalize and Add
            if blurred_halo.max() > 0:
                blurred_halo = (blurred_halo / blurred_halo.max()) * self.w0
                
            weight_map += blurred_halo

        # Final Base Weight
        weight_map += 1.0
        
        # Add Batch and Channel dimensions: (D,H,W) -> (1, 1, D, H, W)
        return weight_map[None, None, ...]

    def forward(self, pred, gt, roi_mask, bridge_weight_map=None):
        """
        pred: (B, 1, D, H, W) Logits
        gt:   (B, 1, D, H, W) Binary Target
        roi_mask: (B, 1, D, H, W) 
        bridge_weight_map: (Optional) Pre-calculated map. 
        """
        # Activation (Logits -> Probabilities)
        pred_prob = torch.sigmoid(pred)

        # Handle Weight Map Generation (CPU <-> GPU Bridge)
        if bridge_weight_map is None:
            # WARNING: Generating this on the fly is SLOW. 
            # It moves tensors to CPU, runs Scipy, and moves back.
            
            generated_maps = []
            # Loop over batch (B)
            for i in range(gt.shape[0]):
                # Detach, move to CPU, convert to Numpy, remove Channel dim
                gt_np = gt[i, 0].detach().cpu().numpy()
                
                # Generate Map
                w_map_np = self._get_edge_weight_map(gt_np)
                
                # Convert back to Tensor and move to correct device
                w_map_tensor = torch.from_numpy(w_map_np).to(pred.device)
                generated_maps.append(w_map_tensor)
            
            # Stack batch back together
            bridge_weight_map = torch.cat(generated_maps, dim=0)

        # Compute Weighted Loss
        # reduction='none' is crucial to keep the shape for masking
        pixel_loss = F.binary_cross_entropy(pred_prob, gt, reduction='none')
        
        # Apply Anti-Bridge Weights
        weighted_loss = pixel_loss * bridge_weight_map

        # Apply ROI Mask
        # Zero out loss in ignored regions
        masked_loss = weighted_loss * roi_mask

        # Average over VALID pixels only
        loss_scalar = masked_loss.sum() / (roi_mask.sum() + self.smooth)

        return loss_scalar