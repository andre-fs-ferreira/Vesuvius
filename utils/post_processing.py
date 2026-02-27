import os
import numpy as np
import tifffile
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.measure import label
from torch.nn.functional import sigmoid
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
import numpy as np
import torch
import torch.nn.functional as F
import cc3d
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

class VesuviusPostProcessing():
    def __init__(self, config):
        self.config = config
    
    def crop_prob(self, prob, margin=5):
        """Crop probability map by margin."""
        prob[:margin, :, :] = 0
        prob[-margin:, :, :] = 0
        prob[:, :margin, :] = 0
        prob[:, -margin:, :] = 0
        prob[:, :, :margin] = 0
        prob[:, :, -margin:] = 0
        return prob
    
    def split_by_object_np(self, binary_mask):
        # Ensure the volume is treated as a binary mask for labeling
        # (This assumes all objects you want to separate are non-zero)
        binary_mask = binary_mask > 0

        # 2. Perform Connected Component Labeling
        # connectivity=None defaults to allowing diagonal connections. 
        # Use connectivity=1 for strict face-to-face connections only.
        labeled_volume = label(binary_mask, connectivity=None)

        # 3. Optimize Data Type to save space
        # We check how many objects were found to choose the smallest file size possible.
        num_features = np.max(labeled_volume)
        if num_features < 255:
            final_volume = labeled_volume.astype(np.uint8)
        elif num_features < 65535:
            final_volume = labeled_volume.astype(np.uint16)
        else:
            final_volume = labeled_volume.astype(np.uint32)
        return final_volume

    def clean_and_split_seeds(self, seeds, min_size=1000):
        """Label components and remove those smaller than min_size."""
        labeled, _ = ndi.label(seeds)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  
        keep_labels = np.where(sizes >= min_size)[0]
        seeds_clean = np.isin(labeled, keep_labels)
        
        # Assuming split_by_object_np is available in your environment
        return self.split_by_object_np(seeds_clean)

    def pad_volume(self, volume, pad_size=32):
        if isinstance(pad_size, int):
            pad_size = (pad_size, pad_size, pad_size)
        pad_width = tuple((p, p) for p in pad_size)
        padded_volume = np.pad(volume, pad_width=pad_width, mode='constant', constant_values=0)
        return padded_volume, pad_width

    def unpad_volume(self, padded_volume, pad_width):
        slices = tuple(slice(pw[0], -pw[1] if pw[1] > 0 else None) for pw in pad_width)
        return padded_volume[slices]

    def solidify_sheets_by_label_force(self, labels, connectivity=3, iterations=1):
        """Fill holes and seal thin tunnels via closing (dilation then erosion)."""
        labels, pad_w = self.pad_volume(labels, pad_size=iterations)
        solid_labels = np.zeros_like(labels, dtype=np.int32)
        structure = ndi.generate_binary_structure(3, connectivity)

        for lbl in np.unique(labels):
            if lbl == 0: continue
            
            mask = labels == lbl
            mask = ndi.binary_fill_holes(mask, structure=structure)
            
            dilated = ndi.binary_dilation(mask, structure=structure, iterations=iterations)
            closed = ndi.binary_erosion(dilated, structure=structure, iterations=iterations) 
            solid_labels[closed] = lbl
            
        return self.unpad_volume(solid_labels, pad_w)
    
    def instance_split_voi(self, probs, base_labels, max_iters=25):
        """
        Multi-thresholding to separate "stuck" layers.
        Iteratively increases thresholds for large components to split mergers .
        Maximizes the VOI (Variation of Information) score.
        """
        final_instances = np.zeros_like(base_labels)
        current_max_label = 0
        
        stats = cc3d.statistics(base_labels)
        voxel_counts = stats['voxel_counts']
        
        # Identify components that are likely "mergers" (unusually large)
        mean_size = np.mean(voxel_counts[1:]) if len(voxel_counts) > 1 else 0
        
        for label_id in range(1, len(voxel_counts)):
            component_mask = (base_labels == label_id)
            
            # If component is suspiciously large, attempt a split
            if voxel_counts[label_id] > mean_size * 2:
                print(f"Found bridges")
                component_probs = probs * component_mask
                best_split_labels = component_mask
                
                # Search for a higher threshold that causes a split
                for step in range(1, max_iters + 1):
                    thresh = self.config['post_process_hysteresis_low_th'] + (step * 0.01)
                    sub_binary = (component_probs > thresh).astype(np.uint8)
                    sub_labels = cc3d.connected_components(sub_binary, connectivity=26)
                    
                    if np.max(sub_labels) > 1: # A split occurred
                        best_split_labels = sub_labels
                        break
                
                # Map back to global labels
                split_mask = (best_split_labels > 0)
                final_instances[split_mask] = best_split_labels[split_mask] + current_max_label
                current_max_label = np.max(final_instances)
            else:
                final_instances[component_mask] = current_max_label + 1
                current_max_label += 1
                
        return final_instances


    def _run_mine(self, prob_pred):
    
        # 2. Probability Prep
        prob_pred = np.transpose(prob_pred.squeeze().cpu().numpy(), (2, 1, 0))
        prob_pred = self.crop_prob(prob_pred, margin=self.config['post_process_crop_margin'])
        

        # 3. Refined Logic (Threshold -> Clean -> Solidify -> Clean)
        if self.config['simple_TH']:
            grown_seeds = prob_pred > self.config['post_process_hysteresis_high_th']
        elif self.config['hysteresis_TH_propragation']:
            grown_seeds = prob_pred > self.config['post_process_hysteresis_low_th']
            strong_seeds = prob_pred > self.config['post_process_hysteresis_high_th']
            grown_seeds = ndi.binary_propagation(strong_seeds, mask=grown_seeds, structure=ndi.generate_binary_structure(3, 3))
        elif self.config['hysteresis_TH_threshold']:
            grown_seeds = apply_hysteresis_threshold(prob_pred, self.config['post_process_hysteresis_low_th'], self.config['post_process_hysteresis_high_th'])
        else:
            raise ValueError("Invalid post-processing thresholding method specified in config. Only available simple_TH and hysteresis_TH.")
        
        grown_seeds = self.clean_and_split_seeds(grown_seeds, min_size=self.config['post_process_min_size'])
        
        if self.config['post_process_instance_split']:
            grown_seeds = self.instance_split_voi(
                prob_pred, 
                grown_seeds
            )
        
        grown_seeds = self.solidify_sheets_by_label_force(grown_seeds, connectivity=self.config['post_process_connectivity'], iterations=self.config['post_process_solidify_iterations'])
        grown_seeds = self.clean_and_split_seeds(grown_seeds, min_size=self.config['post_process_min_size'])

        grown_seeds = grown_seeds>0.5
        return grown_seeds.astype(np.uint16)
    
    def build_anisotropic_struct(self,z_radius: int, xy_radius: int):
        z, r = z_radius, xy_radius
        if z == 0 and r == 0:
            return None
        if z == 0 and r > 0:
            size = 2 * r + 1
            struct = np.zeros((1, size, size), dtype=bool)
            cy, cx = r, r
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[0, cy + dy, cx + dx] = True
            return struct
        if z > 0 and r == 0:
            struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
            struct[:, 0, 0] = True
            return struct
        depth = 2 * z + 1
        size = 2 * r + 1
        struct = np.zeros((depth, size, size), dtype=bool)
        cz, cy, cx = z, r, r
        for dz in range(-z, z + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[cz + dz, cy + dy, cx + dx] = True
        return struct

    def topo_postprocess(
        self,
        probs,
        T_low=0.90,
        T_high=0.90,
        z_radius=3,
        xy_radius=1,
        dust_min_size=100,
        ):
        # Step 1: 3D Hysteresis
        strong = probs >= T_high
        weak   = probs >= T_low

        if not strong.any():
            return np.zeros_like(probs, dtype=np.uint8)

        struct_hyst = ndi.generate_binary_structure(3, 3)
        mask = ndi.binary_propagation(
            strong, mask=weak, structure=struct_hyst
        )

        if not mask.any():
            return np.zeros_like(probs, dtype=np.uint8)

        # Step 2: 3D Anisotropic Closing
        if z_radius > 0 or xy_radius > 0:
            struct_close = self.build_anisotropic_struct(z_radius, xy_radius)
            if struct_close is not None:
                mask = ndi.binary_closing(mask, structure=struct_close)

        # Step 3: Dust Removal
        if dust_min_size > 0:
            mask = remove_small_objects(
                mask.astype(bool), min_size=dust_min_size
            )

        return mask.astype(np.uint8)

    def _run_kaggle_postprocess(self, prob_pred):
        # It seems that it is expected the output of the model to be in (Z, Y, X)
        grown_seeds = self.topo_postprocess(
            probs=prob_pred,
            T_low=self.config['post_process_hysteresis_low_th'],
            T_high=self.config['post_process_hysteresis_high_th'],
            z_radius=self.config['post_process_kaggle_z_radius'],
            xy_radius=self.config['post_process_kaggle_xy_radius'],
            dust_min_size=self.config['post_process_min_size'],
        )
        # Converting back to the original order
        grown_seeds = np.transpose(grown_seeds.squeeze().cpu().numpy(), (2, 1, 0))
        return grown_seeds.astype(np.uint16)
    
    def topology_repair(self, binary_mask, min_size=200):
        """
        Dusting (Noise Removal) and Morphological Closing (Hole Filling).
        Directly addresses Betti-0 and Betti-1 errors in TopoScore.[2, 3]
        """
        # 1. Remove small noise (Betti-0 fix)
        cleaned = cc3d.dust(binary_mask, threshold=min_size, connectivity=26)
        
        # 2. Fill small internal holes (Betti-1/Betti-2 fix)
        # Using 26-connectivity structure is essential 
        struct = ndi.generate_binary_structure(3, 3) 
        repaired = ndi.binary_closing(cleaned, structure=struct, iterations=1)
        
        return repaired

    def apply_surfaceness_filter(self, probs, sigma=1.0, confidence_thresh=0.70):
        """
        Refined: Masking low-confidence regions before Hessian analysis .
        """
        # 1. Zero out low-confidence voxels to prevent noise amplification
        masked_probs = np.where(probs > confidence_thresh, probs, 0)
        
        # 2. Compute 3D Hessian eigenvalues
        H = hessian_matrix(masked_probs, sigma=sigma, order='rc')
        eigs = hessian_matrix_eigvals(H)
        l1, l2, l3 = eigs, eigs[1], eigs[2] # Corrected indexing order
        
        # 3. Plate-like enhancement
        Ra = np.abs(l2) / (np.abs(l3) + 1e-5)
        S = np.sqrt(l1**2 + l2**2 + l3**2)
        
        beta, gamma = 0.5, 15.0
        enhanced = (1 - np.exp(-(Ra**2) / (2 * beta**2))) * (1 - np.exp(-S**2 / (2 * gamma**2)))
        
        return enhanced * (l3 < 0)

    def _run_curated_post_process(self, prob_pred, threshold):
        """
        The full sequence recommended for the Vesuvius Surface Detection competition.
        """
        prob_pred = np.transpose(prob_pred.squeeze().cpu().numpy(), (2, 1, 0))

        print("Step 1: Enhancing Surfaceness...")
        enhanced_probs = self.apply_surfaceness_filter(prob_pred)
        
        print(f"Step 2: Binarizing at threshold {threshold}...")
        binary = (enhanced_probs > threshold).astype(np.uint8)
        
        print("Step 3: Repairing Topology (Dusting & Closing)...")
        repaired_binary = self.topology_repair(binary)
        
        print("Step 4: Initial Connected Component Analysis...")
        base_labels = cc3d.connected_components(repaired_binary, connectivity=26)
        
        print("Step 5: Refining Instances (VOI Splitting)...")
        final_labels = self.instance_split_voi(prob_pred, base_labels)
        
        return final_labels