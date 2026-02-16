import os
import numpy as np
import tifffile
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.measure import label
from torch.nn.functional import sigmoid
from skimage.filters import apply_hysteresis_threshold

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
            mask = self.remove_small_objects(
                mask.astype(bool), min_size=dust_min_size
            )

        return mask.astype(np.uint8)

    def _run_kaggle_postprocess(self, prob_pred):
        prob_pred = np.transpose(prob_pred.squeeze().cpu().numpy(), (2, 1, 0))
        grown_seeds = self.topo_postprocess(
            probs=prob_pred,
            T_low=self.config['post_process_hysteresis_low_th'],
            T_high=self.config['post_process_hysteresis_high_th'],
            z_radius=self.config['post_process_kaggle_z_radius'],
            xy_radius=self.config['post_process_kaggle_xy_radius'],
            dust_min_size=self.config['post_process_min_size'],
        )
        return grown_seeds.astype(np.uint16)