from os import listdir, makedirs
from os.path import join
import numpy as np

from torch.utils.data import Dataset
import zarr
import random

class ZarrVolumeDataset(Dataset):
    def __init__(self, zarr_path, transform_input, transform_deform, patch_size=(128, 128, 128), threshold=10.0): # threshold 10 to avoid some background noise
        """
        it expects to receive:
            zarr_path -> path to the root zarr folder
            transform_input -> MONAI transforms to load the data
            transform_deform -> MONAI transforms for self-supervised training
            patch_size -> Patch size (double check if a pre-trained network is being used)
            threshold -> Only returns the data if any voxel inside of the patch is greater than threshold.

        """
        self.zarr_path = zarr_path
        self.transform_input = transform_input
        self.transform_deform = transform_deform
        self.patch_size = patch_size
        self.threshold = threshold  # Value below which we consider the pixel "background"

        print(f"Loading data from -> {zarr_path}")

        self.zarr_vols_paths = []
        for zarr_folder in listdir(zarr_path):
            if zarr_folder.endswith(".zarr"):
                complete_zarr_path = join(zarr_path, zarr_folder)
                self.zarr_vols_paths.append(complete_zarr_path) # save a list of paths
                
                # Load the zarr file in the __getitem___
                # shape = vol.shape
                # entry = {
                #     "name": zarr_folder,
                #     "volume": vol,
                #     "shape": shape
                # }
                # self.zarr_vols.append(entry)
        print(f"All ZARR paths: {self.zarr_vols_paths}")

    def __len__(self):
        # Defining a length of one epoch
        return 1000 

    def __getitem__(self, index):
        # Transformations on the fly
        # Using lazy loader (memory doesn't handle such big data)
        random_entry = random.choice(self.zarr_vols_paths) # select random path
        # open the file (lazy)
        root = zarr.open(random_entry, mode='r')     
        if 'volume' in root:
            vol = root['volume']
        else:
            vol = root['0']
        
        shape = vol.shape
        
        z_max = max(0, shape[0] - self.patch_size[0])
        y_max = max(0, shape[1] - self.patch_size[1])
        x_max = max(0, shape[2] - self.patch_size[2])

        # --- THE REJECTION SAMPLING LOOP ---
        # Try up to 100 times to find a non-empty chunk (very likely to find one!)
        for attempt in range(100):
            # We'll do it ourselves, it's easier to understand
            # 1. Random Coordinates
            z_start = np.random.randint(0, z_max) if z_max > 0 else 0
            y_start = np.random.randint(0, y_max) if y_max > 0 else 0
            x_start = np.random.randint(0, x_max) if x_max > 0 else 0

            # 2. Load the chunk
            patch = vol[
                z_start : z_start + self.patch_size[0],
                y_start : y_start + self.patch_size[1],
                x_start : x_start + self.patch_size[2]
            ]

            # 3. Check if it contains data
            # If the max value in this patch is greater than our threshold (0), it's valid.
            if np.max(patch) > self.threshold:
                # Found valid data! Break the loop and process it.
                break
            
            # If we are here, the patch was empty. The loop continues to the next attempt.
        
        # Note: If the loop finishes 20 times and finds nothing, it will return the LAST empty patch.
        # This prevents the code from hanging forever if the file is truly empty.

        # 4. MONAI Formatting
        patch = patch.astype(np.float32) # Ensure float for transforms
        patch = patch[np.newaxis, ...]   # Add Channel dim -> (1, Z, Y, X)
        
        tracking_mask = np.ones_like(patch) # Create a volume full one 1s to track the mask generated

        # Normalization
        patch_dict = self.transform_input(
            {"image": patch}
        )

        # Save the clean image
        clean_patch = patch_dict['image'].clone()
        
        # Create masked volume
        deform_patch = self.transform_deform(
            {
                "image": patch_dict['image'], 
                "tracking_mask": tracking_mask
            }
        )

        dropout_mask = 1 - deform_patch['tracking_mask']

        return {
            'clean_patch':clean_patch,
            'deform_patch':deform_patch["image"],
            'dropout_mask': dropout_mask
        }
