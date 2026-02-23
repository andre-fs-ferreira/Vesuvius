import unittest
import torch
import numpy as np
import nibabel as nib
import tempfile
import shutil
import os
import json
import sys

# Add the parent directory to path so we can import your main code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from main_train_class import main_train_STU_Net

class TestDataLoaderContent(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary file structure that mimics your real dataset.
        """
        self.test_dir = tempfile.mkdtemp()
        
        self.vol_path = os.path.join(self.test_dir, "vols")
        self.lbl_path = os.path.join(self.test_dir, "lbls")
        self.map_path = os.path.join(self.test_dir, "maps")
        os.makedirs(self.vol_path)
        os.makedirs(self.lbl_path)
        os.makedirs(self.map_path)

        # --- GENERATE DUMMY DATA ---
        # Shape: 100x100x100 (Larger than patch 64x64x64)
        dummy_shape = (100, 100, 100)
        img_data = np.random.rand(*dummy_shape).astype(np.float32)
        
        # Create labels:
        # 0: Background
        # 1: Foreground (Vessel/Target)
        # 2: Ignore (Bridge/Artifact)
        lbl_data = np.zeros(dummy_shape, dtype=np.uint8)
        
        # IMPORTANT: Force a specific region to be Foreground (1)
        # We put a "cube" of 1s in the center to ensure crop finds it.
        lbl_data[40:60, 40:60, 40:60] = 1
        
        # Add some Ignore regions (2)
        lbl_data[0:10, 0:10, 0:10] = 2

        self.filename = "dummy_case_001.nii.gz"
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img_data, affine), os.path.join(self.vol_path, self.filename))
        nib.save(nib.Nifti1Image(lbl_data, affine), os.path.join(self.lbl_path, self.filename))
        nib.save(nib.Nifti1Image(lbl_data, affine), os.path.join(self.map_path, self.filename)) 

        # Create Dummy Split JSON
        self.json_path = os.path.join(self.test_dir, "dataset.json")
        split_content = {
            # We use the same file for both train and val to ensure both have data
            "train": [self.filename,self.filename,self.filename],
            "val": [self.filename,self.filename,self.filename] 
        }
        with open(self.json_path, "w") as f:
            json.dump(split_content, f)

        # Define the Config
        self.target_patch_size = [64, 64, 64]
        self.config = {
            'data_split_json': self.json_path,
            'vol_data_path': self.vol_path,
            'label_data_path': self.lbl_path,
            'bridge_weight_map_path': self.map_path,
            'patch_size': self.target_patch_size,
            'batch_size': 2, # Training batch size
            'num_workers': 1,      
            'train_cache_rate': 0.0,
            'val_cache_rate': 0.0, 
            'debug': False,
            'data_augmentation': "All",
            'deep_supervision': False,
            'all_data': False,
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_loader_content_validity(self):
        """
        Verify TRAINING loader shapes and content.
        """
        print("\nTesting TRAINING DataLoader content...")

        trainer = main_train_STU_Net.__new__(main_train_STU_Net)
        trainer.config = self.config
        train_loader = trainer._set_train_dataloader()

        batch = next(iter(train_loader))
        roi_mask = batch['roi_mask']
        gt = batch['gt']

        # Check Shape
        img_spatial_shape = list(roi_mask.shape[2:]) 
        self.assertEqual(img_spatial_shape, self.target_patch_size, f"❌ Train Shape Mismatch")

        # Check ROI not empty
        self.assertTrue(torch.sum(roi_mask) > 0, "Error: Train 'roi_mask' is empty!")

        # Check GT contains Foreground
        self.assertTrue(torch.sum(gt == 1) > 0, "Error: Train 'gt' missing foreground!")

        print(f"✅ Success: Train Loader Shape {img_spatial_shape} | ROI Active | GT has target.")

    def test_val_loader_content_validity(self):
        """
        Verify VALIDATION loader shapes and content.
        """
        print("\nTesting VALIDATION DataLoader content...")

        trainer = main_train_STU_Net.__new__(main_train_STU_Net)
        trainer.config = self.config
        
        # This calls the validation specific method
        val_loader = trainer._set_val_dataloader()

        batch = next(iter(val_loader))
        roi_mask = batch['roi_mask']
        gt = batch['gt']

        # --- ASSERTION 1: Check Batch Size ---
        # Your code hardcodes batch_size=1 for validation
        self.assertEqual(roi_mask.shape[0], 1, 
            f"❌ Val Batch Size Error: Expected 1, got {roi_mask.shape[0]}")

        # --- ASSERTION 2: Check Spatial Shape ---
        img_spatial_shape = list(roi_mask.shape[2:]) 
        self.assertEqual(img_spatial_shape, self.target_patch_size, 
            f"❌ Val Shape Mismatch: Expected {self.target_patch_size}, got {img_spatial_shape}")

        # --- ASSERTION 3: Check ROI Mask ---
        self.assertTrue(torch.sum(roi_mask) > 0, "Error: Val 'roi_mask' is empty!")

        # --- ASSERTION 4: Check GT ---
        # Note: Since your validation transforms also use RandCropByPosNegLabeld with pos=1,
        # we expect the validation patch to focus on the target as well.
        self.assertTrue(torch.sum(gt == 1) > 0, "Error: Val 'gt' missing foreground!")

        print(f"✅ Success: Val Loader Shape {img_spatial_shape} | Batch Size 1 | GT has target.")

if __name__ == '__main__':
    unittest.main()