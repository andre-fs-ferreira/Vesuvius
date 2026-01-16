from abs_infer import BaseInfer
from stunet_model import STUNetSegmentation

import torch
import nibabel as nib
import numpy as np
from torch.nn.functional import sigmoid

import monai
from monai.transforms.transform import MapTransform, Transform
from monai.data import CacheDataset
from monai.metrics import DiceHelper
from monai.losses import DiceLoss, TverskyLoss, FocalLoss
from monai.transforms import (
    Compose,
    CopyItemsd,
    LoadImaged, 
    ScaleIntensityRanged, 
    ResizeWithPadOrCropd, 
    RandCoarseDropoutd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld
)
from monai.inferers import SlidingWindowInferer

from mask_utils import GetROIMaskdd, GetBinaryLabeld
import tifffile  

import os
import glob
import pandas as pd
from tqdm import tqdm  # 🚀 Import tqdm

class VesuviusInferer(BaseInfer):
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.criterion = self._set_evaluation_criterion()
        self.val_transforms = self._set_val_transforms()
        self.test_transforms = self._set_test_transforms()
        self.sliding_window = self._set_sliding_window_inferer()

    def _build_model(self):
        """Initialize the neural network architecture. Load pretrained weights if necessary."""
        print(f"Loading weights from: {self.config['checkpoint_path']}")
        
        # Initialize the new Segmentation Model (1 output channels)
        model = STUNetSegmentation()
        
        # Load trained Weights (1 output channel)
        checkpoint_state_dict = torch.load(self.config['checkpoint_path'], map_location='cpu')
        checkpoint_state_dict = checkpoint_state_dict['model_weights']
        load_result = model.load_state_dict(checkpoint_state_dict, strict=True)
        
        # Move to GPU
        model = model.to(self.config['device'])
        return model

    def _set_evaluation_criterion(self):
        """Define the evaluation function."""
        val_metric = DiceHelper
        
        def compute_masked_val(pred, target, roi_mask):
            pred_masked = pred * roi_mask
            pred_masked = (pred_masked > 0.5).float()
            # check if empty
            if torch.sum(pred_masked) == 0:
                # If both pred and target are empty in the ROI, return Dice of 1.0
                if torch.sum(target * roi_mask) == 0:
                    return torch.tensor(1.0, device=target.device, dtype=torch.float32)
                else:
                    return torch.tensor(0.0, device=target.device, dtype=torch.float32)
            target_masked = target * roi_mask
            results = val_metric(include_background=True, sigmoid=False, softmax=False)(pred_masked, target_masked)
            return results
        return compute_masked_val
        
    def save_nifti(self, torch_tensor, save_path, real_file, **kwargs):
        """Logic for saving predictions to disk."""
        # Ensure tensor is 3D (H, W, D) or (H, W)
        data = torch_tensor.detach().cpu().numpy().astype('float32')
        while data.ndim > 3:
            data = np.squeeze(data, axis=0)

        if real_file is not None:
            # Load the real file to get metadata
            real_nii = nib.load(real_file)
            real_affine = real_nii.affine
            real_header = real_nii.header
            pred_nii = nib.Nifti1Image(data, affine=real_affine, header=real_header)
        else:
            # If no real file is provided, use identity affine
            affine = np.eye(4)
            pred_nii = nib.Nifti1Image(data, affine=affine)

        # Save the Nifti1Image to disk
        nib.save(pred_nii, save_path)
    
    def save_tiff(self, torch_tensor, save_path, **kwargs):
        """Logic for saving predictions to disk."""
        # Ensure tensor is 3D (H, W, D) or (H, W)
        data = torch_tensor.detach().cpu().numpy().astype('uint8')
        while data.ndim > 3:
            data = np.squeeze(data, axis=0)
        data = np.transpose(data, (2, 1, 0))
        
        # If the data is a probability map (0-1), 
        # many Vesuvius tools expect uint16 or uint8
        # Convert if necessary: (data * 255).astype('uint8')
        tifffile.imwrite(save_path, data)

    def _set_dataloader_transforms(self, gt_available, **kwargs):
        """Define the data loading and augmentation transforms."""

        if gt_available:
            general_keys = ["image", 'gt']
            all_keys = ["image", 'gt', 'roi_mask']
        else:
            general_keys = ["image"]
            all_keys = ["image"]

        transform_list = [
                LoadImaged(keys=general_keys),
                EnsureChannelFirstd(keys=general_keys),
                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
            ]
        
        if gt_available:
            transform_list.append(GetROIMaskdd(keys=["gt"], ignore_mask_value=2, new_key_names=["roi_mask"]))
            transform_list.append(GetBinaryLabeld(keys=["gt"], ignore_mask_value=2))

        # Get random patches
        transform_list.append(EnsureTyped(keys=all_keys, track_meta=False))

        return transform_list
    
    def _set_val_transforms(self, **kwargs):
        """Define the validation data loading transforms."""
        transform_list = self._set_dataloader_transforms(gt_available=True, **kwargs)
        return Compose(transform_list)

    def _set_test_transforms(self, **kwargs):   
        """Define the test data loading transforms."""
        transform_list = self._set_dataloader_transforms(gt_available=False, **kwargs)
        return Compose(transform_list)

    def load_input_data(self, file_dict, transforms, **kwargs):
        '''
        Monai data loading logic. It must handle tiff files.
        '''
        transformed_data = transforms(file_dict)
        return transformed_data
    
    def _set_sliding_window_inferer(self, **kwargs):
        """Define the sliding window inferer."""
        
        inferer = monai.inferers.SlidingWindowInferer(
            roi_size=self.config['patch_size'],
            sw_batch_size=self.config['infer_sw_batch_size'],
            overlap=self.config['infer_overlap'],
            mode=self.config['blend_mode'],
            padding_mode='constant',
            cval=0.0,
            device=self.config['device'],
            progress=True
        )

        return inferer

    def infer(self, input, test=False, threshold=0.5, **kwargs):
        """Logic for inference. Returns predictions."""

        if test:
            data = self.load_input_data(file_dict=input, transforms=self.test_transforms)
        else:
            data = self.load_input_data(file_dict=input, transforms=self.val_transforms)

        input_image = data['image'].unsqueeze(0).to(self.config['device'])

        self.model.eval()
        with torch.no_grad():
            pred = self.sliding_window(inputs=input_image, network=self.model)
            pred = sigmoid(pred)
            pred[pred>threshold] = 1.0
            pred[pred<=threshold] = 0.0

        if test:
            return pred
        else:
            return pred, data['gt'].unsqueeze(0).to(self.config['device']), data['roi_mask'].unsqueeze(0).to(self.config['device']) 

    def create_dfs(self, path_dir):
        # Generate DataFrame
        result_df = glob.glob(os.path.join(path_dir, '**/*.tif'), recursive=True)
        result_df = pd.DataFrame({'tif_paths': result_df})
        result_df['id'] = result_df['tif_paths'].apply(lambda x: os.path.basename(x).split('.')[0])

        # save dataframe to csv (take name from the path_dir last folder)
        csv_name = os.path.basename(os.path.normpath(path_dir)) + '_df.csv'
        result_df.to_csv(os.path.join(path_dir, csv_name), index=False)
        return result_df

    def evaluate(self, predictions, gt, roi_mask, **kwargs):
        """
        Logic for evaluation. Takes predictions and ground truth as input.
        Returns a dictionary of metrics. 
        Expected to be used after infer.
        """
        results = self.criterion(predictions, gt, roi_mask)
        return results
    
    def dataset_inference(self, dataset_path, pred_save_dir):
        """ inference on all cases in a dataset directory and save predictions """
        os.makedirs(pred_save_dir, exist_ok=True)
        all_cases = glob.glob(os.path.join(dataset_path, '**/*.tif'), recursive=True)
        
        # 🏁 Wrap the list in tqdm for a visual progress bar
        # 'desc' adds a label to the bar, 'unit' labels each iteration
        for case in tqdm(all_cases, desc="🌋 Running Vesuvius Inference", unit="vol"):
            # Optional: you can still print the case, but it might "flicker" the bar. 
            # Using tqdm.write() keeps the bar at the bottom.
            # tqdm.write(f"🔍 Processing: {os.path.basename(case)}")
            
            input_data = {
                'image': str(case),
                'gt': None
            }
            
            pred = self.infer(input_data, test=True)
            
            # 💾 Robust filename extraction
            file_name = os.path.basename(case).replace('.tif', '')
            save_path = os.path.join(pred_save_dir, f"{file_name}.tif")
            
            self.save_tiff(
                torch_tensor=pred, 
                save_path=save_path
            )

        # 📊 Creating the dataframe for testing
        print("\n📝 Generating submission dataframes...")
        self.create_dfs(pred_save_dir)
        print(f"✅ Inference completed. Predictions saved to: {pred_save_dir}")