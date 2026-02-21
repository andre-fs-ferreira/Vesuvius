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
from post_processing import VesuviusPostProcessing
import tifffile  

import os
import glob
import pandas as pd
from tqdm import tqdm  # 🚀 Import tqdm
import itertools
from torch.amp import autocast

class VesuviusInferer(BaseInfer):
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.criterion = self._set_evaluation_criterion()
        self.val_transforms = self._set_val_transforms()
        self.test_transforms = self._set_test_transforms()
        self.sliding_window = self._set_sliding_window_inferer()

        self.axes_combinations = self._define_flip_tta(spatial_dims=(3, 4))

        self.post_processor = VesuviusPostProcessing(self.config)
        

    def _build_model(self):
        """Initialize the neural network architecture. Load pretrained weights if necessary."""
        print(f"Loading weights from: {self.config['checkpoint_path']}")
        
        # Initialize the new Segmentation Model (1 output channels)
        model = STUNetSegmentation(self.config['deep_supervision'], activation=self.config['activation'])
        
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
    
    def save_tiff(self, torch_tensor, save_path, transpose=True, **kwargs):
        """Logic for saving predictions to disk."""
        # Ensure tensor is 3D (H, W, D) or (H, W)
        try:
            data = torch_tensor.detach().cpu().numpy().astype('uint8')
        except:
            data = torch_tensor.astype('uint8')
        while data.ndim > 3:
            data = np.squeeze(data, axis=0)
        if transpose:
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

    def _define_flip_tta(self, spatial_dims=(2, 3, 4)):
        axes_combinations = []
        for i in range(len(spatial_dims) + 1):
            axes_combinations.extend(itertools.combinations(spatial_dims, i))
        return axes_combinations
            
    def infer(self, input, test=False, threshold_list=[], **kwargs):
        """Logic for inference. Returns predictions."""
        if test:
            data = self.load_input_data(file_dict=input, transforms=self.test_transforms)
        else:
            data = self.load_input_data(file_dict=input, transforms=self.val_transforms)

        input_image = data['image'].unsqueeze(0).to(self.config['device'])

        self.model.eval()
        
        # TTA
        if self.config["TTA"]:
            print(f"Doing inference to these axes combinations:")
            print(self.axes_combinations)
            all_predictions = []
            all_logits_pred = []
            for axes in self.axes_combinations:
                print(f"Doing axes: {axes}")
                if axes:
                    # torch.flip requires a tuple of dims
                    aug_input = torch.flip(input_image, dims=axes)
                else:
                    aug_input = input_image  # Original image

                with torch.no_grad():
                        if self.config['activation']:
                            prediction = self.sliding_window(inputs=aug_input, network=self.model)
                        else: # in case sigmoid is not applied in the end of the network (recommended to be True!)
                            logits_pred = self.sliding_window(inputs=aug_input, network=self.model)
                            prediction = sigmoid(logits_pred)
                    
                # revert flip
                if axes:
                    pred_prob_aligned = torch.flip(prediction, dims=axes)
                    if not self.config['activation']:
                        logits_pred = torch.flip(logits_pred, dims=axes)
                else:
                    pred_prob_aligned = prediction

                all_predictions.append(pred_prob_aligned)
                if not self.config['activation']:
                    all_logits_pred.append(logits_pred)

            final_pred = torch.stack(all_predictions).mean(dim=0)
            if not self.config['activation']:
                all_logits_pred = torch.stack(all_logits_pred).mean(dim=0)
                
        else:
            if self.config['activation']:
                
                    with torch.no_grad():
                        final_pred = self.sliding_window(inputs=input_image, network=self.model)
            else:
                with torch.no_grad():
                    
                        logits_pred = self.sliding_window(inputs=input_image, network=self.model)
                        final_pred = sigmoid(logits_pred)
                        all_logits_pred = logits_pred
        
        all_preds = []
        for threshold in threshold_list:
            # Binary seg
            final_pred_copy = final_pred.clone()
            final_pred_copy[final_pred_copy>threshold] = 1.0
            final_pred_copy[final_pred_copy<=threshold] = 0.0
            all_preds.append(final_pred_copy)

        if test:
            if self.config['activation']:
                return final_pred, all_preds
            else:
                return all_logits_pred, all_preds
        else:
            if self.config['activation']:
                return all_preds, data['gt'].unsqueeze(0).to(self.config['device']), data['roi_mask'].unsqueeze(0).to(self.config['device']) 
            else:
                return all_logits_pred, all_preds, data['gt'].unsqueeze(0).to(self.config['device']), data['roi_mask'].unsqueeze(0).to(self.config['device']) 

    def create_dfs(self, path_dir):
        # Generate DataFrame
        result_df = glob.glob(os.path.join(path_dir, '**/*.tif'), recursive=True)
        result_df = pd.DataFrame({'tif_paths': result_df})
        result_df['id'] = result_df['tif_paths'].apply(lambda x: os.path.basename(x).split('.')[0])

        # save dataframe to csv (take name from the path_dir last folder)
        csv_name = os.path.basename(os.path.normpath(path_dir)) + '_df.csv'
        print(f"path_dir: {path_dir}")
        print(f"csv_name: {csv_name}")
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
    
    def dataset_inference_save_logits(self, dataset_path, pred_save_dir):
        """ inference on all cases in a dataset directory and save predictions """
        os.makedirs(pred_save_dir, exist_ok=True)
        tif_files = glob.glob(os.path.join(dataset_path, '**/*.tif'), recursive=True)
        nii_files = glob.glob(os.path.join(dataset_path, '**/*.nii.gz'), recursive=True)

        all_cases = tif_files + nii_files

        print(f"Found cases: {all_cases}")
        
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
            
            logits_pred, pred = self.infer(input_data, test=True)
            
            # 💾 Robust filename extraction
            file_name = os.path.basename(case).replace('.tif', '')
            file_name = os.path.basename(case).replace('.nii.gz', '')
            save_path = os.path.join(pred_save_dir, f"{file_name}.nii.gz")
            
            self.save_nifti(
                torch_tensor=logits_pred, 
                save_path=save_path, 
                real_file=str(case)
            )

        print(f"✅ Inference completed. Predictions saved to: {pred_save_dir}")

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

            # 💾 Robust filename extraction
            file_name = os.path.basename(case).replace('.tif', '')

            
            # Create a list of all potential file paths
            paths_to_check = [
                os.path.join(str(pred_save_dir), f"th_{str(th)}", f"{file_name}.tif") 
                for th in self.config["TH_list"]
            ]
      
            # Check if ANY of those paths do NOT exist
            if any(not os.path.isfile(p) for p in paths_to_check):
                # Your code to generate/save predictions goes here
                
                if self.config.get("post_process", False):
                    if self.config['simple_TH']:
                        add_name = "simple_TH"
                    elif self.config['hysteresis_TH_propragation']:
                        add_name = "hysteresis_TH_propragation"
                    elif self.config['hysteresis_TH_threshold']:
                        add_name = "hysteresis_TH_threshold"
                    else:
                        add_name = "no_TH"
                    dir_path = os.path.join(str(pred_save_dir), f"post_process_{add_name}_{self.config['post_process_hysteresis_low_th']}_{self.config['post_process_hysteresis_high_th']}_iters{self.config['post_process_solidify_iterations']}")
                    os.makedirs(dir_path, exist_ok=True)
                    save_path = os.path.join(dir_path, f"{file_name}.tif")
                    
                elif self.config.get("post_process_kaggle", False):
                    dir_path = os.path.join(str(pred_save_dir), f"post_process_kaggle_{self.config['post_process_hysteresis_low_th']}_{self.config['post_process_hysteresis_high_th']}")
                    os.makedirs(dir_path, exist_ok=True)
                    save_path = os.path.join(dir_path, f"{file_name}.tif")
                
                print(f"Creating {save_path}")
                if os.path.isfile(save_path):
                    print(f"Case already done: {save_path}")
                    continue
                    
                logits_pred, all_preds = self.infer(input_data, test=True, threshold_list=self.config["TH_list"])

                if self.config.get("post_process", False):
                    # Post processing!
                    pred = self.post_processor._run_mine(logits_pred)
                    self.save_tiff(
                            torch_tensor=pred, 
                            save_path=save_path,
                            transpose=False
                        )
                elif self.config.get("post_process_kaggle", False):
                    # Post processing!
                    pred = self.post_processor._run_kaggle_postprocess(logits_pred)
                    self.save_tiff(
                            torch_tensor=pred, 
                            save_path=save_path,
                            transpose=False
                        )
                else:
                    for th_idx, pred in enumerate(all_preds):
                        os.makedirs(os.path.join(str(pred_save_dir), f"th_{str(self.config['TH_list'][th_idx])}"), exist_ok=True)
                        
                        
                        self.save_tiff(
                            torch_tensor=pred, 
                            save_path=os.path.join(str(pred_save_dir), f"th_{str(self.config['TH_list'][th_idx])}", f"{file_name}.tif")
                        )
            else:
                print(f"Case already predicted: {paths_to_check}")
            

        # 📊 Creating the dataframe for testing
        print("\n📝 Generating submission dataframes...")
        if self.config.get("post_process", False) or self.config.get("post_process_kaggle", False):
            print(f"Creating data frame in {dir_path}")
            self.create_dfs(dir_path)
        else:
            for th in self.config["TH_list"]:
                dir_path = os.path.join(str(pred_save_dir), f"th_{str(th)}")
                print(f"Creating data frame in {dir_path}")
                self.create_dfs(dir_path)
        print(f"✅ Inference completed. Predictions saved to: {pred_save_dir}")
        return dir_path