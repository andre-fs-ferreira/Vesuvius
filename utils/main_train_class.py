# Standard Library Imports
import os
import datetime
from os import listdir, makedirs
from os.path import join
import sys

import json

# Third-Party Library Imports
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import sigmoid, binary_cross_entropy_with_logits
import nibabel as nib
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

# MONAI Specific Imports
import monai
from monai.data import CacheDataset
from monai.metrics import DiceMetric
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
    RandCropByPosNegLabeld,
    RandFlipd,
    Rand3DElasticd,
    RandRotated,
    RandZoomd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandSimulateLowResolutiond
)

# Local Project Imports
from abs_training import BaseTrainer
from stunet_model import STUNetSegmentation # We want to seg now -> Last layer is has 2 output channels instead of one # STUNetReconstruction

sys.path.append(os.path.abspath("../utils"))

from cldice.cldice import soft_cldice
from AntiBridgeLoss import AntiBridgeLoss
from mask_utils import GetROIMaskdd, GetBinaryLabeld

class main_train_STU_Net(BaseTrainer):
    def __init__(self, config):
        self.config = config

        # build the pipeline
        self.model, self.mising_weights = self._build_model() # DONE
        self.optimizer = self._set_optimizer() # DONE
        self.train_criterion = self._set_train_criterion() # DONE
        self.val_metric = self._set_val_metric() # DONE
        self.scheduler = self._set_scheduler() # DONE
        # TODO uncomment
        # self.wandb_run = self._set_wandb_checkpoint() # DONE
        self.train_loader = self._set_train_dataloader() # DONE
        self.val_loader = self._set_val_dataloader() # DONE

        # set scaler for mix precision 
        self.scaler = GradScaler()

        # check resume
        self._resume()

    def _set_wandb_checkpoint(self):
        """ Define the wandb run """

        x = datetime.datetime.now()

        date_string = f"{x.day}-{x.month}-{x.year}"

        loss_string = ''
        for metric, w in zip(self.config['criterion'], self.config['criterion_weights']):
            loss_string += f'{w}{metric}_'

        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="faking_it",
            # Set the wandb project where this run will be logged.
            project="Vesuvius",
            # Track hyperparameters and run metadata.
            config=self.config,
            name=f"main_{self.config['training_step']}_Loss_{loss_string}_{date_string}", 
            dir=self.config['save_dir']
        )
        self.model_save_path = join(run.dir, "model")
        makedirs(self.model_save_path, exist_ok=True)

        self.preds_path = join(run.dir, "preds")
        makedirs(self.preds_path, exist_ok=True)

        return run 

    def _build_model(self):
        """Initialize the neural network architecture with transfer learning logging."""
        print(f"Loading weights from: {self.config['checkpoint_path']}")
        
        # Initialize the new Segmentation Model (1 output channels)
        model = STUNetSegmentation()
        model_state_dict = model.state_dict()
        
        # Load the Reconstruction Weights (1 output channel)
        checkpoint_state_dict = torch.load(self.config['checkpoint_path'], map_location='cpu')
        checkpoint_state_dict = checkpoint_state_dict['model_weights']
        
        # Filter and Log
        filtered_state_dict = {}
        skipped_layers = []
        
        for k, v in checkpoint_state_dict.items():
            if k in model_state_dict:
                # Check for Shape Mismatch
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    # Found a layer with same name but wrong shape (e.g. seg_outputs)
                    skipped_layers.append(f"{k} (Shape mismatch: Checkpoint {v.shape} vs Model {model_state_dict[k].shape})")
            else:
                # Found a layer in checkpoint that doesn't exist in new model
                skipped_layers.append(f"{k} (Not in new model)")

        # Load the safe weights
        load_result = model.load_state_dict(filtered_state_dict, strict=False)
        
        
        # Print Report
        print("\n" + "="*50)
        print("TRANSFER LEARNING REPORT")
        print("="*50)
        
        if len(skipped_layers) > 0:
            print(f"❌ SKIPPED {len(skipped_layers)} LAYERS (Will be discarded):")
            for s in skipped_layers:
                print(f"   - {s}")
        else:
            print("✅ No layers skipped from checkpoint.")

        print("-" * 50)
        
        # 'missing_keys' are layers in the new model that found no weights (Randomly Initialized)
        # We expect seg_outputs.0, .1, .2, .3, and .4 to be here.
        if len(load_result.missing_keys) > 0:
            print(f"⚠️  MISSING KEYS ({len(load_result.missing_keys)}) (Initialized Randomly):")
            # Only print the first few to avoid spamming console if many
            for k in load_result.missing_keys[:10]: 
                print(f"   - {k}")
            if len(load_result.missing_keys) > 10:
                print(f"   ... and {len(load_result.missing_keys) - 10} more.")
        else:
            print("✅ No missing keys.")
            
        print("="*50 + "\n")

        # Move to GPU
        model = model.to(self.config['device'])

        return model, load_result.missing_keys

    def _set_optimizer(self):
        """Define the optimizer (e.g., Adam, SGD)."""
        return optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

    def _set_train_criterion(self):
        """Define the loss function."""
        available_criterions = {
            'DSC': DiceLoss( # To ignore the value 2, the gt and pred will have set to 0 the region with mask=2.
                include_background=True, 
                to_onehot_y=False, 
                sigmoid=False, # Sigmoid must be applied before masking
                softmax=False, 
                batch=True
            ),
            'BCE': binary_cross_entropy_with_logits,
            'Tversky': TverskyLoss( # To ignore the value 2, the gt and pred will have set to 0 the region with mask=2.
                include_background=True,
                sigmoid=False, # Sigmoid must be applied before masking
                softmax=False, 
                alpha=0.7, # Trying to reduce bridges
                beta=0.3, 
                batch=True
            ),
            'Focal': FocalLoss(
                include_background=True,  
                gamma=2.0, 
                alpha=None, 
                weight=None, 
                use_softmax=False,
                reduction="none"
            ),
            'CLDICE': soft_cldice( # not use in Deep Supervision
                iter_=10, 
                smooth=1e-5, 
                exclude_background=False
            ),
            'AntiBridge': AntiBridgeLoss( # not use in Deep Supervision
                sigma=1.0, # Don't used if the edge map exists
                w0=10.0 # Don't used if the edge map exists
            )
        }
        # Read criterions and weights from config
        criterions = self.config.get('criterion', [])
        weights = self.config.get('criterion_weights', [])

        # Validate lengths
        if len(criterions) != len(weights):
            raise ValueError(
                f"Number of criterions ({len(criterions)}) does not match number of weights ({len(weights)})"
            )

        # Build list of (loss_fn, weight) pairs
        loss_dict = {}
        for crit, w in zip(criterions, weights):
            if crit not in available_criterions:
                raise ValueError(f"Unknown criterion '{crit}'")
            loss_dict[crit] = (available_criterions[crit], w)


        # Return a function that computes weighted sum of losses
        def combined_loss(pred, target, roi_mask, bridge_weight_map, deep_supervision_weights=[0.25,0.5,1]):
            """
            Compute loss as weighted sum of multiple criterions
            
            :param pred: List of outputs from the model (segmentation in 3 for deep supervision)
            :param target: Ground truth segmentation (3 as well for deep supervison)
            :param roi_mask: Mask of the region of interest where the loss will be computed (same shape as pred and target)
            :param bridge_weight_map: Weight map to penalize bridges (same shape as pred and target)
            :return: total_loss, losses_dict
            """
            total_loss = 0.0
            losses_dict = {}

            # iterate over each metric to compute
            for crit in loss_dict.keys():
                loss_fn, w = loss_dict[crit]
                loss_here = 0.0
                if crit=='DSC' or crit=='Tversky':
                    # Make sure each deep supervision output is handled
                    for sub_pred, sub_target, sub_roi_mask, sub_dsw in zip(pred, target, roi_mask, deep_supervision_weights):
                        # apply activation befofre masking to ensure 0 in the region to ignore
                        pred_masked = sigmoid(sub_pred)*sub_roi_mask
                        target_masked = sub_target*sub_roi_mask
                        loss_value = loss_fn(pred_masked, target_masked)*sub_dsw
                        loss_here += loss_value
                    # Save last for quality control (to be able to compare with val loss)
                    # The last one is the full resolution
                    if len(pred) > 1:
                        losses_dict[f"{crit}_fullres"] = loss_value.item() * w
                # In here, the functions use sigmoid internally
                elif crit=='BCE' or crit=='Focal':
                    for sub_pred, sub_target, sub_roi_mask, sub_dsw in zip(pred, target, roi_mask, deep_supervision_weights):
                        if crit=='BCE':
                            loss_raw = loss_fn(sub_pred, sub_target, reduction='none')
                        else:
                            loss_raw = loss_fn(sub_pred, sub_target)
                        loss_masked = loss_raw * sub_roi_mask
                        loss_value = (loss_masked.sum() / (sub_roi_mask.sum() + 1e-8))*sub_dsw
                        loss_here += loss_value

                    # Save last for quality control (to be able to compare with val loss)
                    # The last one is the full resolution
                    if len(pred) > 1:
                        losses_dict[f"{crit}_fullres"] = loss_value.item() * w

                # NO DEEP SUPERVISON FOR THESE LOSSES
                elif crit=='CLDICE':
                    # apply activation befofre masking to ensure 0 in the region to ignore
                    pred_masked = sigmoid(pred[-1]) * roi_mask[-1]
                    target_masked = target[-1] * roi_mask[-1]
                    loss_here = loss_fn(pred_masked, target_masked)
                elif crit=='AntiBridge':
                    loss_here = loss_fn(pred[-1], target[-1], roi_mask[-1], bridge_weight_map)

                total_loss += w * loss_here
                losses_dict[crit] = loss_here.item()
            return total_loss, losses_dict
        return combined_loss 

    def _set_val_metric(self):
        # Initialize the MONAI metric
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

        def compute_masked_val(pred, target, roi_mask):
            # Prepare Prediction (Sigmoid + Threshold)
            pred_prob = torch.sigmoid(pred)
            pred_masked = (pred_prob * roi_mask > 0.5).float()
            
            # Prepare Target (Masking the ignore regions)
            target_masked = target * roi_mask

            # Handle the "All Empty" Case manually before the metric
            # Dice is usually undefined (0/0) when both are empty. 
            # Here we define "Correctly Empty" as 1.0 and "Wrongly Empty" as 0.0.
            target_sum = torch.sum(target_masked)
            pred_sum = torch.sum(pred_masked)

            if target_sum == 0:
                if pred_sum == 0:
                    return torch.tensor(1.0, device=target.device)
                else:
                    return torch.tensor(0.0, device=target.device)

            # Use MONAI metric for non-empty targets
            # Ensure inputs have a batch dimension [B, C, H, W, D]
            # if they are [C, H, W, D], add a batch dim with .unsqueeze(0)
            if pred_masked.ndim == 3: # assuming 3D image
                pred_masked = pred_masked.unsqueeze(0).unsqueeze(0)
                target_masked = target_masked.unsqueeze(0).unsqueeze(0)

            dice_metric.reset() # Clear previous state
            dice_metric(y_pred=pred_masked, y=target_masked)

            # Unpack the tuple: the first element is the Dice score
            aggregate_results = dice_metric.aggregate()

            if isinstance(aggregate_results, (list, tuple)):
                val_results = aggregate_results[0].item()
            else:
                val_results = aggregate_results.item()
                            
            return val_results
        return compute_masked_val
         
    def _set_scheduler(self):
        """Define learning rate scheduler."""
        # If resuming, last_epoch should be start_epoch - 1
        last_epoch = self.config.get('resume_epoch', 0) - 1 if self.config.get('resume') else -1
        return CosineAnnealingLR(self.optimizer, self.config['num_epochs'], eta_min=0.0, last_epoch=last_epoch)
    
    def _get_transforms(self):
        """
        Composes a list of MONAI-based stochastic data augmentations for 3D medical imaging.

        This method defines a transformation pipeline split into two main phases:
        1. **Spatial Transforms**: Applied to the 'image', 'gt' (ground truth), and 
        'bridge_weight_map' to ensure geometric consistency across the data triplet.
        Includes flipping, elastic deformation, rotation, and zooming.
        2. **Intensity Transforms**: Applied strictly to the 'image' key to simulate 
        variations in acquisition noise, resolution, and contrast without altering 
        the labels.

        Returns:
            list: A list of MONAI dictionary-based transform objects.

        Note:
            - Spatial transforms use 'trilinear' interpolation for intensity data and 
            'nearest' for categorical ground truth masks.
            - The `Rand3DElasticd` padding mode is currently set to 'zeros'; 
        """
        transforms_list = [
            # --- Spatial Transforms ---
            
            # Flips data along axes
            RandFlipd(
                keys=["image", 'gt', 'bridge_weight_map'],
                prob=1
            ),
            # Applies smooth, non-linear grid deformation
            Rand3DElasticd(
                keys=["image", 'gt', 'bridge_weight_map'],
                sigma_range=(5, 25),
                magnitude_range=(10, 50),
                spatial_size=self.config['patch_size'],
                prob=0.2,
                mode=("trilinear", "nearest", "trilinear"),
                padding_mode="zeros",
            ),
            # Rotates data along random axes
            RandRotated(
                keys=["image", 'gt', 'bridge_weight_map'],
                range_x=np.pi,
                range_y=np.pi,
                range_z=np.pi,
                prob=0.5,
                mode=("trilinear", "nearest", "trilinear"),
                padding_mode="zeros",
            ),
            # Zooms in/out while maintaining patch size
            RandZoomd(
                keys=["image", 'gt', 'bridge_weight_map'],
                min_zoom=0.7,
                max_zoom=1.4,
                prob=0.2,
                keep_size=True,
                mode=("trilinear", "nearest", "trilinear"),
                padding_mode="constant",
            ),
            
            # --- Intensity Transforms ---
            # Adds Gaussian noise to image intensity
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                std=(0.15),
                mean=0.0
            ),
            
            # Blurs image using Gaussian filter
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5)
            ),
            
            # Adjusts brightness by multiplying intensity
            RandScaleIntensityd(
                keys=["image"],
                prob=0.2,
                factors=(-0.5, 0.5)
            ),
            
            # Simulates low resolution by downsampling then upsampling
            RandSimulateLowResolutiond(
                keys=["image"],
                prob=0.2,
                zoom_range=(0.25, 1.0),
                downsample_mode="nearest",
                align_corners=False
            ),
            
            # Non-linear contrast adjustment (inverted image)
            RandAdjustContrastd(
                keys=["image"],
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True
            ),
            
            # Non-linear contrast adjustment (standard image)
            RandAdjustContrastd(
                keys=["image"],
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=False
            )
        ]
        return transforms_list
    
    def _set_train_dataloader(self):
        """ Getting the list of cases for training and loading using MONAI (all into memory)"""
        data_list = []
        with open(self.config['data_split_json'], "r") as f:
            split = json.load(f)

        train_cases = split["train"]
        for train_case in train_cases:
            complete_data_dict = {}
            complete_data_dict["image"] = join(self.config['vol_data_path'], train_case)
            complete_data_dict["gt"] = join(self.config['label_data_path'], train_case)
            complete_data_dict["bridge_weight_map"] = join(self.config['bridge_weight_map_path'], train_case)
            data_list.append(complete_data_dict)
            
            if self.config['debug']:
                for i in range(30):
                    data_list.append(complete_data_dict)
                print(f"training using case: {data_list[0]}")
                break  # repeat 30 cases for debug mode

        print(f"Train cases: {len(train_cases)}")
        print(f"Some examples:")
        print(train_cases[:5])

        transforms_list = [   
                # Load image 
                LoadImaged(keys=["image", 'gt', 'bridge_weight_map']),
                EnsureChannelFirstd(keys=["image", 'gt', 'bridge_weight_map']),

                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),

                # Create a ROI mask for cropping 
                GetROIMaskdd(keys=["gt"], ignore_mask_value=2, new_key_names=["roi_mask"]),

                # Cropping random patches
                #ResizeWithPadOrCropd(keys=["image", "gt"], spatial_size=self.config['patch_size']),
                # Not ResizeWithPadOrCropd -> random crop
                #RandSpatialCropSamplesd(keys=["image", 'gt'], roi_size=self.config['patch_size'], num_samples=1, random_size=False),
                RandCropByPosNegLabeld(keys=["image", 'gt', 'bridge_weight_map'], label_key='roi_mask', spatial_size=self.config['patch_size'], pos=1, neg=0, num_samples=1, image_key=None),
                ResizeWithPadOrCropd(keys=["image", 'gt', 'bridge_weight_map'], spatial_size=self.config['patch_size'], mode="minimum"),
        ]
        
        if self.config['data_augmentation']:
            da_transforms_list = self._get_transforms()
            for da_trans in da_transforms_list:
                transforms_list.append(da_trans)
            
        # END transformations!
        # Create multi-resolution ground truths for deep supervision
        transforms_list.append(CopyItemsd(keys=["gt"], times=2, names=["gt_2layer", "gt_3layer"]))
        transforms_list.append(Resized(keys=["gt_2layer"], spatial_size=[s // 2 for s in self.config['patch_size']], mode='nearest'))
        transforms_list.append(Resized(keys=["gt_3layer"], spatial_size=[s // 4 for s in self.config['patch_size']], mode='nearest'))
        #roi_mask = ground_truth != 2 -> roi_mask = roi_mask.float()
        transforms_list.append(GetROIMaskdd(keys=["gt", "gt_2layer", "gt_3layer"], ignore_mask_value=2, new_key_names=["roi_mask","roi_mask_2layer", "roi_mask_3layer"]))
        transforms_list.append(GetBinaryLabeld(keys=["gt", "gt_2layer", "gt_3layer"], ignore_mask_value=2))
        transforms_list.append(EnsureTyped(keys=["image", "gt",  "gt_2layer", "gt_3layer", "roi_mask", "roi_mask_2layer", "roi_mask_3layer", "bridge_weight_map"], track_meta=False))

        transforms = Compose(transforms_list)
        
        print("Initializing Dataset...")
        train_ds = CacheDataset(
            data=data_list, 
            transform=transforms, 
            cache_rate=self.config['train_cache_rate'],  
            num_workers=self.config['num_workers'], 
            progress=True
        )

        print("Initializing Train DataLoader...")
        train_loader = monai.data.DataLoader(
            train_ds, 
            batch_size=self.config['batch_size'], 
            num_workers=self.config['num_workers'],
            shuffle=True,      
            pin_memory=True
        )
        return train_loader

    def _set_val_dataloader(self):
        # TODO make this static, i.e., not random crop!
        data_list = []
        with open(self.config['data_split_json'], "r") as f:
            split = json.load(f)

        val_cases = split["val"]

        if self.config['debug']: 
            print(f"Debug mode: Using training cases for validation dataloader")
            train_cases = split["train"]
            for train_case in train_cases:
                complete_data_dict = {}
                complete_data_dict["image"] = join(self.config['vol_data_path'], train_case)
                complete_data_dict["gt"] = join(self.config['label_data_path'], train_case)
                complete_data_dict["bridge_weight_map"] = join(self.config['bridge_weight_map_path'], train_case)
                data_list.append(complete_data_dict)
                print(f"Validation using case: {train_case}")
                break  # The same training sample for validation in debug mode
        else:
            for val_case in val_cases:
                complete_data_dict = {}
                complete_data_dict["image"] = join(self.config['vol_data_path'], val_case)
                complete_data_dict["gt"] = join(self.config['label_data_path'], val_case)
                complete_data_dict["bridge_weight_map"] = join(self.config['bridge_weight_map_path'], val_case)
                data_list.append(complete_data_dict)

        print(f"Val cases: {len(val_cases)}")
        print(f"Some examples:")
        print(val_cases[:5])

        transforms = Compose(
            [   
                # Load image 
                LoadImaged(keys=["image", 'gt', 'bridge_weight_map']),
                EnsureChannelFirstd(keys=["image", 'gt', 'bridge_weight_map']),
                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                # Create a ROI mask for cropping 
                GetROIMaskdd(keys=["gt"], ignore_mask_value=2, new_key_names=["roi_mask"]),
                # Get random patches
                RandCropByPosNegLabeld(keys=["image", 'gt', 'roi_mask', 'bridge_weight_map'], label_key='roi_mask', spatial_size=self.config['patch_size'], pos=1, neg=0, num_samples=1, image_key=None),
                ResizeWithPadOrCropd(keys=["image", "gt", "roi_mask", 'bridge_weight_map'], spatial_size=self.config['patch_size']),
                GetBinaryLabeld(keys=["gt"], ignore_mask_value=2),
                EnsureTyped(keys=["image", "gt", "roi_mask", 'bridge_weight_map'], track_meta=False)
            ]
        )

        print("Initializing Dataset...")
        val_ds = CacheDataset(
            data=data_list, 
            transform=transforms, 
            cache_rate=self.config['val_cache_rate'],  
            num_workers=self.config['num_workers'], 
            progress=True
        )
        
        print("Initializing Val DataLoader...")
        val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=self.config['num_workers'])
        return val_loader

    def _resume(self):
        if self.config.get('resume'):
            checkpoint = torch.load(self.config['resume'], map_location="cpu", weights_only=False) 
            # parameters
            self.start_epoch = checkpoint['epoch'] + 1 # To continue to the next epoch instead of repeating  
            self.val_value = checkpoint['val_value']
            # model load
            model_weights = checkpoint['model_weights']  
            self.model.load_state_dict(model_weights, strict=True)
            self.model = self.model.to(self.config['device'])
            # optimizer load
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.start_epoch = 0
            self.val_value = 0

    def save_vol(self, tensor, path):
        # prediction: torch.Tensor
        # shape example: [B, C, H, W] or [B, 1, H, W]

        tensor_cpu = tensor.detach().cpu()

        # Remove batch and channel dims if needed
        tensor_cpu = tensor_cpu[0]          
        tensor_cpu = tensor_cpu.squeeze(0)  

        tensor_np = tensor_cpu.numpy()

        affine = np.eye(4)  # identity affine (OK if no spatial metadata)

        nii = nib.Nifti1Image(tensor_np.astype(np.float32), affine)
        nib.save(nii, path)
        
    def saving_logic(self, best_val_value, val_avg_value, epoch):
        """ Logic to save the best model and periodic checkpoints """

        if best_val_value < val_avg_value: 
            best_val_value = val_avg_value
            save_path = join(self.model_save_path, f"model_best.pth")
            torch.save({
                    'epoch': epoch,
                    'model_weights': self.model.state_dict(),  
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_value': val_avg_value,
                }, save_path)
            print(f"Saved checkpoint: {save_path}")
        
        # Save Checkpoint
        if epoch % 10 == 0: 
            save_path = join(self.model_save_path, f"model_epoch_{epoch}.pth")
            torch.save({
                    'epoch': epoch,
                    'model_weights': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_value': val_avg_value,
                }, save_path)
            print(f"Saved checkpoint: {save_path}")
        return best_val_value

    def train_epoch(self, **kwargs):
        """Logic for a single training epoch. Returns average loss."""
        epoch = kwargs.get('epoch')
        optimizer = kwargs.get('optimizer')
        warmup = kwargs.get('warmup')
        
        self.model.train()
        epoch_loss = 0

        per_criterio_loss = {}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        for idx, batch_dict in enumerate(pbar):
            # Load input vol and gt
            input_patch = batch_dict['image'].to(self.config['device'])
            ground_truth = [
                batch_dict['gt_3layer'].to(self.config['device']),
                batch_dict['gt_2layer'].to(self.config['device']),
                batch_dict['gt'].to(self.config['device'])
            ]
            # Mask of the region to compute the loss
            roi_mask = [
                batch_dict['roi_mask_3layer'].to(self.config['device']),
                batch_dict['roi_mask_2layer'].to(self.config['device']),
                batch_dict['roi_mask'].to(self.config['device'])
            ]
             # pre-computed weight map to penalize bridges - set None for on-fly computation
            bridge_weight_map = batch_dict['bridge_weight_map'].to(self.config['device'])
            optimizer.zero_grad()

            # --- FP16 FORWARD PASS ---
            with autocast(device_type=self.config['device']):
                # Forward Pass
                # The model tries to predict the segmentation
                prediction = self.model(input_patch) # the prediction has the deep supervision! 3 outputs

                # Calculate Loss (Compare Prediction vs. GT)
                # prediction, target and roi_mask should be a list of 3 tensors
                train_loss, losses_dict = self.train_criterion(prediction, ground_truth, roi_mask=roi_mask, bridge_weight_map=bridge_weight_map) 

                # commented to avoid overwhelming 
                #losses_dict["train_loss"] = train_loss.item()
                #losses_dict["train_step"] = epoch*len(self.train_loader)+idx
                #self.wandb_run.log(
                #        losses_dict
                #)
                
            # Backward
            self.scaler.scale(train_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            epoch_loss += train_loss.item()
            for criterio_name in losses_dict.keys():
                if criterio_name in per_criterio_loss:
                    per_criterio_loss[criterio_name] += losses_dict[criterio_name]
                else:
                    per_criterio_loss[criterio_name] = losses_dict[criterio_name]
            pbar.set_postfix({"Loss": train_loss.item()})
        
        if warmup:
            pre_name = "warmup_"
        else:
            pre_name = ""
        if epoch%10 == 0:
            # Save a prediction
            self.save_vol(prediction[-1], join(self.preds_path, f"{pre_name}epoch_{epoch}_pred_train.nii.gz"))
            self.save_vol(input_patch, join(self.preds_path, f"{pre_name}epoch_{epoch}_input_train.nii.gz"))
            self.save_vol(ground_truth[-1], join(self.preds_path, f"{pre_name}epoch_{epoch}_gt_train.nii.gz"))

        train_avg_loss = epoch_loss / len(self.train_loader)
        print(f"{pre_name} Epoch {epoch} Finished. Avg Loss: {train_avg_loss:.6f}")
        # This will replace each element in the dict with the mean
        for criterio_name in losses_dict.keys():
            per_criterio_loss[criterio_name] = per_criterio_loss[criterio_name] / len(self.train_loader)
        return train_avg_loss, per_criterio_loss
    
    def val(self, **kwargs):
        """Logic for evaluation. Returns a dictionary of metrics."""
        epoch = kwargs.get('epoch')
        warmup = kwargs.get('warmup')
        self.model.eval()
        
        # General DSC validation value for quality controll
        val_value_sum = 0
        epoch_val_loss = 0
        # Add the per criterio val loss for checking overfitting
        per_criterio_val_loss = {}
        for val_criterio_name in self.config['criterion']:
            per_criterio_val_loss[f"val_{val_criterio_name}"] = 0

        pbar = tqdm(self.val_loader, desc=f"Val epoch {epoch}/{self.config['num_epochs']}")
        for idx, batch_dict in enumerate(pbar):
            input_patch = batch_dict['image'].to(self.config['device'])
            ground_truth = batch_dict['gt'].to(self.config['device'])
            # Create the mask of the region to compute the loss
            roi_mask = batch_dict['roi_mask'].to(self.config['device'])
            bridge_weight_map = batch_dict['bridge_weight_map'].to(self.config['device'])

            with torch.no_grad():
                # Forward Pass
                # The model tries to predict the segmentation
                prediction = self.model(input_patch)
                # Calculate DSC (Compare Prediction vs. GT)
                val_value = self.val_metric(pred=prediction, target=ground_truth, roi_mask=roi_mask)
                # Also compute val losses for logging (no deep supervision here)
                val_loss, val_losses_dict = self.train_criterion(prediction, ground_truth, roi_mask=roi_mask, bridge_weight_map=bridge_weight_map, deep_supervision_weights=[1.0]) 
                # commented to avoid overwhelming 
                #self.wandb_run.log({"val_value": val_value.item()})

            val_value_sum += val_value
            epoch_val_loss += val_loss.item()
            for val_criterio_name in val_losses_dict.keys():
                per_criterio_val_loss[f"val_{val_criterio_name}"] += val_losses_dict[f"{val_criterio_name}"]
            pbar.set_postfix({"DSC": val_value})

        if epoch%10 == 0:
            # Save a prediction
            if warmup:
                pre_name = "warmup_"
            else:
                pre_name = ""
            pred_save = sigmoid(prediction)
            pred_save[pred_save>0.5] = 1.0
            pred_save[pred_save<=0.5] = 0.0
            self.save_vol(prediction, join(self.preds_path, f"{pre_name}epoch_{epoch}_logits_val.nii.gz"))
            self.save_vol(pred_save, join(self.preds_path, f"{pre_name}epoch_{epoch}_pred_val.nii.gz"))
            self.save_vol(input_patch, join(self.preds_path, f"{pre_name}epoch_{epoch}_input_val.nii.gz"))
            self.save_vol(ground_truth, join(self.preds_path, f"{pre_name}epoch_{epoch}_gt_val.nii.gz"))

        # computing mean of metrics
        val_avg_value = val_value_sum / len(self.val_loader)
        val_avg_loss = epoch_val_loss / len(self.val_loader)

        for val_criterio_name in val_losses_dict.keys():
            per_criterio_val_loss[f"val_{val_criterio_name}"] = per_criterio_val_loss[f"val_{val_criterio_name}"] / len(self.val_loader)

        print(f"Epoch {epoch} with validation avg DSC: {val_avg_value:.6f} and avg Loss: {val_avg_loss:.6f}")
        return val_avg_value, val_avg_loss, per_criterio_val_loss

    def _freeze_weights(self, **kwargs):
        """ Logic to freeze the layers that were loaded and unfrozing all randomly initiated """
        print("\n" + "="*50)
        print("FREEZING REPORT")
        print("="*50)

        always_train_layers = ["seg_outputs"]
        
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            # If the parameter name is NOT in missing_keys, it means it was loaded successfully.
            is_forced_trainable = any(layer_name in name for layer_name in always_train_layers)

            if is_forced_trainable:
                param.requires_grad = True
                trainable_count += 1
                print(f"🔥 Trainable (Forced): {name}")
            # We want to FREEZE these.
            elif name not in self.mising_weights:
                param.requires_grad = False
                frozen_count += 1
            else:
                # It is a missing key (new head or mismatched shape), keep trainable.
                param.requires_grad = True
                trainable_count += 1
                print(f"🔥 Trainable Layer (missing/new): {name}")
                
        print(f"❄️  Frozen Layers (Backbone): {frozen_count}")
        print(f"🔥 Trainable Layers (New Heads): {trainable_count}")
        print("="*50 + "\n")
        return 0

    def unfreeze_next_layer(self):
        """
        Unfreezes the next 'block' of layers moving from Output -> Input.
        It identifies the first frozen parameter it encounters (in reverse)
        and unfreezes all parameters belonging to that same parent module.
        """
        print("\n" + "="*50)
        print("UNFREEZING NEXT LAYER")
        print("="*50)

        # 1. Get all parameters in REVERSE order (Output -> Input)
        # We convert to list so we can reverse it easily
        params = list(self.model.named_parameters())
        reversed_params = params[::-1]

        layer_prefix_to_unfreeze = None
        
        # 2. Find the first parameter that is currently FROZEN
        for name, param in reversed_params:
            if not param.requires_grad:
                # We found a frozen parameter!
                # Get its parent module name (remove .weight or .bias)
                # e.g., "encoder.blocks.1.conv1.weight" -> "encoder.blocks.1.conv1"
                parts = name.split(".")
                
                # Dynamic slice: if deep, go up 2 levels (block); if shallow, go up 1 level
                slice_idx = -2 if len(parts) > 2 else -1
                
                layer_prefix_to_unfreeze = ".".join(parts[:slice_idx])
                
                # Check for empty string edge case
                if not layer_prefix_to_unfreeze:
                    layer_prefix_to_unfreeze = parts[0]

                print(f"🧊 Found frozen parameter: {name}")
                print(f"🎯 Target prefix to unfreeze: {layer_prefix_to_unfreeze}")
                break
                
        # 3. If everything is already trainable, we are done
        if layer_prefix_to_unfreeze is None:
            print("🎉 All layers are already unfrozen!")
            return True

        # 4. Unfreeze all parameters that belong to this detected module
        count = 0
        print(f"🔓 Unfreezing Block: {layer_prefix_to_unfreeze}")
        
        for name, param in self.model.named_parameters():
            if name.startswith(layer_prefix_to_unfreeze):
                param.requires_grad = True
                count += 1
                # print(f"   -> {name}") # Uncomment for verbose details

        print(f"✅ Unfrozen {count} parameters in this step.")
        
        # 5. Quick check on what remains frozen
        remaining_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        print(f"❄️  Remaining Frozen Parameters: {remaining_frozen}")
        print("="*50 + "\n")
        return False

    def warmup_train(self, **kwargs):
        """ Perform warmup by freezen all loaded weights and training only the new weights """
        pre_train_config = self.config['pre_train_config']

        with open(pre_train_config, "r") as f:
            pre_train_config = json.load(f)

        # Prepare self.model by doing smart freezing
        self._freeze_weights()

        # Take old learning rate (used for pre-training)
        warm_up_opt = optim.AdamW(self.model.parameters(), lr=pre_train_config['learning_rate'])
        
        warmup_epoch = 0

        ## warmup loop for the first set of frozen weights
        for warmup_epoch in range(self.config['warmup_epochs']):
            # train one warmup epoch 
            train_avg_loss, per_criterio_loss = self.train_epoch(
                epoch=warmup_epoch,
                optimizer=warm_up_opt,
                warmup=True
            )
            # Perform evaluation 
            val_avg_value, val_avg_loss, per_criterio_val_loss = self.val(
                epoch=warmup_epoch, 
                warmup=True
            ) 

            warmup_log_train_data = {
                    "warmup_epoch": warmup_epoch,
                    "train_avg_loss": train_avg_loss,
                    "val_Dice": val_avg_value,
                    "val_avg_loss": val_avg_loss,
                    "lr": warm_up_opt.param_groups[0]['lr']
                }
            # Add the per criterio losses to wandb
            train_fullres_loss = 0.0
            for criterio_name in per_criterio_loss.keys():
                warmup_log_train_data[criterio_name] = per_criterio_loss[criterio_name]
                if criterio_name.endswith("_fullres"):
                    train_fullres_loss += per_criterio_loss[criterio_name]
            warmup_log_train_data["train_fullres_loss"] = train_fullres_loss # making sure the train and val are comparable

            for val_criterio_name in per_criterio_val_loss.keys():
                warmup_log_train_data[val_criterio_name] = per_criterio_val_loss[val_criterio_name]

            self.wandb_run.log(
                warmup_log_train_data
            )
        ## Progressive unfrozening and warmup (for decoder to encoder)  
        ALL_UNFROZEN = False
        warm_up_opt = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        while not ALL_UNFROZEN:
            warmup_epoch += 1
            # Unfreeze the next layer
            ALL_UNFROZEN = self.unfreeze_next_layer()

            if ALL_UNFROZEN:
                print("🏁 Full model unfrozen. Proceeding to main training.")
                break
            
            # train one warmup epoch 
            train_avg_loss, per_criterio_loss = self.train_epoch(
                epoch=warmup_epoch,
                optimizer=warm_up_opt,
                warmup=True
            )
            # Perform evaluation 
            val_avg_value, val_avg_loss, per_criterio_val_loss = self.val( # TODO
                epoch=warmup_epoch, 
                warmup=True
            )

            warmup_log_train_data = {
                    "warmup_epoch": warmup_epoch,
                    "train_avg_loss": train_avg_loss,
                    "val_Dice": val_avg_value,
                    "val_avg_loss": val_avg_loss,
                    "lr": warm_up_opt.param_groups[0]['lr']
                }
            
            # Add the per criterio losses to wandb
            train_fullres_loss = 0.0
            for criterio_name in per_criterio_loss.keys():
                warmup_log_train_data[criterio_name] = per_criterio_loss[criterio_name]
                if criterio_name.endswith("_fullres"):
                    train_fullres_loss += per_criterio_loss[criterio_name]
            warmup_log_train_data["train_fullres_loss"] = train_fullres_loss # making sure the train and val are comparable

            for val_criterio_name in per_criterio_val_loss.keys():
                warmup_log_train_data[val_criterio_name] = per_criterio_val_loss[val_criterio_name]

            self.wandb_run.log(
                warmup_log_train_data
            )

        return 0
        
    def train_loop(self, **kwargs):
        """Standardized training loop."""
        best_val_value = self.val_value
        
        # Doing warmup stage
        if self.config['warmup_epochs'] > 0 and self.config['resume'] is None:
            self.warmup_train()
        
        # Make sure all weights are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        for self.epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            # Train one epoch
            train_avg_loss, per_criterio_loss = self.train_epoch(
                epoch=self.epoch,
                optimizer=self.optimizer,
                warmup=False
            )
            # Perform evaluation 
            val_avg_value, val_avg_loss, per_criterio_val_loss = self.val(
                epoch=self.epoch, 
                warmup=False
            )

            # Save in wandb
            log_train_data = {
                    "epoch_train": self.epoch,
                    "train_total_avg_loss": train_avg_loss,
                    "val_Dice": val_avg_value,
                    "val_avg_loss": val_avg_loss,
                    "lr_train": self.optimizer.param_groups[0]['lr'] 
                }
            # Add the per criterio losses to wandb
            train_fullres_loss = 0.0
            for criterio_name in per_criterio_loss.keys():
                log_train_data[criterio_name] = per_criterio_loss[criterio_name]
                if criterio_name.endswith("_fullres"):
                    train_fullres_loss += per_criterio_loss[criterio_name]
            log_train_data["train_fullres_loss"] = train_fullres_loss # making sure the train and val are comparable

            for val_criterio_name in per_criterio_val_loss.keys():
                log_train_data[val_criterio_name] = per_criterio_val_loss[val_criterio_name]

            self.wandb_run.log(
                log_train_data
            )

            # Checking if saving 
            best_val_value = self.saving_logic(
                best_val_value=best_val_value, 
                val_avg_value=val_avg_value, 
                epoch=self.epoch
            )

            # Applying learning rate Cosine Annealing
            self.scheduler.step()
