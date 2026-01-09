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
    EnsureChannelFirstd
)

# Local Project Imports
from abs_training import BaseTrainer
from stunet_model import STUNetSegmentation # We want to seg now -> Last layer is has 2 output channels instead of one # STUNetReconstruction

sys.path.append(os.path.abspath("/mounts/disk4_tiago_e_andre/vesuvius/Vesuvius/utils"))

from cldice.cldice import soft_cldice
from AntiBridgeLoss import AntiBridgeLoss

class main_train_STU_Net(BaseTrainer):
    def __init__(self, config):
        self.config = config

        # build the pipeline
        self.model, self.mising_weights = self._build_model() # DONE
        self.optimizer = self._set_optimizer() # DONE
        self.train_criterion = self._set_train_criterion() # DONE
        self.val_metric = self._set_val_metric() # DONE
        self.scheduler = self._set_scheduler() # DONE
        if not self.config['debug']:
            self.wandb_run = self._set_wandb_checkpoint() # DONE
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
        
        # Initialize the new Segmentation Model (2 output channels)
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
            'CLDICE': soft_cldice(
                iter_=10, 
                smooth=1e-5, 
                exclude_background=False
            ),
            'AntiBridge': AntiBridgeLoss(
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
        def combined_loss(pred, target, roi_mask, bridge_weight_map):
            total_loss = 0.0
            losses_dict = {}

            for crit in loss_dict.keys():
                loss_fn, w = loss_dict[crit]

                if crit=='DSC' or crit=='Tversky' or crit=='CLDICE':
                    # apply activation befofre masking to ensure 0 in the region to ignore
                    pred_masked = sigmoid(pred)*roi_mask
                    target_masked = target*roi_mask
                    loss_here = loss_fn(pred_masked, target_masked)

                elif crit=='BCE' or crit=='Focal':
                    # in here, the functions use sigmoid internally
                    if crit=='BCE':
                        loss_raw = loss_fn(pred, target, reduction='none')
                    else:
                        loss_raw = loss_fn(pred, target)
                    loss_masked = loss_raw * roi_mask
                    loss_here = loss_masked.sum() / (roi_mask.sum() + 1e-8)

                elif crit=='AntiBridge':
                    loss_here = loss_fn(pred, target, roi_mask, bridge_weight_map)

                total_loss += w * loss_here
                losses_dict[crit] = loss_here
            return total_loss, losses_dict
        
        return combined_loss 

    def _set_val_metric(self):
        """Define the loss function."""
        val_metric = DiceMetric(
            include_background=True
        )
        def compute_masked_val(pred, target, roi_mask):
            pred_masked = sigmoid(pred) * roi_mask
            pred_masked = (pred_masked > 0.5).float()
            target_masked = target * roi_mask
            return val_metric(pred_masked, target_masked)
        return compute_masked_val
         
    def _set_scheduler(self):
        """Define learning rate scheduler."""
        # If resuming, last_epoch should be start_epoch - 1
        last_epoch = self.config.get('resume_epoch', 0) - 1 if self.config.get('resume') else -1
        return CosineAnnealingLR(self.optimizer, self.config['num_epochs'], eta_min=0.0, last_epoch=last_epoch)
    
    def _set_train_dataloader(self):
        """ Getting the list of cases for training and loading using MONAI (all into memory)"""
        # TODO
        data_list = []
        with open(self.config['data_split_json'], "r") as f:
            split = json.load(f)

        train_cases = split["train"]
        for train_case in train_cases:
            complete_data_dict = {}
            complete_data_dict["image"] = join(self.config['vol_data_path'], train_case)
            complete_data_dict["gt"] = join(self.config['label_data_path'], train_case)
            data_list.append(complete_data_dict)

        print(f"Train cases: {len(train_cases)}")
        print(f"Some examples:")
        print(train_cases[:5])

        transforms = Compose(
            [   
                # Load image 
                LoadImaged(keys=["image", 'gt']),
                EnsureChannelFirstd(keys=["image", 'gt']),
                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                ResizeWithPadOrCropd(keys=["image", "gt"], spatial_size=self.config['patch_size']),
                EnsureTyped(keys=["image", "gt"], track_meta=False)
            ]
        )

        print("Initializing Dataset...")
        train_ds = CacheDataset(
            data=data_list, 
            transform=transforms, 
            cache_rate=self.config['train_cache_rate'],  
            num_workers=self.config['num_workers'], 
            progress=True
        )
        
        print("Initializing Val DataLoader...")
        train_loader = monai.data.DataLoader(train_ds, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'])
        return train_loader

    def _set_val_dataloader(self):
        # TODO
        data_list = []
        with open(self.config['data_split_json'], "r") as f:
            split = json.load(f)

        val_cases = split["val"]
        for val_case in val_cases:
            complete_data_dict = {}
            complete_data_dict["image"] = join(self.config['vol_data_path'], val_case)
            complete_data_dict["gt"] = join(self.config['label_data_path'], val_case)
            data_list.append(complete_data_dict)

        print(f"Val cases: {len(val_cases)}")
        print(f"Some examples:")
        print(val_cases[:5])

        transforms = Compose(
            [   
                # Load image 
                LoadImaged(keys=["image", 'gt']),
                EnsureChannelFirstd(keys=["image", 'gt']),
                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                ResizeWithPadOrCropd(keys=["image", "gt"], spatial_size=self.config['patch_size']),
                EnsureTyped(keys=["image", "gt"], track_meta=False)
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
        # TODO
        if self.config.get('resume'):
            checkpoint = torch.load(self.config['resume'], map_location="cpu", weights_only=False) 
            # parameters
            self.start_epoch = checkpoint['epoch'] + 1 # To continue to the next epoch instead of repeating  
            self.val_loss = checkpoint['val_loss']
            # model load
            model_weights = checkpoint['model_weights']  
            self.model.load_state_dict(model_weights, strict=True)
            self.model = self.model.to(self.config['device'])
            # optimizer load
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.start_epoch = 0
            self.val_loss = 10000

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
        
    def saving_logic(self, best_val_loss, val_avg_loss, epoch):
        # TODO
        if best_val_loss > val_avg_loss: 
            best_val_loss = val_avg_loss
            save_path = join(self.model_save_path, f"model_best.pth")
            torch.save({
                    'epoch': epoch,
                    'model_weights': self.model.state_dict(),  
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_avg_loss,
                }, save_path)
            print(f"Saved checkpoint: {save_path}")

        # Save Checkpoint
        if epoch % 10 == 0:
            save_path = join(self.model_save_path, f"model_epoch_{epoch}.pth")
            torch.save({
                    'epoch': epoch,
                    'model_weights': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_avg_loss,
                }, save_path)
            print(f"Saved checkpoint: {save_path}")
        return best_val_loss
   
    def train_epoch(self, **kwargs):
        """Logic for a single training epoch. Returns average loss."""
        epoch = kwargs.get('epoch')
        optimizer = kwargs.get('optimizer')
        warmup = kwargs.get('warmup')
        
        self.model.train()
        epoch_loss = 0

        per_criterio_loss = {}
        for criterio_name in self.config['criterion']:
            per_criterio_loss[criterio_name] = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        for idx, batch_dict in enumerate(pbar):
            # Load input vol and gt
            input_patch = batch_dict['image'].to(self.config['device'])
            ground_truth = batch_dict['gt'].to(self.config['device'])
            # Create the mask of the region to compute the loss
            roi_mask = ground_truth != 2
            roi_mask = roi_mask.float().to(self.config['device'])

            optimizer.zero_grad()
            # --- FP16 FORWARD PASS ---
            with autocast(device_type=self.config['device']):
                # Forward Pass
                # The model tries to predict the segmentation
                prediction = self.model(input_patch) # TODO the prediction has the deep supervision! 5 outputs

                # Calculate Loss (Compare Prediction vs. Clean)
                train_loss, losses_dict = self.train_criterion(prediction, ground_truth, roi_mask=roi_mask) # TODO make sure train_criterion takes care

                # commented to avoid overwhelming TODO
                losses_dict["train_loss"] = train_loss.item()
                losses_dict["train_step"] = epoch*len(self.train_loader)+idx
                self.wandb_run.log(
                        losses_dict
                )
                
            # Backward
            self.scaler.scale(train_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            epoch_loss += train_loss.item()
            for criterio_name in self.config['criterion']:
                per_criterio_loss[criterio_name] += losses_dict[criterio_name]
            pbar.set_postfix({"Loss": train_loss.item()})
            
        # Save a prediction
        if warmup:
            pre_name = "warmup_"
        else:
            pre_name = ""
        self.save_vol(prediction, join(self.preds_path, f"{pre_name}epoch_{epoch}_pred_train.nii.gz"))
        self.save_vol(input_patch, join(self.preds_path, f"{pre_name}epoch_{epoch}_input_train.nii.gz"))
        self.save_vol(ground_truth, join(self.preds_path, f"{pre_name}epoch_{epoch}_gt_train.nii.gz"))

        train_avg_loss = epoch_loss / len(self.train_loader)
        print(f"{pre_name} Epoch {epoch} Finished. Avg Loss: {train_avg_loss:.6f}")
        # This will replace each element in the dict with the mean
        for criterio_name in self.config['criterion']:
            per_criterio_loss[criterio_name] = per_criterio_loss[criterio_name] / len(self.train_loader)
        return train_avg_loss, per_criterio_loss
    
    def val(self, **kwargs):
        """Logic for evaluation. Returns a dictionary of metrics."""
        epoch = kwargs.get('epoch')
        warmup = kwargs.get('warmup')

        self.model.eval()
        val_loss_sum = 0

        pbar = tqdm(self.val_loader, desc=f"Val epoch {epoch}/{self.config['num_epochs']}")

        for batch_dict in pbar:
            input_patch = batch_dict['image'].to(self.config['device'])
            ground_truth = batch_dict['gt'].to(self.config['device'])
            # Create the mask of the region to compute the loss
            roi_mask = ground_truth != 2
            roi_mask = roi_mask.float().to(self.config['device'])

            # --- FP16 FORWARD PASS ---
            with torch.no_grad():
                # Forward Pass
                # The model tries to predict the segmentation
                prediction = self.model(input_patch)
                # Calculate Loss (Compare Prediction vs. GT)
                val_loss = self.val_metric(pred=prediction, target=ground_truth, roi_mask=roi_mask)
                
                # commented to avoid overwhelming # TODO remove
                self.wandb_run.log({"val_loss": val_loss.item()})

            val_loss_sum += val_loss.item()
            pbar.set_postfix({"Loss": val_loss.item()})

        # Save a prediction
        if warmup:
            pre_name = "warmup_"
        else:
            pre_name = ""
        self.save_vol(prediction, join(self.preds_path, f"{pre_name}epoch_{epoch}_pred_val.nii.gz"))
        self.save_vol(input_patch, join(self.preds_path, f"{pre_name}epoch_{epoch}_input_val.nii.gz"))
        self.save_vol(ground_truth, join(self.preds_path, f"{pre_name}epoch_{epoch}_gt_val.nii.gz"))
        val_avg_loss = val_loss_sum / len(self.val_loader)
        print(f"Epoch {self.epoch} with validation avg Loss: {val_avg_loss:.6f}")
        return val_avg_loss

    def _freeze_weights(self, **kwargs):
        """ Logic to freeze the layers that were loaded and unfrozing all randomly initiated """
        print("\n" + "="*50)
        print("FREEZING REPORT")
        print("="*50)
        
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            # If the parameter name is NOT in missing_keys, it means it was loaded successfully.
            # We want to FREEZE these.
            if name not in self.mising_weights:
                param.requires_grad = False
                frozen_count += 1
            else:
                # It is a missing key (new head or mismatched shape), keep trainable.
                param.requires_grad = True
                trainable_count += 1
                print(f"🔥 Trainable Layer: {name}")
                
        print(f"❄️  Frozen Layers (Backbone): {frozen_count}")
        print(f"🔥 Trainable Layers (New Heads): {trainable_count}")
        print("="*50 + "\n")
        return 0

    def warmup_train(self, **kwargs):
        """ Perform warmup by freezen all loaded weights and training only the new weights """
        pre_train_config = self.config['pre_train_config']

        # Prepare self.model by doing smart freezing
        self._freeze_weights()

        # Take old learning rate (used for pre-training)
        warm_up_opt = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        
        warmup_epoch = 0
        for warmup_epoch in range(self.config['warmup_epochs']):
            # train one warmup epoch 
            train_avg_loss, per_criterio_loss = self.train_epoch(
                epoch=warmup_epoch,
                optimizer=warm_up_opt,
                warmup=True
            )
            # Perform evaluation 
            val_avg_loss = self.val(
                epoch=warmup_epoch, 
                warmup=True
            )

            warmup_log_train_data = {
                    "warmup_epoch": warmup_epoch,
                    "train_avg_loss": train_avg_loss,
                    "val_avg_loss": val_avg_loss,
                    "lr": warm_up_opt.param_groups[0]['lr']
                }
            for criterio_name in per_criterio_loss.keys():
                warmup_log_train_data[criterio_name] = per_criterio_loss[criterio_name]

            self.wandb_run.log(
                warmup_log_train_data
            )
        return 0
        
    def train_loop(self, **kwargs):
        """Standardized training loop."""
        best_val_loss = self.val_loss
        
        # Doing warmup stage
        if self.config['warmup_epochs'] > 0:
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
            val_avg_loss = self.val(
                epoch=self.epoch, 
                warmup=False
            )

            # Save in wandb
            log_train_data = {
                    "epoch": self.epoch,
                    "train_avg_loss": train_avg_loss,
                    "val_avg_loss": val_avg_loss,
                    "lr": self.optimizer.param_groups[0]['lr'] 
                }
            for criterio_name in per_criterio_loss.keys():
                log_train_data[criterio_name] = per_criterio_loss[criterio_name]
            self.wandb_run.log(
                log_train_data
            )

            # Checking if saving 
            best_val_loss = self.saving_logic(
                best_val_loss=best_val_loss, 
                val_avg_loss=val_avg_loss, 
                epoch=self.epoch
            )

            # Applying learning rate Cosine Annealing
            self.scheduler.step()
