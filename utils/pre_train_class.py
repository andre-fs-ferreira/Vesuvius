# Standard Library Imports
import os
import datetime
from os import listdir, makedirs
from os.path import join

# Third-Party Library Imports
import numpy as np
import torch
import torch.optim as optim
import nibabel as nib
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

# MONAI Specific Imports
import monai
from monai.data import CacheDataset
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
from stunet_model import STUNetReconstruction
from criterion_l1 import CriterionL1
from zarr_dataloader import ZarrVolumeDataset


class pre_training_STU_Net(BaseTrainer):
    def __init__(self, config):
        self.config = config

        # build the pipeline
        self.model = self._build_model()
        self.optimizer = self._set_optimizer()
        self.train_criterion = self._set_train_criterion()
        self.val_metric = self._set_val_metric()
        self.scheduler = self._set_scheduler()
        self.wandb_run = self._set_wandb_checkpoint()
        self.train_loader = self._set_train_dataloader()
        self.val_loader = self._set_val_dataloader()

        # set scaler for mix precision 
        self.scaler = GradScaler()

        # check resume
        self._resume()

    def _set_wandb_checkpoint(self):
        """ Define the wandb run """

        x = datetime.datetime.now()

        date_string = f"{x.day}-{x.month}-{x.year}"
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="faking_it",
            # Set the wandb project where this run will be logged.
            project="Vesuvius",
            # Track hyperparameters and run metadata.
            config=self.config,
            name=f"Pre_training_with_5_scrolls_{date_string}", 
            dir=self.config['save_dir']
        )
        self.model_save_path = join(run.dir, "model")
        makedirs(self.model_save_path, exist_ok=True)

        self.preds_path = join(run.dir, "preds")
        makedirs(self.preds_path, exist_ok=True)

        return run 

    def _build_model(self):
        """Initialize the neural network architecture."""
        model = STUNetReconstruction()
        state_dict = torch.load(self.config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.config['device'])
        return model

    def _set_optimizer(self):
        """Define the optimizer (e.g., Adam, SGD)."""
        return optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

    def _set_train_criterion(self):
        """Define the loss function."""
        return CriterionL1(mask_weight=self.config['mask_weight']) #nn.L1Loss() # Sharpness preference (Better for restoration)

    def _set_val_metric(self):
        """Define the loss function."""
        return CriterionL1(mask_weight=self.config['mask_weight']) #nn.L1Loss() # Sharpness preference (Better for restoration)
    
    def _set_scheduler(self):
        """Define learning rate scheduler."""
        # If resuming, last_epoch should be start_epoch - 1
        last_epoch = self.config.get('resume_epoch', 0) - 1 if self.config.get('resume') else -1
        return CosineAnnealingLR(self.optimizer, self.config['num_epochs'], eta_min=0.0, last_epoch=last_epoch)
    
    def _set_train_dataloader(self):
        transform_input = Compose(
            [
                # Load image will be handeled by the lazzy zarr loading data
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                EnsureTyped(keys=["image"])
            ]
        )


        # The Corruption Transforms
        # We want to force the model to fix heavy defects.
        transform_deform = Compose(
            [
                # Cut out 10 holes, each spatial size roughly 32x48x48
                # 35% of data loss
                RandCoarseDropoutd(
                    keys=["image", "tracking_mask"],
                    holes=10, 
                    spatial_size=(32, 48, 48), 
                    fill_value=0,
                    prob=1.0 # Always apply
                ),
                # Add noise (not do it)
                #RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
                EnsureTyped(keys=["image", "tracking_mask"])
            ]
        )

        print("Initializing Dataset...")
        ds = ZarrVolumeDataset(
            zarr_path=self.config['data_path'], 
            transform_input=transform_input,
            transform_deform=transform_deform,
            patch_size=self.config['patch_size']
        )

        print("Initializing Train DataLoader...")
        train_loader = monai.data.DataLoader(ds, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'])
        return train_loader

    def _set_val_dataloader(self):
        data_list = []
        train_images_nii = join(self.config['val_data_path'], 'train_images_nii')
        tracking_mask_path = join(self.config['val_data_path'], 'tracking_mask.nii.gz')
        for file_name in listdir(train_images_nii):
            complete_path = join(train_images_nii, file_name)
            
            data_list.append(
                {
                    "image": complete_path,
                    "tracking_mask": tracking_mask_path},
            )
            
            if len(data_list)>=100:
                break
        transforms = Compose(
            [   
                # Load image 
                LoadImaged(keys=["image", 'tracking_mask']),
                EnsureChannelFirstd(keys=["image", 'tracking_mask']),
                # Normalize uint8 input
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                ResizeWithPadOrCropd(keys=["image"], spatial_size=self.config['patch_size']),
                # Make a clean copy
                # Copy the image tensor to a new key
                CopyItemsd(keys=["image"], times=1, names=["deform_patch"]),

                # Cut out 10 holes, each spatial size roughly 32x48x48
                # 35% of data loss
                RandCoarseDropoutd(
                    keys=["deform_patch", 'tracking_mask'],
                    holes=10, 
                    spatial_size=(32, 48, 48), 
                    fill_value=0,
                    prob=1.0 # Always apply
                ),
                EnsureTyped(keys=["image", "deform_patch", 'tracking_mask'], track_meta=False)
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
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.config['num_epochs']}")
        for idx, batch_dict in enumerate(pbar):
            clean_patch = batch_dict['clean_patch'].to(self.config['device'])
            deform_patch = batch_dict['deform_patch'].to(self.config['device'])
            dropout_mask = batch_dict['dropout_mask'].to(self.config['device'])
            
            self.optimizer.zero_grad()
            # --- FP16 FORWARD PASS ---
            with autocast(device_type=self.config['device']):
                # Forward Pass
                # The model tries to predict the CLEAN image from the DEFORMED input
                prediction = self.model(deform_patch)
                # Calculate Loss (Compare Prediction vs. Clean)
                train_loss = self.train_criterion(prediction, clean_patch, dropout_mask)

                # commented to avoid overwhelming 
                # run.log(
                #     {
                #         "train_loss": train_loss.item(),
                #         "train_step": epoch*len(train_loader)+idx
                    
                #     }
                # )
                
            # Backward
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += train_loss.item()
            pbar.set_postfix({"Loss": train_loss.item()})
            
        # Save a prediction
        self.save_vol(prediction, join(self.preds_path, f"{self.epoch}_pred_train.nii.gz"))
        self.save_vol(deform_patch, join(self.preds_path, f"{self.epoch}_deform_train.nii.gz"))
        self.save_vol(clean_patch, join(self.preds_path, f"{self.epoch}_clean_train.nii.gz"))
        self.save_vol(dropout_mask, join(self.preds_path, f"{self.epoch}_mask_train.nii.gz"))
        train_avg_loss = epoch_loss / len(self.train_loader)
        print(f"Epoch {self.epoch} Finished. Avg Loss: {train_avg_loss:.6f}")
        return train_avg_loss
    
    def val(self, **kwargs):
        """Logic for evaluation. Returns a dictionary of metrics."""
        self.model.eval()
        val_loss_sum = 0
        pbar = tqdm(self.val_loader, desc=f"Val epoch {self.epoch}/{self.config['num_epochs']}")
        for batch_dict in pbar:
            clean_patch = batch_dict['image'].to(self.config['device'])
            deform_patch = batch_dict['deform_patch'].to(self.config['device'])
            dropout_mask = 1 - batch_dict['tracking_mask']
            dropout_mask = dropout_mask.to(self.config['device'])
            # --- FP16 FORWARD PASS ---
            with torch.no_grad():
                # Forward Pass
                # The model tries to predict the CLEAN image from the DEFORMED input
                prediction = self.model(deform_patch)
                # Calculate Loss (Compare Prediction vs. Clean)
                val_loss = self.val_metric(prediction, clean_patch, dropout_mask)
                # commented to avoid overwhelming 
                #run.log({"val_loss": val_loss.item()})

            val_loss_sum += val_loss.item()
            pbar.set_postfix({"Loss": val_loss.item()})

        # Save a prediction
        self.save_vol(prediction, join(self.preds_path, f"{self.epoch}_pred.nii.gz"))
        self.save_vol(deform_patch, join(self.preds_path, f"{self.epoch}_deform.nii.gz"))
        self.save_vol(clean_patch, join(self.preds_path, f"{self.epoch}_clean.nii.gz"))
        self.save_vol(dropout_mask, join(self.preds_path, f"{self.epoch}_mask.nii.gz"))
        val_avg_loss = val_loss_sum / len(self.val_loader)
        print(f"Epoch {self.epoch} with validation avg Loss: {val_avg_loss:.6f}")
        return val_avg_loss

    
    def train_loop(self, **kwargs):
        """Standardized training loop."""
        best_val_loss = self.val_loss
    
        for self.epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            # Train one epoch
            train_avg_loss = self.train_epoch(
                epoch=self.epoch
            )

            # Run validation step
            val_avg_loss = self.val( 
                epoch=self.epoch
            )

            # Save in wandb
            self.wandb_run.log(
                {
                    "epoch": self.epoch,
                    "train_avg_loss": train_avg_loss,
                    "val_avg_loss": val_avg_loss,
                    "lr": self.optimizer.param_groups[0]['lr'] 
                }
            )

            # Checking if saving 
            best_val_loss = self.saving_logic(
                best_val_loss=best_val_loss, 
                val_avg_loss=val_avg_loss, 
                epoch=self.epoch
            )

            # Applying learning rate Cosine Annealing
            self.scheduler.step()
