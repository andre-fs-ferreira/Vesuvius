import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import spectral_norm

class Generator(torch.nn.Module):
    def __init__(self, in_channels=2, first_channels=16, out_channels=1, use_checkpointing=True):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.first_channels = first_channels
        self.out_channels = out_channels
        self.use_checkpointing = use_checkpointing
        
        # ----------------------
        # ENCODER
        # ----------------------
        self.enc1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.first_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(self.first_channels),
            nn.LeakyReLU(inplace=True)
        ) 

        self.enc2 = nn.Sequential(
            nn.Conv3d(self.first_channels, self.first_channels*2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(self.first_channels*2),
            nn.LeakyReLU(inplace=True)
        ) 

        # ----------------------
        # BOTTLENECK (Hybrid Dilated)
        # ----------------------
        # FIX: Added Dilation=4 to close larger holes.
        self.bottleneck = nn.Sequential(
            # 1. Standard
            nn.Conv3d(self.first_channels*2, self.first_channels*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.first_channels*2),
            nn.LeakyReLU(inplace=True),
            
            # 2. Wide View (Dilation 2)
            nn.Conv3d(self.first_channels*2, self.first_channels*2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm3d(self.first_channels*2),
            nn.LeakyReLU(inplace=True),

            # 3. Wider View (Dilation 4) <--- ADDED
            # Receptive field increases significantly without memory cost
            nn.Conv3d(self.first_channels*2, self.first_channels*2, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.InstanceNorm3d(self.first_channels*2),
            nn.LeakyReLU(inplace=True),
            
            # 4. Consolidate
            nn.Conv3d(self.first_channels*2, self.first_channels*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.first_channels*2)
        )

        # ----------------------
        # DECODER
        # ----------------------
        
        # Dec 2: It's okay to use Nearest here because Dec 1 will clean it up.
        self.dec2 = nn.Sequential(
            nn.Conv3d(self.first_channels*4, self.first_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(self.first_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest') 
        )

        # Dec 1: FIX -> Switched to TRILINEAR
        # Since we feed directly into a 1x1 Conv, 'nearest' would leave blocky artifacts.
        # Trilinear ensures smooth gradients at the final boundary.
        self.dec1 = nn.Sequential(
            nn.Conv3d(self.first_channels + self.first_channels*2, self.first_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(self.first_channels),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # <--- CHANGED
        )

        # ----------------------
        # OUTPUT
        # ----------------------
        # FIX: bias=True. 
        # Allows the network to learn a global "confidence shift" easily.
        self.out = nn.Sequential(
            nn.Conv3d(self.first_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True) 
        )
        
        # Zero Init for Residual Learning
        nn.init.constant_(self.out[0].weight, 0)
        nn.init.constant_(self.out[0].bias, 0) # Init the bias to 0 too

    def forward(self, x):
        def run_layer(layer, inp):
            if self.use_checkpointing and inp.requires_grad:
                return checkpoint(layer, inp, use_reentrant=False)
            return layer(inp)

        # Encoder
        x_enc1 = self.enc1(x)
        x_enc2 = run_layer(self.enc2, x_enc1)
        
        # Bottleneck
        x_bottleneck = run_layer(self.bottleneck, x_enc2)
        
        # Decoder 2
        cat_d2 = torch.cat([x_enc2, x_bottleneck], dim=1)
        x_dec2 = run_layer(self.dec2, cat_d2)
        
        # Decoder 1
        cat_d1 = torch.cat([x_enc1, x_dec2], dim=1)
        x_dec1 = run_layer(self.dec1, cat_d1)
        
        # Delta (Correction)
        delta_logits = self.out(x_dec1)
        
        # Input Logits (Assumption: Input Ch 1 is ALREADY logits)
        input_logits = x[:, 1:2, :, :, :]
        if self.training and input_logits.max() <= 1.0 and input_logits.min() >= 0.0:
            import warnings
            warnings.warn("Input Channel 1 looks like Probabilities (0-1), but the network expects Logits (-inf to inf). Residual learning will fail!")
        
        return input_logits + delta_logits

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, initial_filters=16):
        super(PatchDiscriminator, self).__init__()
        
        # Input: (B, 2, 320, 320, 320)
        # Channel 0: CT Scan
        # Channel 1: Predicted Logits (or Probability)
        
        # We use Spectral Norm on all layers.
        # It is the modern standard to prevent the Discriminator from becoming too strong too fast.
        
        # Layer 1: 320 -> 160
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, initial_filters, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 2: 160 -> 80
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(initial_filters, initial_filters*2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm3d(initial_filters*2), # Norm helps stability
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 3: 80 -> 40
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv3d(initial_filters*2, initial_filters*4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm3d(initial_filters*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 4: 40 -> 38 (Stride 1, just reducing FOV slightly)
        # We stop downsampling here to keep the receptive field focused.
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv3d(initial_filters*4, initial_filters*8, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm3d(initial_filters*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output Layer: 38 -> 37
        # Maps to a 1-channel grid of "Real/Fake" scores
        self.final = nn.Conv3d(initial_filters*8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        # x should be concatenation of [CT, Logits/Mask]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.final(x)
        
        return out

