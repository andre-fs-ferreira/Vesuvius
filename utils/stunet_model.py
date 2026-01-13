import torch
import torch.nn as nn

class BasicResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, stride=self.stride, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True, track_running_stats=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True, track_running_stats=False)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv3 = None
        if self.stride != (1,1,1) or in_c != out_c:
            self.conv3 = nn.Conv3d(in_c, out_c, kernel_size=1, stride=self.stride, bias=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.conv3 is not None:
            residual = self.conv3(x)
        
        out += residual
        
        # --- FIX: Activation MUST happen after addition ---
        out = self.act2(out) 
        
        return out

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=(2,2,2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)

class STUNetReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.conv_blocks_context = nn.ModuleList([
            nn.Sequential(BasicResBlock(1, 64, stride=1), BasicResBlock(64, 64)),
            nn.Sequential(BasicResBlock(64, 128, stride=2), BasicResBlock(128, 128)),
            nn.Sequential(BasicResBlock(128, 256, stride=2), BasicResBlock(256, 256)),
            nn.Sequential(BasicResBlock(256, 512, stride=2), BasicResBlock(512, 512)),
            nn.Sequential(BasicResBlock(512, 1024, stride=2), BasicResBlock(1024, 1024)),
            nn.Sequential(BasicResBlock(1024, 1024, stride=(1,1,2)), BasicResBlock(1024, 1024)),
        ])

        # --- DECODER ---
        self.upsample_layers = nn.ModuleList([
            Upsample_Layer_nearest(1024, 1024, scale_factor=(1,1,2)), 
            Upsample_Layer_nearest(1024, 512, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(512, 256, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(256, 128, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(128, 64, scale_factor=(2,2,2)),
        ])
        
        self.conv_blocks_localization = nn.ModuleList([
            nn.Sequential(BasicResBlock(2048, 1024), BasicResBlock(1024, 1024)),
            nn.Sequential(BasicResBlock(1024, 512), BasicResBlock(512, 512)),
            nn.Sequential(BasicResBlock(512, 256), BasicResBlock(256, 256)),
            nn.Sequential(BasicResBlock(256, 128), BasicResBlock(128, 128)),
            nn.Sequential(BasicResBlock(128, 64), BasicResBlock(64, 64)),
        ])

        # --- OUTPUT ---
        self.seg_outputs = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv3d(64, 1, kernel_size=1, stride=1) 
        ])

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.conv_blocks_context):
            x = block(x)
            if i < len(self.conv_blocks_context) - 1:
                skips.append(x)
                
        for i in range(len(self.upsample_layers)):
            x = self.upsample_layers[i](x)
            skip = skips[-(i+1)]
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[i](x)

        return self.seg_outputs[4](x)


class STUNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.conv_blocks_context = nn.ModuleList([
            nn.Sequential(BasicResBlock(1, 64, stride=1), BasicResBlock(64, 64)),
            nn.Sequential(BasicResBlock(64, 128, stride=2), BasicResBlock(128, 128)),
            nn.Sequential(BasicResBlock(128, 256, stride=2), BasicResBlock(256, 256)),
            nn.Sequential(BasicResBlock(256, 512, stride=2), BasicResBlock(512, 512)),
            nn.Sequential(BasicResBlock(512, 1024, stride=2), BasicResBlock(1024, 1024)),
            nn.Sequential(BasicResBlock(1024, 1024, stride=(1,1,2)), BasicResBlock(1024, 1024)),
        ])

        # --- DECODER ---
        self.upsample_layers = nn.ModuleList([
            Upsample_Layer_nearest(1024, 1024, scale_factor=(1,1,2)), 
            Upsample_Layer_nearest(1024, 512, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(512, 256, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(256, 128, scale_factor=(2,2,2)),
            Upsample_Layer_nearest(128, 64, scale_factor=(2,2,2)),
        ])
        
        self.conv_blocks_localization = nn.ModuleList([
            nn.Sequential(BasicResBlock(2048, 1024), BasicResBlock(1024, 1024)),
            nn.Sequential(BasicResBlock(1024, 512), BasicResBlock(512, 512)),
            nn.Sequential(BasicResBlock(512, 256), BasicResBlock(256, 256)),
            nn.Sequential(BasicResBlock(256, 128), BasicResBlock(128, 128)),
            nn.Sequential(BasicResBlock(128, 64), BasicResBlock(64, 64)),
        ])

        # --- OUTPUT ---
        self.seg_outputs = nn.ModuleList([
            #nn.Conv3d(1024, 1, kernel_size=1, stride=1), 
            #nn.Conv3d(512, 1, kernel_size=1, stride=1), 
            None, # ignoring the deepest two outputs for memory efficiency
            None, # not using these outputs for deep supervision
            nn.Conv3d(256, 1, kernel_size=1, stride=1), 
            nn.Conv3d(128, 1, kernel_size=1, stride=1),
            nn.Conv3d(64, 1, kernel_size=1, stride=1) 
        ])

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.conv_blocks_context):
            x = block(x)
            if i < len(self.conv_blocks_context) - 1:
                skips.append(x)
        
        deep_supervision_preds = []
        for i in range(len(self.upsample_layers)):
            x = self.upsample_layers[i](x)
            skip = skips[-(i+1)]
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[i](x)
            if i>=2:  # Start collecting predictions from the 3rd decoder block
                deep_supervision_preds.append(self.seg_outputs[i](x))

        if self.training:
            # Return all scales for Deep Supervision Loss
            # index 0 = low res, index -1 = high res
            return deep_supervision_preds
        else:
            # Return only the final high-res prediction for validation/inference
            return deep_supervision_preds[-1]

if __name__ == "__main__":
    model = STUNetSegmentation()
    model.train()
    # Test
    x = torch.randn(1, 1, 128, 128, 128)
    deep_supervision_preds = model(x)
    print(f"Model Built. In: {x.shape}, Out:")
    for pred in deep_supervision_preds:
        print(f"{pred.shape}")
    print(f"Testing evaluation mode")
    model.eval()
    # Test
    x = torch.randn(1, 1, 128, 128, 128)
    deep_supervision_preds = model(x)
    print(f"Model Built. In: {x.shape}, Out:")
    for pred in deep_supervision_preds:
        print(f"{pred.shape}")