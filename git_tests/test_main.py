import unittest
import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to path so we can import your main code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_train_class import main_train_STU_Net

# --- MOCK MODEL ---
# We create a simple model to mimic your STUNet structure
class MockSTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulated "Pretrained/Loaded" layers
        self.encoder = nn.Linear(10, 10)
        self.decoder = nn.Linear(10, 10)
        
        # Simulated "New Head/Missing" layers (e.g., seg_outputs)
        self.seg_outputs = nn.ModuleList([
            nn.Conv2d(10, 2, kernel_size=1),
            nn.Conv2d(10, 2, kernel_size=1)
        ])

class TestFreezeWeights(unittest.TestCase):

    def setUp(self):
        """
        Prepare the trainer instance without running its heavy __init__
        """
        # 1. Create a "blank" instance of your class. 
        # using __new__ bypasses __init__, so we don't need config/wandb/gpu here.
        self.trainer = main_train_STU_Net.__new__(main_train_STU_Net)
        
        # 2. Inject a mock model
        self.trainer.model = MockSTUNet()
        
        # 3. Simulate the state produced by _build_model
        # We define which keys are "missing" (the head).
        # Note: We use the exact spelling from your code 'mising_weights'
        self.trainer.mising_weights = [
            'seg_outputs.0.weight', 
            'seg_outputs.0.bias',
            'seg_outputs.1.weight', 
            'seg_outputs.1.bias'
        ]

    def test_freeze_weights_functionality(self):
        """
        Test if _freeze_weights correctly freezes the backbone and keeps the head open.
        """
        print("\nTesting _freeze_weights logic...")

        # 1. Run the function under test
        self.trainer._freeze_weights()

        # 2. Verification Loop
        for name, param in self.trainer.model.named_parameters():
            
            # CASE A: The parameter is in the "missing" list (The Head)
            if name in self.trainer.mising_weights:
                self.assertTrue(
                    param.requires_grad, 
                    f"❌ Error: New Head layer '{name}' should be TRAINABLE, but it is frozen."
                )
                
            # CASE B: The parameter is NOT in the list (The Backbone)
            else:
                self.assertFalse(
                    param.requires_grad, 
                    f"❌ Error: Loaded Backbone layer '{name}' should be FROZEN, but it is trainable."
                )

        print("✅ Success: Backbone is frozen, Head (missing keys) is trainable.")

if __name__ == '__main__':
    unittest.main()