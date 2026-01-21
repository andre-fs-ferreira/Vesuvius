import sys
sys.path.append("../utils")
from infer_class import VesuviusInferer

import os
import json

# 🔧 Configuration Loading
CONFIG_FILE = '../configs/save_logits.json'

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Configuration file {CONFIG_FILE} not found!")
    sys.exit(1)

with open(CONFIG_FILE, "r") as f:
    config_content = json.load(f)

# 🧪 Initialize Inference Object
infer_object = VesuviusInferer(config_content)

print(f"🚀 Starting inference on dataset (save logits!): {config_content['dataset_path_imgs']}")
print("📋 Using the following configuration:")
for key, value in config_content.items():
    print(f"  🔹 {key}: {value}")

print("=" * 50)
# 🏃 Run Inference
infer_object.dataset_inference_save_logits(
    dataset_path = config_content["dataset_path_imgs"], 
    pred_save_dir = config_content["pred_save_dir"]
)
print("=" * 50)
