import sys
sys.path.append("../utils")
from infer_class import VesuviusInferer
from compute_metrics_obj import VesuviusMetric

import os
import json

# 🔧 Configuration Loading
CONFIG_FILE = '../configs/infer.json'

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Configuration file {CONFIG_FILE} not found!")
    sys.exit(1)

with open(CONFIG_FILE, "r") as f:
    config_content = json.load(f)

# 🧪 Initialize Inference Object
infer_object = VesuviusInferer(config_content)

print(f"🚀 Starting inference on dataset: {config_content['dataset_path']}")
print("📋 Using the following configuration:")
for key, value in config_content.items():
    print(f"  🔹 {key}: {value}")

print("=" * 50)
# 🏃 Run Inference
infer_object.dataset_inference(
    dataset_path = config_content["dataset_path"], 
    pred_save_dir = config_content["pred_save_dir"]
)
print("=" * 50)

# 📊 Evaluation Phase
print("🧐 Starting evaluation using VesuviusMetric...")

test_metric_obj = VesuviusMetric(
    solution_path=f"{config_content['dataset_path_gt']}/test_labels_df.csv",
    submission_path=f"{config_content['pred_save_dir']}/test_pred_df.csv",
    output_file=f"{config_content['pred_save_dir']}/detailed_scores_obj.csv"
)

test_metric_obj._run()
print("🎉 Evaluation completed. Results saved! 🏆")