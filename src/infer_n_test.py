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


# One or more experiments are expected to be tested in this script
# checkpoint_path should be a list
# Then tests will be done with:
    # default (th=0.5, no_TTA, overlap=0.5)
    # With TTA and not TTA:
        # With overlaps 0.5, 0.6, 0.7, 0.8:
            # th=0.4
            # th=0.5
            # th=0.6
            # th=0.7
            # th=0.75

tta_list = config_content['TTA_list']
overlap_list = config_content['overlap_list']
th_list = config_content['TH_list']

for checkpoint_path in config_content['checkpoint_paths']:
    print(f"Using checkpoint weights {checkpoint_path}")
    config_content['checkpoint_path'] = checkpoint_path

    model_name_str = checkpoint_path.split('/')[-2]
    epoch = checkpoint_path.split('/')[-1].split('epoch_')[-1].split('.pth')[0]
    
    for tta in tta_list:
        config_content['TTA'] = tta # tta and no tta
        if tta:
            tta_str = "TTA"
        else:
            tta_str = "no_TTA"
        for overlap in overlap_list:
            config_content['infer_overlap'] = overlap # overlap between windows
            for th in th_list:
                config_content['TH'] = th # threshold of prob to become 1 in segmentation
                config_content["pred_save_dir"] = os.path.join(config_content["root_pred_save_dir"], f"{model_name_str}_epoch_{epoch}", str(tta_str), f"overlap_{overlap}", f"th_{th}")
                csv_name = f"{model_name_str}_epoch_{epoch}_{tta_str}_overlap_{overlap}_th_{th}.csv"

                print(f"🚀 Starting inference on dataset: {config_content['dataset_path_imgs']}")
                print("📋 Using the following configuration:")
                for key, value in config_content.items():
                    print(f"  🔹 {key}: {value}")
                
                # 🧪 Initialize Inference Object
                infer_object = VesuviusInferer(config_content)

                print("=" * 50)
                # 🏃 Run Inference
                infer_object.dataset_inference(
                    dataset_path = config_content["dataset_path_imgs"], 
                    pred_save_dir = config_content["pred_save_dir"]
                )
                print("=" * 50)

                # 📊 Evaluation Phase
                print("🧐 Starting evaluation using VesuviusMetric...")
                if os.path.exists(f"{config_content['pred_save_dir']}/{csv_name}"):
                    pass
                else:
                    test_metric_obj = VesuviusMetric(
                        solution_path=f"{os.path.join(config_content['dataset_path_gt'], os.path.basename(os.path.normpath(config_content['dataset_path_gt'])) + '_df.csv')}",
                        submission_path=f"{os.path.join(config_content['pred_save_dir'], os.path.basename(os.path.normpath(config_content['pred_save_dir'])) + '_df.csv')}",
                        output_file=f"{config_content['pred_save_dir']}/{csv_name}"
                    )

                    test_metric_obj._run()
                print("🎉 Evaluation completed. Results saved! 🏆")