import json
import sys
sys.path.append("/mounts/disk4_tiago_e_andre/vesuvius/Vesuvius/utils")
from pre_train_class import pre_training_STU_Net

# Warning Suppressions (Must be at the top)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.numcodecs._codecs")

if __name__=="__main__":
    CONFIG_FILE = "../configs/pre_training.json"
    with open(CONFIG_FILE, "r") as f:
        pretrain_config = json.load(f)
    trainer = pre_training_STU_Net(pretrain_config)
    trainer.train_loop()