import json
import sys
sys.path.append("../utils")
from main_train_class import main_train_STU_Net

# Warning Suppressions (Must be at the top)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.numcodecs._codecs")

if __name__=="__main__":
    CONFIG_FILE = "../configs/all_data_finetune.json"
    with open(CONFIG_FILE, "r") as f:
        second_step_config = json.load(f)
    trainer = main_train_STU_Net(second_step_config)
    trainer.train_loop()