# Experiments for the Vesuvius Challenge

## Downloading the data for pre-training:
  * Use the notebook in `notebooks/5_Download_tif_files.ipynb` to download the data. Two approaches are in there. Second one recommended.
  * We downloaded every case, one by one, removed the background and converted to uint8 to save space.
  * We ended up having around 1200 GB of training data.
  * Note: a large disk is necessary for this. I have all data pre-processed in a google drive. I can share this volumes.

## Pre-training approach: 
  * Use of the model STU-Net large (https://github.com/openmedlab/STU-Net?tab=readme-ov-file) converted to one channel output.
  * Use the notebook `notebooks/6_STU-Model_Loading.ipynb` for this conversion.
  * Pre-trained using the 5 entire volumes available in (https://dl.ash2txt.org/full-scrolls/), downloaded in the step before.
  * We use self-supervised learning with mean absolute error and random dropout (we cut holes in the volumes and train the model to reconstruct). This paper and this figure might help to understand better the approach: [https://www.nature.com/articles/s41598-025-11688-2/figures/1](https://www.nature.com/articles/s41598-025-11688-2/figures/1).
  * The script in `src/pre_train.py` was used for this pre-training. The configuration is available in configs/pre_training.json.
  * For changing the model used for pre-training, you must change the function `_build_model` in `utils/pre_train_class.py`. Perhaps, some newer architecture might work better.
  * The notebook `7_STU-Pre-training.ipynb` can be used for some testing, but the python script file should be used instead.

## Main training (challenge segmentation):
  * For the challenge, and to mimic the nnUNet we first:
    * Added deep supervision to the network.
  * The training was divided in several steps in order to curate the predictions:
    * Label 2 was ignored in loss computation.
    * First we used BCE and DSC (100 epochs to stabilize with phased warmup, unfreezing layer by layer from the output to the input. Aggressive data augmentation).
    * Then Focal and Tversky (200 epochs, aggressive data augmentation).
    * Finally, BCE and Tversky (160 epochs, soft data augmentation).
  * The data augmentations strategies can be further analised in the file `utils/main_train_class.py`.

  * For training each step, you should run the script `src/main_train.py` changing the configs in `configs/main_second_step.json`.
  * I have some implementations of loss functions (DSC, BCE, Tversky, Focal, CLDICE, BCESoftDiceclDiceLoss, Antibridge), you can add other in `_set_train_criterion` in `utils/main_train_class.py`.
  * Note: Antibridge didn't work very well. I would ignore it. 

## Fine-tuning (using all data):
  * For the step before, 10% of the data was hold out for testing.
  * For the final solution, this data should be included, so we use all training data to fine-tune the model a little bit more, with some improvements (around 4%).
  * Therefore, we trained fine-tuning with all data (10 epochs only), suing the script `src/finetune_train.py` and the config file `configs/all_data_finetune.json`.json
  * with the loss function 0.2BCE + 0.4SoftDice + 0.4clDice.

## Inference:
  * TTA only in the x and y axis (4 inferences per case)
  * Patch size of 128x128x128
  * Post-processing: th_low=0.45 | th_high=0.75 with smaller than 1000 voxels structures removed and binary dilation + binary erosion 1 iteration each.
  * Use the python script `src/infer_n_test.py` for inference and testing using the platform metrics (I installed from here https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo/input). 

## Results:
  * Before fine-tuning:
    * In the challenge: 0.535
    * On local test set: 0.569314 (post-processing reduced the surface dice from 0.83 and increased the topo score from 0.17)
      * topo_score=0.284945
      * surface_dice=0.828291
      * voi_score=0.554081
      * voi_split=1.291170
      * voi_merge=1.424576
  * After fine-tuning:
    * In the challenge: 0.561
    * On local test set (biased): 0.599561					
      * topo_score=0.333240
      * surface_dice=0.864176
      * voi_score=0.563221
      * voi_split=1.286570
      * voi_merge=1.337359

# TODO
* [ ] Sharing the weights of each step
* [ ] Sharing the data used for pre-training
