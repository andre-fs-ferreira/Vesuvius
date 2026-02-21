# Experiements for the Vesuvius Challenge

* Use of the model STU-Net large (https://github.com/openmedlab/STU-Net?tab=readme-ov-file) converted to binary output.
* Pre-trained using the 5 entire volumes available in (https://dl.ash2txt.org/full-scrolls/)

  * For this, we downloaded every case, one by one, removed the background and converted to unit8 to save space.
  * We ended up having around 1200 GB of training data.
  * We use self-supervised learning with mean absolute error and random dropout (we cut holes in the volumes and train the model to reconstruct)
* Performed regular training using the dataset from the challenge:

  * Added deep supervision to the network.
  * Label 2 was ignored in loss computation:
    * First we used BCE and DSC (100 epochs to stabelize with fased warmup, unfrezing layer by layer from the output to the input. Agressive data agumentation)
    * Then Focal and Tversky (200 epochs, agressive data augmentation).
    * Finaly, BCE and Tversky (160 epochs).
* Then we trained fine-tuning with all data.

  * with the loss function 0.2BCE + 0.4SoftDice + 0.4clDice
* Inference:

  * TTA only in the x and y axis (4 inferences per case)
  * Patch size of 128x128x128
  * Post-processing: th_low=0.45 | th_high=0.75 with smaller than 1000 voxels structures removed and binary dilation + binary erosion 1 iteration each.

**Results in the challenge: 0.535**

**Results on local test set**: 0.569314 (post-processing reduced the surface dice from 0.83 and increased the topo score from 0.17)

* topo_score=0.284945
* surface_dice=0.828291
* voi_score=0.554081
* voi_split=1.291170
* voi_merge=1.424576

# TODO

* [ ] Sharing the weights of each step
* [ ] Sharing the data used for pre-training
