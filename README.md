# Experiemnts for the Vesuvius Challenge



# Training ideas
1. Prepare a clean dataset with only 0 and 1 labels.
## Baseline
1. Train the nnUNet on the clean dataset.

## Pre-training
1. Train in an unsupervised matter with all data, without labels.
2. Fine-tune only on the regions with label 0 and 1.