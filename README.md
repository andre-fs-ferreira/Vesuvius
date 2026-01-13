# Experiemnts for the Vesuvius Challenge

# TODO:

## Pre-training
[] Pre-training on the 5 full resolution scrolls
    [x] Training (up to 1000 epohcs)
    [] DONE

## Fine-tuning
[] Fine tune:
    [] Fine-tune only on the regions with label 0 and 1.
        [] Train for 100 epochs (check overfitting)
            [] With Tversky and BCE (1:1)
        [] Aftwerwards, train using the other metrics. Perform some testing.


    [] In case of time...
        [] Loss functions to try:
            [] DSC (baseline)
            [] BE + DSC
            [] Tversky α = 0.7 $\beta=0.3$
            [] 0.5*Focal + Tversky α = 0.7 $\beta=0.3$
            [] border loss (to implement!)

## Baselines
[] Train the baseline without pre-training
    [] Build the training cycle

[] Train the nnUNet on the clean dataset


