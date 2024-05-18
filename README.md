This is the repository linked to the paper "Robust Conformal Volume Estimation in 3D Medical Images", early accepted at MICCAI 2024.

The core concept of the paper is to propose an improvement of the standard conformal prediction paradigm to handle covariate shifts in medical image segmentation. 
To achieve this result we rely on compressed latent representations extracted by a trained segmentation model to estimate the density ratio between the calibration and test distribution.

The code contains an implementation of the multi-head segmentation model (multi_head_model.py --> Triad Net) used to compute the intervals on volumes. 
See https://arxiv.org/abs/2307.15638 for more information on this technique. 

Latent representations are obtained using feature_extractor.py 

Finally the standard Conformal Prediction calibration as well as the proposed weighted version is provided in weighted_cp.py 

