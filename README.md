# ShuffleNet1D
One dimensional ShuffleNet model used for arrhythmia classification

Structure of project files

MATLAB: To prepare training and testing dataset from the dataset .mat files
* segmentation_MIT_BIH_flexibleSTRIDE.m
* ecg_all.mat
* ann_loc_all.mat
* ann_type_all.mat
* seg_samples_STRIDED_No_Q.mat
* label_samples_STRIDED_No_Q.mat

Python: 
* Train_shufflenet_1D_Focal_loss.py 
* Train_CNN_1D_Focal_loss.py 
* shufflenet_1D.py
* utils_1D.py
* focal_loss.py

