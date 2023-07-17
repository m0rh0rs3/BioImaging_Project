# BioImaging_Project NL1
## Description

This project was the final assignment for the "Biomedical Computer Vision Course" at NecstCamp. It is based on the [Kits19 Challenge](https://github.com/neheller/kits19).

## Dataset Exploration and Analysis
First, I analyzed the dataset structure and the type of .nii files using MANGO (a CT visualizer). During this analysis, I noticed variations among different cases: not all cases have the same number of slices.

## Data Preparation and Neural Network Format Conversion
I implemented a series of functions for converting .nii files to a format compatible with the neural network (see keras_array_of_cases()), also including directory management.

## Initial Data Cleaning for Anomalies
While directly inspecting the CT scans, I noticed that some of them presented anomalies such as high contrasts and halos. To address this, I performed an initial cleaning using _normalize().

## Implementation of One-Hot Encoding for Multiclass Segmentation
I realized that for multiclass segmentation, the preferable encoding to use is one-hot encoding. Therefore, I implemented this type of conversion.

## Development of U-Net Architecture and Hyperparameter Settings
I modeled a U-Net following the standard model, thus developing the general structure of the chosen U-Net, together with its architecture and hyper-parameter settings.

(Reference: 219 - Understanding U-Net architecture and building it from scratch -DigitalSreeni )

## Uniform Cropping of Images for Training
I decided to use ten images per case, uniformly cropped to include only the central part, discarding 1/5 from the top and 1/5 from the bottom, as those regions generally contain only background in the mask.

## Visualization and Verification of Predictions
To visually verify the predictions, I implemented sub-functions using matplotlib that display the test patient's CT scan, the real mask, and the predicted mask simultaneously.

![img_1](https://github.com/m0rh0rs3/BioImaging_Project/blob/main/Output/out1.png)
![img_2](https://github.com/m0rh0rs3/BioImaging_Project/blob/main/Output/out2.png)



NB: As of today, the network still struggles to correctly segment the kidneys. I suspect this issue is due to limited data availability .
