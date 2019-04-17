# Anomaly-Detection
Anomaly detection in Chest X-Ray scans (Medical Imaging)

## Problem Understanding
Neural networks have revolutionised image processing in several different domains. Among these is the field of medical imaging. We will be building a classifier on Chest X-Ray images to detect Anomalies.
The objective of this exercise is to identify images where an "effusion" is present. This is a classification problem, where we will be dealing with two classes - 'effusion' and 'nofinding'. Here, the latter represents a "normal" X-ray image.
This same methodology can be used to spot various other illnesses that can be detected via a chest x-ray. For the scope of this demonstration, we will specifically deal with "effusion".

## Domain Understanding
Chest radiography (chest X-ray or CXR) is an economical and easy-to-use medical imaging and diagnostic technique. The technique is the most commonly used diagnostic tool in medical practice and has an important role in the diagnosis of the lung disease. Well-trained radiologists use chest X-rays to detect illnesses, such as pneumonia, tuberculosis, interstitial lung disease, and early lung cancer. 

## Data Understanding
Application of deep learning to medical images has become easier due to the availability of open-source images. We will use X-Ray images from https://www.kaggle.com/nih-chest-xrays

We will be working with two classes - 'effusion' and 'nofinding'.

Our data is in the form of grayscale (black and white) images of chest x-rays. To perform our classification task effectively, we need to perform some pre-processing of the data.

#### Data Preparation: 
- All our images were of the same resolution.
- Placed the images in two different folders - 'effusion' and 'nofinding'. 

#### Data Pre-Processing: Augmentation
- Used ImageDataGenerator from Keras Library
- Used different ways to augment - translation, shifting, scaling
- Vertical flip needs to be set as 'False'. This is because CXR images have a natural orientation - up to down. 
- We should not do a centre crop for CXR images, as the anomaly can be in an area outside the cropped portion of the image.

#### Data Pre-Processing: Normalisation
Since the CXR images are not "natural images", we do not use the "divide by 255" strategy. Instead, we take the max-min approach to normalisation. Since we do not know for sure that the range of each pixel is 0-255, we normalise using the min-max values.

### Network Building
<b>Architecture</b>: We used a ResNet together with a decaying learning rate and a weighted cross-entropy loss (to account for class imbalance). We used AUC as the metric. 

<b>Metrics</b>: While choosing a metric for medical images with a prevalence problem, we pick recall over precision. We don't want to miss out on any cases of effusion. In any case, working with AUC and a manual threshold is the best option.

<b>Weighted Cross-entropy</b>: This loss is used when the error in one direction is costlier than the other, for example, it is much more undesirable to diagnose 'effusion' as 'normal' than the other way around. This is done by assigning 'higher weights' to the errors in certain classes.











