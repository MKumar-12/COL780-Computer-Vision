# COL780 : Computer Vision

This repository contains the codebase for assignments completed during the course structure of CV (WINTER-24). 
The assignments were completed as part of the COL780 course at IIT Delhi under the supervision of Prof. Chetan Arora.

## Assignment 1 - Micro-Suturing Analysis

**Objective:** Analyze a subset of a large micro-suturing dataset by calculating various parameters from sample images and output the results in a CSV file, without using built-in OpenCV functions.

### Image Analysis

- **Iterate through all images in the given directory.**
- **Calculate and record the following parameters for each image:**
  - **Image Name:** Name of the image file.
  - **Number of Sutures:** Count of sutures present in the image.
  - **Mean Inter-Suture Spacing:** Average distance between consecutive sutures.
  - **Variance of Inter-Suture Spacing:** Variance of the distances between consecutive sutures.
  - **Mean Suture Angle with Respect to the X-Axis:** Average angle of sutures relative to the x-axis.
  - **Variance of Suture Angle with Respect to the X-Axis:** Variance of the angles of sutures relative to the x-axis.

## Assignment 2 - Image Panorama Generation

**Objective:** Create seamless panoramic images from a set of input images or video frames by aligning and blending them.

### Approach

- **Input Handling:**
  - **Images:** Read and order images from the specified directory.
  - **Video:** Extract frames at regular intervals for panorama creation.
- **Cylindrical Projection:**
  - Correct perspective distortion by projecting images onto a cylindrical surface.
- **Feature Matching:**
  - Use SIFT to detect keypoints and match features between images.
- **Homography Estimation:**
  - Estimate the transformation matrix using RANSAC to align images.
- **Stitching:**
  - Warp and blend images using the homography matrix to create a smooth panorama.
- **Artifact Reduction:**
  - Refine homography and blending to minimize misalignments and distortions.

## Assignment 3 - Hand Gesture Detection

**Objective:** Classify hand images into open or closed hand gestures using a binary classifier like Support Vector Machine (SVM).

### Approach

1. **Preprocessing:**
   - Prepare dataset images by identifying Regions of Interest (ROI) containing hand gestures.
2. **Feature Extraction:**
   - Extract Histogram of Oriented Gradients (HOG) descriptors to represent hand gestures effectively.
3. **Training:**
   - Train an SVM model using HOG descriptors to classify hand images as either open or closed gestures.

## Assignment 4 - Breast Cancer Detection

**Objective:** Classify malignant patches from given test images into malignant or benign classes using DNN architectures such as Faster R-CNN.

### Approach

1. **Dataset Structure Validation:**
   - Verify the structural integrity of the mammogram dataset, ensuring the presence of 'images' and 'labels' subfolders.
2. **Annotation Parsing:**
   - Parse YOLO format annotations from text files to identify bounding boxes around malignant regions.
3. **Dataset Loading and Organization:**
   - Load the dataset and organize the annotations into a structured dataframe for model training.
