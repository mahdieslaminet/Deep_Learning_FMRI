# Autism Spectrum Disorder Diagnosis Using Deep Learning

This repository contains the implementation of a deep learning model for the diagnosis of Autism Spectrum Disorder (ASD) using the ABIDE (Autism Brain Imaging Data Exchange) fMRI dataset. The project explores the use of Convolutional Neural Networks (CNNs) combined with the F-score feature selection method to differentiate between ASD and typically developing (TD) individuals, as well as classifications by gender and task vs. rest states.

## Project Overview

### Abstract
Autism Spectrum Disorder (ASD) presents significant challenges in diagnosis due to its heterogeneity and the high costs associated with neuroimaging. This study leverages deep learning techniques to analyze fMRI data, aiming to enhance diagnostic accuracy and provide insights into the neural underpinnings of ASD. Our approach involves training a CNN on a large, multi-site dataset, achieving promising results in classifying ASD vs. TD, gender, and task vs. rest states.

### Introduction
ASD is a neurodevelopmental disorder characterized by difficulties in social interaction and repetitive behaviors. Despite the potential for early detection, diagnosis often occurs later due to overlapping symptoms with other conditions. This project utilizes machine learning, specifically CNNs, to analyze fMRI data and improve diagnostic accuracy.

### Methods
- **Dataset**: ABIDE fMRI data, encompassing neuroimaging and phenotypic information from multiple international sites.
- **Preprocessing**: Includes motion correction, normalization, and time series extraction using the BASC atlas.
- **Model**: A CNN trained on connectivity matrices derived from the fMRI data, using cross-validation for robust performance assessment.

### Results
The model demonstrated notable classification performance, with AUROCs of 0.6774, 0.7680, and 0.9222 for ASD vs. TD, gender, and task vs. rest classifications, respectively. Class Activation Maps (CAMs) highlighted significant brain regions, including the temporal and cerebellar areas, potentially elucidating ASD pathology.

### Conclusion
This study underscores the potential of deep learning in ASD diagnosis, offering both diagnostic accuracy and interpretability. The findings suggest shifts in network topology among ASD individuals, pointing to a transition from small-world to random network architectures in the brain.

## Resources

- [Code Video of the Main Article](https://drive.google.com/file/d/1dYn2Rah2hnB5Rux5iL3cWJiYpe362B4D/view?usp=sharing)
- [Video Explaining the Main Article](https://drive.google.com/file/d/1o1buPSPWcbRwMhmvIjydy_cE2ocmR4aH/view?usp=sharing)
  - [Part 2](https://drive.google.com/file/d/1E3ueKz3La5RKwWgz_hMVs-mRWVvSmEYD/view?usp=sharing)
  - [Part 3](https://drive.google.com/file/d/1rwfiV7cbBLoKWSZiWqHWUq6BVMjVAa-r/view?usp=sharing)
- [Innovation Article Video Code](https://drive.google.com/file/d/1a0lesHV6vGTe56Hj80ftZatPQA6vcXip/view?usp=sharing)
  - [Part 2](https://drive.google.com/file/d/1qdqc9ml4HWRo6D-4Bl2ory1-Y1aYAy83/view?usp=sharing)

## How to Use

1. Clone the repository and navigate to the project directory.
2. Follow the instructions in the `requirements.txt` to set up your environment.
3. Run the provided scripts to preprocess the data, train the model, and evaluate the results.
