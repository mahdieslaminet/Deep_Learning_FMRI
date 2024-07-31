# Autism Spectrum Disorder Diagnosis Using Deep Learning

This repository contains the implementation of a deep learning model for the diagnosis of Autism Spectrum Disorder (ASD) using the ABIDE (Autism Brain Imaging Data Exchange) fMRI dataset. The project explores the use of Convolutional Neural Networks (CNNs) combined with the F-score feature selection method to differentiate between ASD and typically developing (TD) individuals, as well as classifications by gender and task vs. rest states.

## Project Overview

### Abstract
Autism Spectrum Disorder (ASD) presents significant challenges in diagnosis due to its heterogeneity and the high costs associated with neuroimaging. This study leverages deep learning techniques to analyze fMRI data, aiming to enhance diagnostic accuracy and provide insights into the neural underpinnings of ASD. Our approach involves training a CNN on a large, multi-site dataset, achieving promising results in classifying ASD vs. TD, gender, and task vs. rest states.

### Introduction
ASD is a neurodevelopmental disorder characterized by difficulties in social interaction and repetitive behaviors. Despite the potential for early detection, diagnosis often occurs later due to overlapping symptoms with other conditions. This project utilizes machine learning, specifically CNNs, to analyze fMRI data and improve diagnostic accuracy.ASD is a lifelong neurodevelopmental disorder characterized by challenges in social interaction and communication, along with repetitive behaviors. Despite being identifiable in early childhood, ASD diagnosis often faces delays. Machine learning and deep learning techniques are increasingly used to aid in diagnosing ASD, leveraging large datasets to identify patterns that may not be immediately apparent to human experts.

### Methods
- **Dataset**: ABIDE fMRI data, encompassing neuroimaging and phenotypic information from multiple international sites.
- **Preprocessing**: Includes motion correction, normalization, and time series extraction using the BASC atlas.
- **Model**: A CNN trained on connectivity matrices derived from the fMRI data, using cross-validation for robust performance assessment.
- **Neural Network Model**: A CNN architecture is employed, with a focus on feature extraction from functional connectivity matrices. The model includes layers designed to capture complex patterns in the data, aided by techniques like batch normalization, dropout, and an ensemble approach to enhance accuracy and generalization.


### Key Features
- Deep Learning Architecture: A CNN is used to classify ASD and typically developing (TD) controls, as well as gender and task vs. rest states. The model benefits from a large dataset of 43,858 data points, the most extensive compilation of fMRI connectomic data to date.

- Feature Selection and Analysis: F-score feature selection and network topology analysis reveal changes in brain network architecture in ASD, such as a shift towards a more random network structure. This insight could provide valuable understanding into the pathology of ASD.

- Interpretability: To address the "black box" nature of deep learning models, the project employs class activation maps to identify significant brain connections and analyzes hidden layer activations, focusing on areas like the right caudate nucleus and paracentral sulcus.

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



### Guide Script for ASD Diagnosis with Deep Learning

---

## Introduction

Welcome to the ASD Diagnosis with Deep Learning project! This guide will help you understand the structure of the codebase, how to run the code, and the requirements needed to set up the environment.

### 1. **Setting Up the Environment**

Before running the scripts, ensure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### 2. **Directory Structure**

- `data/`: This folder contains the dataset used in the project. Make sure you have the ABIDE dataset properly formatted and stored here.
- `scripts/`: Contains all the Python scripts for data preprocessing, model training, and evaluation.
- `models/`: Stores trained models and logs for analysis.
- `results/`: Includes output files such as evaluation metrics and visualizations.

### 3. **Key Scripts**

#### `data_preprocessing.py`

- **Purpose**: Prepares the ABIDE fMRI data for training. This includes data augmentation, feature extraction, and splitting the data into training and test sets.
- **Usage**: 
  ```bash
  python scripts/data_preprocessing.py --data_dir data/ --output_dir processed_data/
  ```

#### `train_model.py`

- **Purpose**: Trains the convolutional neural network (CNN) model on the preprocessed data.
- **Key Parameters**:
  - `--data_dir`: Directory containing the processed data.
  - `--epochs`: Number of training epochs.
  - `--batch_size`: Batch size for training.
  - `--output_dir`: Directory to save the trained models and logs.
- **Usage**:
  ```bash
  python scripts/train_model.py --data_dir processed_data/ --epochs 50 --batch_size 32 --output_dir models/
  ```

#### `evaluate_model.py`

- **Purpose**: Evaluates the trained model on the test set and generates performance metrics.
- **Key Parameters**:
  - `--model_dir`: Directory containing the trained model.
  - `--data_dir`: Directory containing the test data.
  - `--output_dir`: Directory to save evaluation results.
- **Usage**:
  ```bash
  python scripts/evaluate_model.py --model_dir models/ --data_dir processed_data/ --output_dir results/
  ```

#### `visualize_results.py`

- **Purpose**: Generates visualizations such as confusion matrices, ROC curves, and class activation maps to interpret the model's performance.
- **Key Parameters**:
  - `--results_dir`: Directory containing evaluation results.
  - `--output_dir`: Directory to save the visualizations.
- **Usage**:
  ```bash
  python scripts/visualize_results.py --results_dir results/ --output_dir visualizations/
  ```

## Requirements

Below is a list of the main Python libraries required for this project:

- `tensorflow` or `torch` (depending on whether you are using TensorFlow or PyTorch)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nibabel` (for handling neuroimaging data)
- `scipy`

Create a `requirements.txt` file to list these dependencies:

```
tensorflow==2.5.0
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
nibabel==3.2.1
scipy==1.7.0
```

## Running the Project

1. **Prepare the Data**: Ensure the ABIDE dataset is in the `data/` directory.
2. **Data Preprocessing**: Run the `data_preprocessing.py` script.
3. **Model Training**: Train the model using the `train_model.py` script.
4. **Model Evaluation**: Evaluate the model using `evaluate_model.py`.
5. **Visualization**: Generate visualizations using `visualize_results.py`.

---

