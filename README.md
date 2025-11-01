# Vehicle Detection Using YOLO based CNN with TensorFlow

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Logs](#training-logs)
- [Results](#results)
- [Observations](#observations)
- [Limitations](#limitations)

## Project Overview

This project implements a simplified object detection system to identify and localize cars in images. The model uses a custom convolutional neural network architecture inspired by YOLO (You Only Look Once) for efficient single-car detection.

**Key Features:**
- Custom CNN architecture with 22.2M parameters.
- Bounding box regression for car localization.
- Trained on 355 annotated images.
- Achieves 13.06% precision at IoU threshold 0.5.

## Dataset

**Training Data:**
- **Total Images:** 1,001 files in `training_images/` folder
- **Annotated Images:** 355 images with bounding box labels
- **Total Bounding Boxes:** 559 (average 1.6 cars per annotated image)
- **Annotations:** Provided in `train_solution_bounding_boxes.csv`

**Note:** The dataset is not included in this repository due to licensing considerations. 
Download: https://www.kaggle.com/datasets/sshikamaru/car-object-detection

**Test Data:**
- **Total Images:** 175 files in `testing_images/` folder
- **Annotations:** Not available (placeholder values in `sample_submission.csv`)

**Data Format:**
- Image size: Variable (resized to 416×416 for model input).
- Bounding box format: (xmin, ymin, xmax, ymax) in pixel coordinates.
- Normalization: Coordinates normalized by original image dimensions.

**Why Test Images Weren't Used:**
The `sample_submission.csv` file contains placeholder values (all "0.0 0.0 1.0 1.0"), not real ground truth annotations. Without actual labels, evaluation metrics like IoU, precision, and recall cannot be calculated. So evaluation was performed on training images instead.

## Model Architecture

**Model Specifications:**
- Total Parameters: 22,175,525
- Trainable Parameters: 22,175,525
- Model Size: 84.59 MB
- Output: Single bounding box per image

**Training Configuration:**
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 8
- Epochs: 25 (best performance at epoch 15)
- Steps per Epoch: 44

## Training Logs

Complete training outputs are available in [training_log.txt](./training_log.txt), including:
- Loss values for all 25 epochs.
- Checkpoint comparison results.

## Results

### Training Progress

| Epoch | Training Loss | Notes              |
|:------:|:--------------:|--------------------|
| 1  | 0.0362 | Initial training |
| 5  | 0.0060 | Rapid improvement |
| 10 | 0.0033 | Continued learning |
| 15 | 0.0025 | Best performance |
| 20 | 0.0016 | Overfitting begins |
| 25 | 0.00094 | Severe overfitting |

### Evaluation Metrics (Epoch 15, IoU > 0.5)

| Metric | Value |
|:--------|:------|
| **True Positives** | 73 / 559 |
| **False Positives** | 486 / 559 |
| **Precision** | 13.06% |
| **Recall** | 13.06% |

### Model Analysis — Checkpoint Comparison

| Epoch | True Positives | Precision |
|:------:|:---------------:|:----------:|
| 10 | 4 | 0.72% |
| 13 | 41 | 7.33% |
| 15 | 73 | 13.06% |
| 17 | 6 | 1.07% |
| 25 | 22 | 3.94% |

## Observations

- **Model performance peaked at Epoch 15**, after which it started to degrade due to **overfitting**, even though the **training loss kept decreasing**.  
- **Sample predictions** show that the model can **detect cars confidently (confidence > 0.99)**, but it **struggles to place the bounding box accurately** in complex scenes with **multiple or partly hidden cars**.
  
## Limitations

### 1. Architecture Simplicity
- Uses a **simplified CNN** instead of full YOLO grid-based detection. 
- **Single bounding box output** (cannot detect multiple cars simultaneously).

### 2. Dataset Size
- Only **355 annotated images** for training.
- Limited diversity in **car types** and **scenarios**.

### 3. Evaluation Method
- Evaluated on **training data** only (no separate validation set).
- **Test set** lacks ground-truth annotations.

### 4. Performance
- **13.06% precision** indicates room for improvement. 
- Model trained on **first car only** in multi-car images.

### 5. Multi-Car Scenes
- Model detects cars but may not select the **annotated car**. 
- Actual detection rate is likely **higher than metrics suggest**.
