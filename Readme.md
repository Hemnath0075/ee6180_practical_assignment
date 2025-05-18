# Image-to-Image Translation Training and Inference

This project allows training and testing various image-to-image translation with  Pix2Pix model.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt  # Install necessary packages
```

## Training the Model

To train a model, run the following command:

```bash
python train.py \
  --dataset_path ./datasets/facades \
  --output_dir ./outputs \
  --epochs 150 \
  --batch_size 1 \
  --image_size 256 \
  --lambda_l1 100.0
```

#### Example to train the model

```
python3 train.py  --dataset_path datasets/cityscapes/train --output_dir results/ --epochs 30 --batch_size 1
```


### Training Options:

* `--dataset_path` : Path to the dataset folder (should include train and test subdirectories)
* `--output_dir` : Directory to store outputs and saved models  
* `--epochs` : Total number of training epochs  
* `--batch_size` : Batch size for training  
* `--image_size` : Target height and width of input/output images  
* `--lambda_l1` : L1 loss weight  


## Existing Experiments

This section documents the experiments conducted in the `existing_experiments.ipynb` notebook. Each experiment explores different aspects of the Pix2Pix model performance, dataset characteristics, and loss function impact.

### Experiment 1: FCN Scores on Cityscapes, Facades, and Maps Datasets

**Setup:**  
FCN (Fully Convolutional Network) scores were computed for Pix2Pix models trained on Cityscapes, Facades, and Maps datasets using different loss methods (L1, cGAN, L1 + cGAN). The objective was to analyze how the choice of loss influences the segmentation quality.

### Experiment 2: Training on Maps Dataset

**Goal:**  
To evaluate Pix2Pix's output on the Maps dataset, where the semantic structure differs from datasets like Facades and Cityscapes.

**Setup:**  
The model was trained for 20 epochs to convert aerial (satellite) images into map layouts. This explores the generalization of Pix2Pix to highly distinct domains.

### Experiment 3: Evaluation with Different Losses

**Goal:**  
To understand the impact of various loss functions on the output quality.

**Setup:**  
Inference was performed using models trained with L1 loss only, cGAN loss only, and a combination of both. This experiment shows how each loss function contributes to image realism and structural fidelity.

### Experiment 4: Reproduction on Multiple Domains

**Goal:**  
To reproduce Pix2Pix results across different datasets and validate its performance consistency.

**Setup:**  
The model was trained and evaluated on Cityscapes, Maps, and Facades datasets to test the reproducibility of Pix2Pix under varying domain-specific settings.


## New Experiments

These experiments explore cross-domain performance, perceptual quality evaluation, and robustness of the Pix2Pix model.

### Experiment 1: Cross-Domain Evaluation

**Goal:**  
Test the model’s generalization by applying it on unseen domain pairs. For example, train on Facades (labels → photo) and test on Cityscapes (semantic labels → photo) to evaluate output quality on datasets with different semantic labels.

**Setup:**  
The trained model was evaluated on different datasets to check how well it generates images across domains with varied semantic distributions.

- Figure 10: Cross Domain Evaluation

### Experiment 2: LPIPS Score Evaluation

**Goal:**  
Evaluate model performance using the Learned Perceptual Image Patch Similarity (LPIPS) metric.

**Setup:**  
The model was trained on the Facades dataset and tested on unseen images to compute LPIPS scores using an AlexNet-based evaluation metric. The LPIPS score between real and predicted images was measured.

**Results:**  
The model achieved an LPIPS score of 1.0112 after training for 40,000 steps. It is expected that increasing training steps to 80,000 could improve the LPIPS score to around 0.5, indicating better perceptual similarity.

### Experiment 3: Robustness to Noisy Inputs

**Goal:**  
Assess the model’s robustness by introducing salt-and-pepper noise to semantic label input images and comparing output quality.

**Setup:**  
The model was trained on the Facades dataset. Salt-and-pepper noise was added to the input by randomly setting pixel values to 0 (pepper) or 1 (salt) at specified ratios, using tensor operations to modify image tensors.

**Results:**  
LPIPS scores for clean inputs and noisy inputs were 1.0174 and 1.0270 respectively, showing minimal degradation. This suggests that the model maintains robustness even with noisy semantic label inputs.


## References

https://github.com/phillipi/pix2pix


---
