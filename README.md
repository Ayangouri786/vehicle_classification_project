# Vehicle Classification Project

## Overview

The Vehicle Classification Project is designed to classify images of various vehicles into distinct categories using deep learning techniques. This project involves data analysis, cleaning, preprocessing, model training, evaluation, and exporting the model in ONNX format. This README file provides a comprehensive guide on how to set up, run, and reproduce the results of this project.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Running the Code](#running-the-code)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Report Summary](#report-summary)
6. [Additional Resources](#additional-resources)
7. [Contact Information](#contact-information)

## Project Structure

```
Vehicle_Classification_Project/
│
├── vehicle_dataset/
│   ├── train/
│   └── val/
│
├── verification_images/
│
├── models/
│   └── vehicle_dataset/
│       └── vehicle_test/
│           ├── best_val_loss.pt
│           ├── vehicle_test.onnx
│
├── scripts/
│   ├── data_analysis_and_cleaning.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── export_onnx.py
│   └── verify_model.py
│
├── reports/
│   └── report.pdf
│
├── requirements.txt
├── README.md
└── sample_classes.txt
```

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/vehicle_classification_project.git
   cd vehicle_classification_project
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

Follow these steps to execute the scripts and train your model:

1. **Data Analysis and Cleaning**

   Analyzes the dataset and removes duplicates and irrelevant images.

   ```bash
   python scripts/data_analysis_and_cleaning.py
   ```

2. **Data Preprocessing**

   Preprocesses the data by resizing, normalizing, and loading it into DataLoader.

   ```bash
   python scripts/data_preprocessing.py
   ```

3. **Model Training**

   Trains the model using the ResNet-18 architecture, saves the best model based on validation loss.

   ```bash
   python scripts/model_training.py
   ```

4. **Model Evaluation**

   Evaluates the trained model on the validation set and generates metrics, learning curves, and a confusion matrix.

   ```bash
   python scripts/model_evaluation.py
   ```

5. **Export Model to ONNX**

   Exports the trained model to ONNX format.

   ```bash
   python scripts/export_onnx.py
   ```

6. **Verify ONNX Model**

   Verifies that the exported ONNX model is functional.

   ```bash
   python scripts/verify_model.py
   ```

## Evaluation Metrics

The model is evaluated based on the following criteria:

- **Mean Average Precision (mAP)**: Measures the accuracy of the model on the test set.
- **Accuracy**: Overall correctness of the model's predictions.
- **Precision, Recall, and F1-Score**: Metrics for evaluating model performance across different vehicle categories.
- **Learning Curves and Confusion Matrix**: Visual tools to understand the model's training and performance.

## Report Summary

The detailed report is available in `reports/report.pdf`. It includes:

- **Abstract**: A brief summary of methods and results.
- **Data Analysis and Cleaning**: Observations and cleaning steps.
- **Data Preprocessing**: Explanation of preprocessing steps.
- **Model Architecture**: Details and rationale for choosing the model.
- **Training and Experimentation**: Training settings, hyperparameters, and modifications.
- **Results and Key Findings**: Performance metrics, learning curves, and confusion matrix.
- **Future Work**: Potential improvements and future enhancements.

## Additional Resources

For GPU access, consider using these free resources:

- [Google Colab](https://colab.research.google.com/)
- [Kaggle Kernels](https://www.kaggle.com/kernels)
- [Gradient by Paperspace](https://www.paperspace.com/gradient)
- [SageMaker by Amazon](https://aws.amazon.com/sagemaker/)
- [Codesphere](https://codesphere.com/)

## Contact Information

For any questions or issues, please reach out to:

- **Email**: ayangori7890@gmail.com
- **GitHub**: https://github.com/Ayangouri786
