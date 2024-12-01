# Credit Card Fraud Detection using Autoencoders in Keras
# Credit Card Fraud Detection using Autoencoders in Keras

This project aims to detect fraudulent credit card transactions using Autoencoders, a type of neural network used for unsupervised anomaly detection. The model is built using the Keras library in Python.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview
Credit card fraud detection is a critical problem in the financial sector. The goal of this project is to build an unsupervised learning model that can detect anomalies (i.e., fraudulent transactions) in credit card transactions. An Autoencoder is trained on normal (non-fraudulent) transactions, and any significant deviations from the reconstructed data will be classified as potential fraud.

## Dataset
This project uses the **Credit Card Fraud Detection** dataset from Kaggle. It contains information about various credit card transactions and includes labels that indicate whether a transaction is fraudulent.

- **Source**: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features**: The dataset contains features like transaction amount, time, and anonymized features for security reasons. The label for fraud detection is binary (1 for fraud, 0 for legitimate).

### Dataset Preprocessing
- Scaling the numerical features using `StandardScaler`.
- Splitting the dataset into training and testing sets.
- Handling imbalanced classes with oversampling or using appropriate metrics like ROC-AUC.

## Installation
To run the project, you need Python 3.x along with the following libraries:

- `tensorflow` (for Keras and neural network implementation)
- `numpy` (for numerical operations)
- `pandas` (for data manipulation)
- `matplotlib` (for visualizations)
- `scikit-learn` (for preprocessing and splitting data)

You can install the required dependencies using the following:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
Usage
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/KUMU-THA/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Place the dataset file (creditcard.csv) in the project folder.

Run the Python script to train and evaluate the model:

bash
Copy code
python train_model.py
The script will:

Load and preprocess the dataset.
Train an Autoencoder model to detect fraudulent transactions.
Evaluate the model's performance using metrics like precision, recall, and ROC-AUC.
Model Architecture
The Autoencoder architecture consists of the following layers:

Encoder: A few dense layers that reduce the input features to a compressed representation.
Decoder: A few dense layers that reconstruct the input features from the compressed representation.
Activation Function: ReLU for hidden layers and Sigmoid for the output layer.
Loss Function: Mean Squared Error (MSE) for measuring reconstruction error.
The model is trained to minimize the reconstruction error on normal transactions, and anomalies are detected based on how poorly the Autoencoder reconstructs the test data.

Results
The model's performance can be evaluated using metrics such as:

Precision: The proportion of true positives among the predicted positives.
Recall: The proportion of true positives among the actual positives.
F1-Score: The harmonic mean of precision and recall.
ROC-AUC: Area under the ROC curve to evaluate classification performance.
The threshold for fraud detection is determined by the reconstruction error; transactions with error greater than a specific threshold are flagged as fraudulent.

Acknowledgments
The dataset is sourced from Kaggle.
The Autoencoder model is built using Keras with TensorFlow as the backend.
Special thanks to the developers of Keras and TensorFlow for their powerful libraries.
License
This project is licensed under the MIT License - see the LICENSE file for details.

This `README.md` includes sections on the project overview, dataset, installation instructions, usage, model architecture, and evaluation metrics, providing a clear understanding of your Credit Card Fraud Detection project. Adjust the paths, repository name, and any other details according to your project setup.
