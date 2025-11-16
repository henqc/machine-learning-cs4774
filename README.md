# CS 4774 - Machine Learning Coursework

Assignments and codeathons completed for CS 4774 (Machine Learning) at the University of Virginia. The coursework covers fundamental machine learning algorithms and techniques, including regression, clustering, classification, and neural networks.

## Repository Overview

Organized into assignments and codeathons that demonstrate machine learning concepts:

- **Assignments 1-4**: Core machine learning algorithms implemented from scratch
- **Codeathons 1-2**: Applied machine learning projects on real-world datasets

## Repository Structure

```
ML-CS4774/
├── 01-regression-medical-costs/
│   ├── cqd3uk_assignment_1.ipynb
│   └── insurance-data/
│       └── insurance.csv
│
├── 02-kmeans-housing-clustering/
│   ├── cqd3uk_assignment_2_kmeans.ipynb
│   └── housing-data/
│       └── housing.csv
│
├── 03-svm-classification/
│   └── cqd3uk_assignment_3_svm.ipynb
│
├── 04-neural-networks-xor/
│   └── cqd3uk_assignment_4_ann.ipynb
│
├── codeathon-01-ames-housing-regression/
│   ├── cqd3uk_codeathon_1.ipynb
│   └── ames-housing-data/
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
│
├── codeathon-02-uva-landmark-cnn/
│   └── cqd3uk_codeathon_2.ipynb
│
└── README.md
```

## Project Descriptions

### Assignment 1: Regression Models on Medical Costs

Implementation and comparison of linear regression models (gradient descent, normal equation, and SGD) to predict medical insurance costs. Includes data preprocessing, feature scaling, and model evaluation using RMSE.

### Assignment 2: K-Means Clustering on California Housing

Custom implementation of the K-means clustering algorithm with multiple distance metrics (Manhattan, Euclidean, Sup distance) applied to California housing data. Includes analysis of optimal cluster numbers using elbow plots.

### Assignment 3: SVM Classification

Implementation of Support Vector Machines with multiple kernels (linear, polynomial, RBF) for non-linear classification on the Moon dataset. Includes custom SVM implementation and comparison with scikit-learn.

### Assignment 4: Neural Networks on XOR Problem

Custom implementation of a feedforward neural network with backpropagation to solve the XOR classification problem. Includes comparison with TensorFlow/Keras implementations.

### Codeathon 1: Ames Housing Price Prediction

End-to-end machine learning project predicting housing prices in Ames, Iowa. Includes data exploration, preprocessing, model selection (linear regression, decision trees, random forests, gradient boosting), hyperparameter tuning, and evaluation.

### Codeathon 2: UVA Landmark Recognition with CNNs

Image classification project using convolutional neural networks to recognize UVA landmarks. Includes custom CNN architecture design and transfer learning with pre-trained models.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn

## Course Information

**Course**: CS 4774 - Machine Learning  
**Institution**: University of Virginia
