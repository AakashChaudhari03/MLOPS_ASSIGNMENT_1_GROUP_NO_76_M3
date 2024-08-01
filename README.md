"# MLOPS_ASSIGNMENT_1_GROUP_NO_76_M3" 

# Machine Learning Model Deployment

## Hyperparameter Tuning

### Dataset
- **Name**: Iris Dataset
- **Description**: The dataset consists of 150 samples from three species of Iris flowers, with four features: sepal length, sepal width, petal length, and petal width.

### Model
- **Name**: Random Forest Classifier

### Hyperparameter Tuning Process
- **Tool Used**: Optuna
- **Number of Trials**: 100

### Best Parameters
- `n_estimators`: 76
- `max_depth`: 12
- `min_samples_split`: 2
- `min_samples_leaf`: 1

### Best Accuracy
- **Accuracy**: 0.9667

## Model Packaging

### Flask Application
- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: JSON with feature array
- **Output**: JSON with prediction

### Dockerfile
- **Base Image**: `python:3.9-slim`
- **Exposed Port**: 5000

## Instructions

### Running the Hyperparameter Tuning Script
```bash
python hyperparameter_tuning.py
