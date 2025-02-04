# AutoAI-IDS: Automated Model Selection Approach for IDS

## Overview
**AutoAI-IDS** is a meta-learning framework designed for automated model selection in **Intrusion Detection Systems (IDS)**. This approach leverages various AI models with different hyperparameters to optimize classification accuracy and inference speed. The framework provides a second-phase meta-learning process to enhance model selection efficiency, ensuring robust network security solutions.

## Datasets
This study utilizes two well-known IDS datasets:
- **NSL-KDD**: A refined version of the KDD Cup 99 dataset, used for benchmarking IDS models.
- **CICIDS-2017**: A modern dataset that includes real-world attack scenarios, designed to evaluate intrusion detection models.

## Methodology
Our framework follows a **meta-learning** approach that involves:
1. **Training multiple AI models** (Random Forest, SVM, LGBM, ADA, MLP, DNN, and KNN) with various hyperparameter configurations.
2. **Conducting frequency analysis** to determine the most frequently top-performing models.
3. **Performing Hit-at-K analysis** to evaluate the effectiveness of model selection for optimal accuracy.
4. **Comparing baseline accuracies** with meta-learning predictions to validate improvements.

## Evaluation
- **Frequency Analysis:** Determines the models that appear most frequently in the top rankings.
- **Hit-at-K Accuracy:** Measures how often the best models are selected within the top K predictions.
- **Inference Speed:** Assesses the computational efficiency of different models for real-time IDS applications.

Key findings include:
- **RF and ADA models** consistently rank highest in frequency analysis.
- **Hit-at-1 accuracy** for CICIDS-2017: **RF (99.27%)**, **ADA (96.90%)**, outperforming other models.
- **LGBM exhibits faster inference times**, but its accuracy is lower for some datasets.
- The framework significantly reduces model selection time while maintaining high accuracy.

## Features
✔ **Automated model selection** for IDS tasks.
✔ **Meta-learning-based approach** to optimize classification.
✔ **Supports multiple AI models** with varied hyperparameters.
✔ **Scalable and adaptable** for different IDS datasets.
✔ **Performance evaluation metrics** including accuracy, inference time, and frequency analysis.

## Installation
Clone the repository:
```sh
 git clone https://github.com/yourusername/AutoAI-IDS.git
 cd AutoAI-IDS
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
1. **Prepare Dataset**: Download and preprocess the NSL-KDD and CICIDS-2017 datasets.
2. **Train Models**: Run `train_models.py` to train AI models with various hyperparameters.
3. **Evaluate Performance**: Use `evaluate.py` to generate frequency and Hit-at-K analysis.
4. **Model Selection**: Run `model_selector.py` to automate the model selection process.

## Contributions
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes and push them to GitHub.
4. Submit a pull request.

## Future Work
- **Extend the framework** to other IDS datasets (e.g., UNSW-NB15, DARPA).
- **Optimize inference speed** for real-time IDS applications.
- **Enhance generalization capabilities** to detect emerging threats.

