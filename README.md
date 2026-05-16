# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network to predict customer churn using a structured telecom dataset. The work covers all six required tasks: dataset exploration, preprocessing, model building, evaluation, hyperparameter experimentation, and conceptual reflection.

## Dataset
- **File:** `customer_churn_nn.csv`
- **Rows:** 2,000 | **Features:** 16 (after dropping customer_id)
- **Target:** `churn` (0 = Retained, 1 = Churned)
- **Class imbalance:** ~98.5% retained, ~1.55% churned — handled via class weighting

## Project Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.py              ← Main script (Tasks 1–6)
├── requirements.txt
└── results/
    ├── target_distribution.png
    ├── evaluation_outputs.png       ← Confusion matrix + loss curves
    ├── model_comparison_table.csv   ← Hyperparameter experiment results
    └── model_comparison_table.png   ← Visual comparison chart
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset in the same directory as notebook.py
#    (customer_churn_nn.csv)

# 3. Run the script
python notebook.py
```

## Tasks Covered
| Task | Description |
|------|-------------|
| Task 1 | Dataset understanding — shape, types, nulls, stats, class distribution |
| Task 2 | Preprocessing — OHE encoding, StandardScaler, stratified 80/20 split |
| Task 3 | Model building — feed-forward NN with Dense + Dropout layers |
| Task 4 | Training and evaluation — confusion matrix, classification report, AUC |
| Task 5 | Hyperparameter experiments — 5 configurations compared |
| Task 6 | Reflection — weights/biases, activations, learning rate, over/underfitting |

## Key Results
- Baseline model achieves **~98% accuracy** and strong AUC on the test set
- Class weighting improves recall for the minority churn class
- Deeper networks show slightly more overfitting with this dataset size
- Tanh activation performed comparably to ReLU with proper initialisation
