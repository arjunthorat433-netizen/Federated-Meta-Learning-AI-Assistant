# Federated Meta-Learning Project

## 1. Project Overview
This project implements Federated Learning and Model-Agnostic Meta-Learning (MAML) using Python and PyTorch.

The project evaluates model performance across three independent clients.

## 2. Technologies Used
- Python
- PyTorch
- Flower Federated Learning Framework
- NumPy
- Matplotlib

## 3. Project Structure
- clients/
- federated_app/
- models/
- results/
- server/
- utils/
- quickstart-pytorch/
- final_model.pt
- maml_model.pt
- test_model.py
- test_maml_model.py
- create_results.py

## 4. Model Testing

Run model testing:

```powershell
python .\test_model.py
python .\test_maml_model.py
```

## 5. Final Independent Test Results

### Federated Learning
- Client 1: 69.00%
- Client 2: 66.50%
- Client 3: 65.50%
- Average Accuracy: 67.00%

### MAML
- Client 1: 79.00%
- Client 2: 78.50%
- Client 3: 78.00%
- Average Accuracy: 78.50%

## 6. Model Comparison
MAML Improvement: +11.50 percentage points.

## 7. Results
- Graph: results/accuracy_comparison.png
- Detailed results: results/accuracy_results.txt

## 8. License
See LICENSE file.
