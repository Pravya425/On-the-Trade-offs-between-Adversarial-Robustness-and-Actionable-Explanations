# pranay

# 🔒 Adversarial Robustness and Recourse in Image Classification

This repository contains code to train and evaluate models on MNIST and CIFAR-10 datasets under adversarial attacks and recourse mechanisms. It supports clean and adversarial training, evaluation, and visualization.

---

## 📁 Project Structure

| Folder/File           | Description |
|-----------------------|-------------|
| `data/`               | Contains dataset folders (`mnist/`, `cifar-10-batches-py/`) |
| `models/`             | Saved trained model checkpoints |
| `outputs/`            | Generated plots, result files, or logs |
| `main.py`             | Script to train or evaluate models |
| `train.py`            | Training logic and model saving |
| `evaluate.py`         | Test and evaluate performance metrics |
| `adversarial.py`      | Implements attacks like FGSM, PGD |
| `recourse.py`         | Code for generating recourse suggestions |
| `data_loader.py`      | Loads CIFAR-10 and MNIST datasets |
| `models.py`           | Model architectures (CNNs, MLPs) |
| `plotting.py`         | Utility functions for plotting |
| `plot_results.py`     | Generates evaluation plots |
| `requirements.txt`    | Python package dependencies |
| `README.md`           | This file |

---

## 🗃️ Datasets

### 📦 1. CIFAR-10

- Download from [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html)
- Extract the downloaded archive and place the folder `cifar-10-batches-py` inside the `data/` folder:

**🧠 Evaluation**
python main.py --dataset mnist --mode evaluate

**⚔️ Adversarial Attack Example**
python adversarial.py --dataset cifar10 --attack fgsm


OUTPUT:

The standard model achieved 84.31% accuracy and 79.00% robustness under adversarial attack after 20 epochs of training. The adversarially trained model had slightly lower clean accuracy (78.23%) but improved robustness (86.00%) after 10 adversarial training epochs.

/usr/local/bin/python3.11 /Users/pranayreddy/Downloads/pythonProject2/main.py 

Training Standard Model
Epoch 1, Loss: 0.6961
Epoch 2, Loss: 0.6712
Epoch 3, Loss: 0.6696
Epoch 4, Loss: 0.6690
Epoch 5, Loss: 0.6683
Epoch 6, Loss: 0.6677
Epoch 7, Loss: 0.6674
Epoch 8, Loss: 0.6668
Epoch 9, Loss: 0.6661
Epoch 10, Loss: 0.6658
Epoch 11, Loss: 0.6655
Epoch 12, Loss: 0.6653
Epoch 13, Loss: 0.6651
Epoch 14, Loss: 0.6651
Epoch 15, Loss: 0.6649
Epoch 16, Loss: 0.6649
Epoch 17, Loss: 0.6645
Epoch 18, Loss: 0.6644
Epoch 19, Loss: 0.6642
Epoch 20, Loss: 0.6641
Accuracy: 0.8431
Robust Accuracy (under attack): 0.7900

Training Adversarially Trained Model
Adversarial Epoch 1 complete
Adversarial Epoch 2 complete
Adversarial Epoch 3 complete
Adversarial Epoch 4 complete
Adversarial Epoch 5 complete
Adversarial Epoch 6 complete
Adversarial Epoch 7 complete
Adversarial Epoch 8 complete
Adversarial Epoch 9 complete
Adversarial Epoch 10 complete
Accuracy: 0.7823
Robust Accuracy (under attack): 0.8600

Process finished with exit code 0


