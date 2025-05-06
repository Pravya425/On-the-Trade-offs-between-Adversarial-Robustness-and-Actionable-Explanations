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



