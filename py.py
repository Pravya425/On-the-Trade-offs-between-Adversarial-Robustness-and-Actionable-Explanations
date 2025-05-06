import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_adult_dataset
from models import SimpleNN
from adversarial import pgd_attack




# Load data
X_train, y_train, X_test, y_test = load_adult_dataset()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Load trained models
std_model = SimpleNN(X_test.shape[0])
std_model.load_state_dict(torch.load("std_model.pth"))
std_model.eval()

adv_model = SimpleNN(X_test.shape[1])
adv_model.load_state_dict(torch.load("adv_model.pth"))
adv_model.eval()

# PGD attack with variable epsilon
epsilons = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
std_accs, std_robust_accs = [], []
adv_accs, adv_robust_accs = [], []

def evaluate(model, X, y):
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        return (preds == y).float().mean().item()

for eps in epsilons:
    print(f"Evaluating ε = {eps}")

    # Standard model
    std_clean_acc = evaluate(std_model, X_test_tensor, y_test_tensor)
    X_adv_std = pgd_attack(std_model, X_test_tensor.clone(), y_test_tensor.clone(), epsilon=eps)
    std_robust_acc = evaluate(std_model, X_adv_std, y_test_tensor)

    # Adversarial model
    adv_clean_acc = evaluate(adv_model, X_test_tensor, y_test_tensor)
    X_adv_adv = pgd_attack(adv_model, X_test_tensor.clone(), y_test_tensor.clone(), epsilon=eps)
    adv_robust_acc = evaluate(adv_model, X_adv_adv, y_test_tensor)

    std_accs.append(std_clean_acc)
    std_robust_accs.append(std_robust_acc)
    adv_accs.append(adv_clean_acc)
    adv_robust_accs.append(adv_robust_acc)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epsilons, std_accs, 'o-', label='Std Model - Clean Acc')
plt.plot(epsilons, std_robust_accs, 'o--', label='Std Model - Robust Acc')
plt.plot(epsilons, adv_accs, 's-', label='Adv Model - Clean Acc')
plt.plot(epsilons, adv_robust_accs, 's--', label='Adv Model - Robust Acc')
plt.xlabel("Epsilon (ε)")
plt.ylabel("Accuracy")
plt.title("Clean vs Robust Accuracy vs Epsilon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_epsilon.png")
plt.show()
