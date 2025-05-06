import torch
import numpy as np

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).cpu().numpy().flatten()
        preds_bin = (preds > 0.5).astype(int)
        accuracy = (preds_bin == y.values).mean()
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

def evaluate_robustness(model, X, y, attack_fn, epsilon=0.1, alpha=0.01, iters=10):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_adv = attack_fn(model, X_tensor, y_tensor, epsilon, alpha, iters)

    with torch.no_grad():
        preds = model(X_adv).cpu().numpy().flatten()
        preds_bin = (preds > 0.5).astype(int)
        robustness = (preds_bin == y.values).mean()
        print(f"Robust Accuracy (under attack): {robustness:.4f}")
        return robustness
