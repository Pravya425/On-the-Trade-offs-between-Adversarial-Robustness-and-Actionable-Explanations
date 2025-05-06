from data_loader import load_adult_dataset
from train import train_model
from evaluate import evaluate_model, evaluate_robustness
from adversarial import pgd_attack
from models import SimpleNN
from plotting import plot_accuracy_vs_robustness

import torch
from torch.utils.data import DataLoader, TensorDataset

# Load data
X_train, X_test, y_train, y_test = load_adult_dataset()
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

results = {}

# === Standard Model ===
print("\nTraining Standard Model")
std_model = train_model(X_train, y_train, input_dim)
results['Standard'] = {
    'accuracy': evaluate_model(std_model, X_test, y_test),
    'robustness': evaluate_robustness(std_model, X_test[:100], y_test[:100], pgd_attack)
}

# === Adversarially Trained Model ===
print("\nTraining Adversarially Trained Model")
adv_model = SimpleNN(input_dim).to(device)
optimizer = torch.optim.Adam(adv_model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(10):
    adv_model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb_adv = pgd_attack(adv_model, xb, yb)
        preds = adv_model(xb_adv)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Adversarial Epoch {epoch+1} complete")

results['Adversarial'] = {
    'accuracy': evaluate_model(adv_model, X_test, y_test),
    'robustness': evaluate_robustness(adv_model, X_test[:100], y_test[:100], pgd_attack)
}

# === Plot Results ===
plot_accuracy_vs_robustness(results)
