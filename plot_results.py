import numpy as np
import matplotlib.pyplot as plt

# Load saved metrics
results = np.load("results/metrics.npy", allow_pickle=True).item()

# Extract points
x = [results["std_robustness"], results["rob_robustness"]]
accuracy = [results["std_accuracy"], results["rob_accuracy"]]
cost = [results["std_scfe"][1], results["rob_scfe"][1]]
validity = [results["std_scfe"][0], results["rob_scfe"][0]]

# Plot Recourse Cost vs Robustness
plt.figure(figsize=(6, 4))
plt.plot(x, cost, 'o-', label="SCFE Cost", color='blue')
plt.xlabel("Adversarial Robustness")
plt.ylabel("Average Recourse Cost (L1)")
plt.title("Recourse Cost vs Robustness")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/recourse_cost_vs_robustness.png")
plt.show()

# Plot Recourse Validity vs Robustness
plt.figure(figsize=(6, 4))
plt.plot(x, validity, 's--', label="SCFE Validity", color='green')
plt.xlabel("Adversarial Robustness")
plt.ylabel("Recourse Validity")
plt.title("Recourse Validity vs Robustness")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/recourse_validity_vs_robustness.png")
plt.show()

# Plot Accuracy vs Robustness
plt.figure(figsize=(6, 4))
plt.plot(x, accuracy, 'x-.', label="Accuracy", color='red')
plt.xlabel("Adversarial Robustness")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Robustness")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_vs_robustness.png")
plt.show()
