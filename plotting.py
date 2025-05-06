import matplotlib.pyplot as plt

def plot_accuracy_vs_robustness(results):
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    robustness = [results[m]['robustness'] for m in models]

    plt.figure(figsize=(8, 6))
    plt.scatter(robustness, accuracy, color='blue')
    for i, model in enumerate(models):
        plt.text(robustness[i] + 0.005, accuracy[i], model, fontsize=9)

    plt.xlabel('Robustness')
    plt.ylabel('Accuracy')
    plt.title('Trade-off between Accuracy and Robustness')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
