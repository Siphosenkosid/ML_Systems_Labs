import matplotlib.pyplot as plt

models = ["Logistic Regression", "Random Forest"]
model_sizes = [1.03, 284.07]

plt.bar(models, model_sizes)

plt.title("Model Size Comparison")
plt.ylabel("Model Size (KB)")

plt.tight_layout()

plt.savefig("lab01/results/model_size_comparison.png")
plt.show()