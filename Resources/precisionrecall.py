import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"runs\detect\training_outputs\suspension_v2\results.csv"
    )

plt.figure(figsize=(8,5))

plt.plot(
    df["metrics/precision(B)"],
    df["metrics/recall(B)"],
    marker="o"
)

plt.xlabel("Precision")
plt.ylabel("Recall")

plt.title(
    "Precision-Recall Performance Curve"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "precision_recall_curve.png",
    dpi=600
)

plt.show()