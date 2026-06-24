import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"runs\detect\training_outputs\suspension_v2\results.csv"
    )

plt.figure(figsize=(8,5))

plt.plot(
    df["epoch"],
    df["train/box_loss"],
    marker="o",
    label="Training Box Loss"
)

plt.plot(
    df["epoch"],
    df["val/box_loss"],
    marker="s",
    label="Validation Box Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Convergence")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "loss_convergence_curve.png",
    dpi=600
)

plt.show()