import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("CEL_fullE60.csv")

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(df["a_0"], df["Original Avg"], label="Original", linewidth=2)
plt.plot(df["a_0"], df["Restricted Avg"], label="Restricted", linewidth=2)
plt.plot(df["a_0"], df["Unrestricted Avg"], label="Unrestricted", linewidth=2)
plt.xlabel("Manipulator initial claim a0")
plt.ylabel("Final reward of manipulator (CEL)")
plt.title("CEL – Manipulator Reward vs a0 (unique combos; E=60, D=100, n=4)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cel_full_results.pdf")
print("✅ CEL full results plot saved to cel_full_results.pdf")
plt.show()
