import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("CEL_Nagents.csv")

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(df["n"], df["Original"], label="Original", linewidth=2, marker='o')
plt.plot(df["n"], df["Restricted"], label="Restricted", linewidth=2, marker='o')
plt.plot(df["n"], df["Unrestricted"], label="Unrestricted", linewidth=2, marker='o')
plt.xlabel("Number of agents (n)")
plt.ylabel("Final reward of manipulator (CEL)")
plt.title("CEL – Manipulator Reward vs n")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cel_n_sweep_results.pdf")
print("✅ CEL n-sweep results plot saved to cel_n_sweep_results.pdf")
plt.show()
