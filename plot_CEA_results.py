import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("CEA_Nagent.csv")

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(df["n"], df["Original"], label="Original", linewidth=2, marker='o')
plt.plot(df["n"], df["Restricted"], label="Restricted", linewidth=2, marker='o')
plt.plot(df["n"], df["Unrestricted"], label="Unrestricted", linewidth=2, marker='o')
plt.xlabel("Number of agents (n)")
plt.ylabel("Final reward of manipulator (CEA)")
plt.title("CEA – Manipulator Reward vs n")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cea_results.pdf")
print("✅ CEA results plot saved to cea_results.pdf")
plt.show()
