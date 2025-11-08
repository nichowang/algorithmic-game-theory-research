import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel("cea_fullE40.xlsx")

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(df["a0"], df["original_avg"], label="Original", linewidth=2)
plt.plot(df["a0"], df["restricted_avg"], label="Restricted", linewidth=2)
plt.plot(df["a0"], df["unrestricted_avg"], label="Unrestricted", linewidth=2)
plt.xlabel("Manipulator initial claim a0")
plt.ylabel("Final reward of manipulator (CEA)")
plt.title("CEA – Manipulator Reward vs a0 (unique combos; E=40, D=100, n=4)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cea_fullE40_results.pdf")
print("✅ CEA fullE40 results plot saved to cea_fullE40_results.pdf")
plt.show()
