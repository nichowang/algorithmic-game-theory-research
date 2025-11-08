import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file
df = pd.read_excel("cea_fullE40.xlsx")

# Calculate ratios
df['restricted_ratio'] = df['restricted_avg'] / df['original_avg']
df['unrestricted_ratio'] = df['unrestricted_avg'] / df['original_avg']

# Replace inf and NaN values (where original_avg might be 0)
df['restricted_ratio'] = df['restricted_ratio'].replace([np.inf, -np.inf], np.nan)
df['unrestricted_ratio'] = df['unrestricted_ratio'].replace([np.inf, -np.inf], np.nan)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df["a0"], df["restricted_ratio"], label="Restricted/Original", linewidth=2, color='blue')
plt.plot(df["a0"], df["unrestricted_ratio"], label="Unrestricted/Original", linewidth=2, color='red')

plt.xlabel("Manipulator initial claim a₀", fontsize=12)
plt.ylabel("Ratio to Original Reward", fontsize=12)
plt.title("CEA - Manipulation Gain Ratios vs a₀ (E=40, D=100, n=4)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save the plot
plt.savefig("cea_ratio_plot.pdf")
print("✅ CEA ratio plot saved to cea_ratio_plot.pdf")

# Print some statistics
print("\n=== Statistics ===")
print(f"Max Restricted/Original ratio: {df['restricted_ratio'].max():.2f} at a₀={df.loc[df['restricted_ratio'].idxmax(), 'a0']:.2f}")
print(f"Max Unrestricted/Original ratio: {df['unrestricted_ratio'].max():.2f} at a₀={df.loc[df['unrestricted_ratio'].idxmax(), 'a0']:.2f}")

plt.show()
