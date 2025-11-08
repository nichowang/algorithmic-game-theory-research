import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv("CEL_fullE60.csv")

# Calculate ratios
df['restricted_ratio'] = df['Restricted Avg'] / df['Original Avg']
df['unrestricted_ratio'] = df['Unrestricted Avg'] / df['Original Avg']

# Replace inf and NaN values (where original_avg might be 0)
df['restricted_ratio'] = df['restricted_ratio'].replace([np.inf, -np.inf], np.nan)
df['unrestricted_ratio'] = df['unrestricted_ratio'].replace([np.inf, -np.inf], np.nan)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df["a_0"], df["restricted_ratio"], label="Restricted/Original", linewidth=2, color='blue')
plt.plot(df["a_0"], df["unrestricted_ratio"], label="Unrestricted/Original", linewidth=2, color='red')

plt.xlabel("Manipulator initial claim a₀", fontsize=12)
plt.ylabel("Ratio to Original Reward", fontsize=12)
plt.title("CEL - Manipulation Gain Ratios vs a₀", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save the plot
plt.savefig("cel_ratio_plot.pdf")
print("✅ CEL ratio plot saved to cel_ratio_plot.pdf")

# Print some statistics (excluding inf/nan values)
print("\n=== Statistics ===")
valid_restricted = df['restricted_ratio'].dropna()
valid_unrestricted = df['unrestricted_ratio'].dropna()

if len(valid_restricted) > 0:
    max_idx = valid_restricted.idxmax()
    print(f"Max Restricted/Original ratio: {df.loc[max_idx, 'restricted_ratio']:.2f} at a₀={df.loc[max_idx, 'a_0']:.2f}")

if len(valid_unrestricted) > 0:
    max_idx = valid_unrestricted.idxmax()
    print(f"Max Unrestricted/Original ratio: {df.loc[max_idx, 'unrestricted_ratio']:.2f} at a₀={df.loc[max_idx, 'a_0']:.2f}")

plt.show()
