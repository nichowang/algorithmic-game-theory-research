import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel files
df_e60 = pd.read_excel("cg_cel_results_full_E60.xlsx")
df_e80 = pd.read_excel("cg_cel_results_full_E80.xlsx")

# Calculate ratios for E=60
df_e60['restricted_ratio'] = df_e60['restricted_avg'] / df_e60['original_avg']
df_e60['unrestricted_ratio'] = df_e60['unrestricted_avg'] / df_e60['original_avg']

# Calculate ratios for E=80
df_e80['restricted_ratio'] = df_e80['restricted_avg'] / df_e80['original_avg']
df_e80['unrestricted_ratio'] = df_e80['unrestricted_avg'] / df_e80['original_avg']

# Create figure with 2 subplots (one for E=60, one for E=80)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot E=60
axes[0].plot(df_e60["a0"], df_e60["restricted_ratio"], label="Restricted/Original", linewidth=2, color='#ff7f0e')  # Orange
axes[0].plot(df_e60["a0"], df_e60["unrestricted_ratio"], label="Unrestricted/Original", linewidth=2, color='#2ca02c')  # Green
axes[0].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Ratio = 1')
axes[0].set_xlabel("Manipulator initial claim a0", fontsize=11)
axes[0].set_ylabel("Ratio to Original", fontsize=11)
axes[0].set_title("CG-CEL Manipulation Ratios (E=60, D=100, n=4)", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Plot E=80
axes[1].plot(df_e80["a0"], df_e80["restricted_ratio"], label="Restricted/Original", linewidth=2, color='#ff7f0e')  # Orange
axes[1].plot(df_e80["a0"], df_e80["unrestricted_ratio"], label="Unrestricted/Original", linewidth=2, color='#2ca02c')  # Green
axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Ratio = 1')
axes[1].set_xlabel("Manipulator initial claim a0", fontsize=11)
axes[1].set_ylabel("Ratio to Original", fontsize=11)
axes[1].set_title("CG-CEL Manipulation Ratios (E=80, D=100, n=4)", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("cg_cel_ratios_comparison.pdf", dpi=300, bbox_inches='tight')
print("✅ Saved: cg_cel_ratios_comparison.pdf")
plt.show()

# Also create individual plots with absolute values
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Plot E=60 absolute values
axes2[0].plot(df_e60["a0"], df_e60["original_avg"], label="Original", linewidth=2, color='#1f77b4')  # Blue
axes2[0].plot(df_e60["a0"], df_e60["restricted_avg"], label="Restricted", linewidth=2, color='#ff7f0e')  # Orange
axes2[0].plot(df_e60["a0"], df_e60["unrestricted_avg"], label="Unrestricted", linewidth=2, color='#2ca02c')  # Green
axes2[0].set_xlabel("Manipulator initial claim a0", fontsize=11)
axes2[0].set_ylabel("Final reward of manipulator", fontsize=11)
axes2[0].set_title("CG-CEL Rewards (E=60, D=100, n=4)", fontsize=12)
axes2[0].grid(True, alpha=0.3)
axes2[0].legend(fontsize=10)

# Plot E=80 absolute values
axes2[1].plot(df_e80["a0"], df_e80["original_avg"], label="Original", linewidth=2, color='#1f77b4')  # Blue
axes2[1].plot(df_e80["a0"], df_e80["restricted_avg"], label="Restricted", linewidth=2, color='#ff7f0e')  # Orange
axes2[1].plot(df_e80["a0"], df_e80["unrestricted_avg"], label="Unrestricted", linewidth=2, color='#2ca02c')  # Green
axes2[1].set_xlabel("Manipulator initial claim a0", fontsize=11)
axes2[1].set_ylabel("Final reward of manipulator", fontsize=11)
axes2[1].set_title("CG-CEL Rewards (E=80, D=100, n=4)", fontsize=12)
axes2[1].grid(True, alpha=0.3)
axes2[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("cg_cel_rewards_comparison.pdf", dpi=300, bbox_inches='tight')
print("✅ Saved: cg_cel_rewards_comparison.pdf")
plt.show()

# Save the updated DataFrames with ratios back to Excel
df_e60.to_excel("cg_cel_results_full_E60_with_ratios.xlsx", index=False)
df_e80.to_excel("cg_cel_results_full_E80_with_ratios.xlsx", index=False)
print("✅ Saved: cg_cel_results_full_E60_with_ratios.xlsx")
print("✅ Saved: cg_cel_results_full_E80_with_ratios.xlsx")

# Print some statistics
print("\n=== E=60 Statistics ===")
print(f"Restricted ratio range: [{df_e60['restricted_ratio'].min():.4f}, {df_e60['restricted_ratio'].max():.4f}]")
print(f"Unrestricted ratio range: [{df_e60['unrestricted_ratio'].min():.4f}, {df_e60['unrestricted_ratio'].max():.4f}]")

print("\n=== E=80 Statistics ===")
print(f"Restricted ratio range: [{df_e80['restricted_ratio'].min():.4f}, {df_e80['restricted_ratio'].max():.4f}]")
print(f"Unrestricted ratio range: [{df_e80['unrestricted_ratio'].min():.4f}, {df_e80['unrestricted_ratio'].max():.4f}]")
