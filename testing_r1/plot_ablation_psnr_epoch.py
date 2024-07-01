import matplotlib.pyplot as plt
import numpy as np

# Given PSNR values
# psnr_model1 = [34.9, 35, 35.1, 35.02, 35, 35.01, 35.01, 34.9, 35.03, 35.02, 35.03] # withou meta-learning and orthogonal learning, with task-driven
psnr_model1 = [33.6, 35.2, 34.6, 34.9, 35, 35, 35.01, 34.8, 34.8, 35.1, 34.90] # withou meta-learning and orthogonal learning, with task-driven
psnr_model2 = [34.4, 35.7, 35.8, 35.9, 35.9, 35.9, 36, 36, 36.1, 36.1, 36.1] # TAMO

# python ./training/ablation_py/main_mdrecon_tta.py "${args[@]}" --lr 1e-5
# 35.4/ 93.1

# Steps (assuming each epoch corresponds to 300 steps)
steps_per_epoch = 300
steps = [i * steps_per_epoch for i in range(len(psnr_model1))]

# Filtered steps for x-ticks (every alternate step)
filtered_steps = steps[::2]

# Plot settings
plt.figure(figsize=(3.5, 2.5))  # typical single column width is 3.5 inches
plt.plot(steps, psnr_model1, '-o', label='Vanilla', color='blue', linewidth=1, markersize=4)
plt.plot(steps, psnr_model2, '-s', label='Ours', color='red', linewidth=1, markersize=4)

# Adding upper bound line
# plt.axhline(y=38.0, color='g', linestyle='--', label='Upper Bound')

# Setting the axes, title, and legend
plt.xlabel('Iteration step', fontsize=8)
plt.ylabel('PSNR (dB)', fontsize=8)
plt.title('MRI anatomy shift', fontsize=9)
# plt.xticks(filtered_steps, labels=[str(step) for step in filtered_steps], fontsize=8, rotation=45)
plt.xticks(filtered_steps, labels=[str(step) for step in filtered_steps], fontsize=8)

plt.yticks(fontsize=8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=7, loc='best')

# Save the figure as a high-resolution image
plt.tight_layout()
plt.savefig("results/fig3_mri_ablation/psnr_comparison_plot_steps.pdf", dpi=300)