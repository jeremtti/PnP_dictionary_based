import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

path_results = "convergence/SD_16K_50C_1L/results_optimal_lambda.pkl"

with open(path_results, "rb") as f:
    list_results = pickle.load(f)
    
x_observed = list_results["observation"]
img = list_results["truth"]
isnr = peak_signal_noise_ratio(img, x_observed)
denoiser_name = list_results["denoiser_name"]
lambda_list = list_results["lambda_list"]    
cvg = list_results["cvg"]
psnr = list_results["psnr"]
error = list_results["error"]
psnr_inner = list_results["psnr_inner"]
error_inner = list_results["error_inner"]
runtime = list_results["runtime"]
stops = list_results["stops"]
best_x_psnr = list_results["best_x_psnr"]
best_x_error = list_results["best_x_error"]
best_psnr = list_results["best_psnr"]
best_error = list_results["best_error"]
best_lambda = list_results["best_lambda"]
best_current_dual = list_results["best_current_dual"]
best_current_dual_fast = list_results["best_current_dual_fast"]


# SHOW PSNR VS LAMBDA
significand = best_lambda / (10 ** np.floor(np.log10(best_lambda)))
exponent = int(np.floor(np.log10(best_lambda)))

colors = plt.get_cmap("Set2").colors
color_vline = colors[0]
color_hline = colors[1]

plt.figure(figsize=(6, 4))
plt.semilogx(lambda_list, psnr, color='black')
plt.axhline(isnr, color='grey', linestyle='--')
plt.axvline(
    best_lambda, color=color_vline, linestyle='--',
    label=rf"$\lambda^* = {significand:.2f} \times 10^{{{exponent}}}$"
)
plt.axhline(
    best_psnr, color=color_hline, linestyle='--',
    label=rf"$\mathrm{{PSNR}}^* = {best_psnr:.2f}\,\mathrm{{dB}}$"
)
plt.xlabel(r"$\lambda$", fontsize=14)
plt.ylabel("PSNR (dB)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

plt.suptitle(rf"$L={denoiser_name.split('_')[-1][:-1]}$", fontsize=16)
plt.tight_layout()
plt.savefig(path_results.replace("results_optimal_lambda.pkl", "psnr_vs_lambda.pdf"), bbox_inches='tight')

# SHOW ERROR VS INNER LAMBDA    
n_lambda = len(lambda_list)

fig, axs = plt.subplots(1, n_lambda, figsize=(4*n_lambda, 3.5))
axs[0].set_ylabel("PSNR (dB)")

y_values = []
for j in range(n_lambda):
    y = psnr_inner[stops[j]:stops[j+1]]
    axs[j].semilogy(y)
    axs[j].set_title(rf"$\lambda=${lambda_list[j]:.2e}")
    y_values.extend(y)
y_min, y_max = min(y_values), max(y_values)

plt.tight_layout()
plt.savefig(path_results.replace("results_optimal_lambda.pkl", "psnr_vs_inner_lambda.pdf"), bbox_inches='tight')