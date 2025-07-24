import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


path_results = "convergence_inpainting/SD_16K_50C_1L/results.pkl"

with open(path_results, "rb") as f:
    list_results = pickle.load(f)
    
x_observed = list_results["observation"]
img = list_results["truth"]
isnr = peak_signal_noise_ratio(img, x_observed)
lambdas = list_results["lambdas"]
denoiser_names = list_results["denoiser_names"]
denoiser_names = sorted(denoiser_names, key=lambda x: int(x.split('_')[-1].rstrip('R')))
# Remove if ends with "20R"
denoiser_names = [name for name in denoiser_names if not name.endswith("20R")]

#denoiser_names = ["SD_10C_1L_1R", "SD_10C_1L_10R", "SD_10C_1L_100R", "SD_10C_1L_1000R"]
n = len(denoiser_names)

for best_lambda in lambdas:
    
    plt.figure(figsize=(16, 3*n))
    
    for i, denoiser_name in enumerate(denoiser_names):
        
        result_dict = list_results[(denoiser_name, best_lambda)]
        psnr_final = result_dict["psnr_final"]
        psnr_final_ista = result_dict["psnr_final_ista"]
        error_final = result_dict["error_final"]
        error_final_ista = result_dict["error_final_ista"]
        runtime_final = result_dict["runtime_final"]
        runtime_final_ista = result_dict["runtime_final_ista"]
        iter_final = len(psnr_final)
        
        plt.subplot(n, 4, 4*i+1)
        
        plt.text(-0.5, 0.5, rf"$L = ${denoiser_name.split('_')[-1][:-1]}", fontsize=16,
                 transform=plt.gca().transAxes, rotation=90,
                 verticalalignment='center', horizontalalignment='right')
        
        plt.semilogx(np.arange(1, iter_final + 1), psnr_final, label="FISTA")
        plt.semilogx(np.arange(1, iter_final + 1), psnr_final_ista, label="ISTA")
        plt.axhline(isnr, color='grey', linestyle='--')
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("PSNR (dB)", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)

        plt.subplot(n, 4, 4*i+2)
        plt.semilogx(runtime_final, psnr_final, label="FISTA")
        plt.semilogx(runtime_final_ista, psnr_final_ista, label="ISTA")
        plt.axhline(isnr, color='grey', linestyle='--')
        plt.xlabel("Runtime (s)", fontsize=14)
        plt.ylabel("PSNR (dB)", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)

        plt.subplot(n, 4, 4*i+3)
        plt.loglog(np.arange(1, iter_final + 1), error_final, label="FISTA")
        plt.loglog(np.arange(1, iter_final + 1), error_final_ista, label="ISTA")
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel(r"$\tilde{F}(z)$", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)

        plt.subplot(n, 4, 4*i+4)
        plt.loglog(runtime_final, error_final, label="FISTA")
        plt.loglog(runtime_final_ista, error_final_ista, label="ISTA")
        plt.xlabel("Runtime (s)", fontsize=14)
        plt.ylabel(r"$\tilde{F}(z)$", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)
        
    significand = best_lambda / (10 ** np.floor(np.log10(best_lambda)))
    exponent = int(np.floor(np.log10(best_lambda)))
    plt.suptitle(rf"$\lambda = {significand:.2f} \times 10^{{{exponent}}}$", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    lambda_name = f"{best_lambda:.2e}"
    plt.savefig(path_results.replace("results.pkl", f"convergence_{lambda_name}.pdf"), bbox_inches='tight')
    
    
    
    # PLOT FOR THE REPORT
    short_den_names = [den_name for den_name in denoiser_names if den_name.endswith(("1R", "10R", "100R", "1000R"))]
    short_n = len(short_den_names)
    fig, axes = plt.subplots(2, short_n, figsize=(4 * short_n, 6), gridspec_kw={'wspace': 0.4})
    
    for i, denoiser_name in enumerate(short_den_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        psnr_final = result_dict["psnr_final"]
        psnr_final_ista = result_dict["psnr_final_ista"]
        error_final = result_dict["error_final"]
        error_final_ista = result_dict["error_final_ista"]
        iter_final = len(psnr_final)

        # Bottom row: PSNR
        ax1 = axes[1, i] if short_n > 1 else axes[1]
        ax1.semilogx(np.arange(1, iter_final + 1), psnr_final, label="Fast")
        ax1.semilogx(np.arange(1, iter_final + 1), psnr_final_ista, label="Slow")
        ax1.axhline(isnr, color='grey', linestyle='--', label="iSNR")
        if i == 0:
            ax1.set_ylabel("PSNR (dB)", fontsize=16)
            ax1.yaxis.set_label_coords(-0.4, 0.5)
        if i == short_n - 1:
            ax1.legend(fontsize=13)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.tick_params(axis='both', which='minor', labelsize=12)
        ax1.set_xlabel("Iterations", fontsize=14)
        ax1.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
    

        # Top row: F(z)
        ax2 = axes[0, i] if short_n > 1 else axes[0]
        ax2.loglog(np.arange(1, iter_final + 1), error_final, label="Fast")
        ax2.loglog(np.arange(1, iter_final + 1), error_final_ista, label="Slow")
        ax2.set_title(rf"$L = ${denoiser_name.split('_')[-1][:-1]}", fontsize=18)
        if i == 0:
            ax2.set_ylabel(r"$\tilde{F}(z)$", fontsize=16)
            ax2.yaxis.set_label_coords(-0.4, 0.5)
        if i == short_n - 1:
            ax2.legend(fontsize=13)
        
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='minor', labelsize=12)
        ax2.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)


    fig.align_ylabels()
    
    lambda_name = f"{best_lambda:.2e}"
    #plt.tight_layout()
    plt.savefig(path_results.replace("results.pkl", f"REPORT_convergence_{lambda_name}.pdf"), bbox_inches='tight')
    

plt.figure(figsize=(12, 4*len(lambdas)))
cmap = cm.get_cmap('viridis', n)
colors = [cmap(j) for j in range(n)]

for i, best_lambda in enumerate(lambdas):
    
    plt.subplot(len(lambdas), 2, 2*i + 1)
    
    significand = best_lambda / (10 ** np.floor(np.log10(best_lambda)))
    exponent = int(np.floor(np.log10(best_lambda)))
    plt.text(-0.3, 0.5, rf"$\lambda = {significand:.2f} \times 10^{{{exponent}}}$", fontsize=16,
                 transform=plt.gca().transAxes, rotation=90,
                 verticalalignment='center', horizontalalignment='right')
    
    for j, denoiser_name in enumerate(denoiser_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        psnr_final = result_dict["psnr_final"]
        runtime_final = result_dict["runtime_final"]
        plt.semilogx(runtime_final, psnr_final, color=colors[j]) #label=f"{denoiser_name.split('_')[-1][:-1]}", color=colors[j])
    
    plt.axhline(isnr, color='grey', linestyle='--')
    plt.xlabel("Runtime (s)", fontsize=14)
    plt.ylabel("PSNR (dB)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    #plt.legend(title=r"$L$", fontsize=12)
    
    
    plt.subplot(len(lambdas), 2, 2*i + 2)
    
    for j, denoiser_name in enumerate(denoiser_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        error_final = result_dict["error_final"]
        runtime_final = result_dict["runtime_final"]
        plt.loglog(runtime_final, error_final, label=f"{denoiser_name.split('_')[-1][:-1]}", color=colors[j])
    
    plt.xlabel("Runtime (s)", fontsize=14)
    plt.ylabel(r"$\tilde{F}(z)$", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(title=r"$L$", fontsize=12)
    
    # plt.subplot(len(lambdas), 3, 3*i + 3)
    
    # for j, denoiser_name in enumerate(denoiser_names):
    #     result_dict = list_results[(denoiser_name, best_lambda)]
    #     error_final = result_dict["cvg_final"]
    #     runtime_final = result_dict["runtime_final"]
    #     plt.loglog(runtime_final, error_final, label=f"{denoiser_name.split('_')[-1][:-1]}", color=colors[j])
    
    # plt.xlabel("Runtime (s)", fontsize=14)
    # plt.ylabel(r"Convergence", fontsize=14)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.tick_params(axis='both', which='minor', labelsize=12)
    # plt.legend(title=r"$L$", fontsize=12)
    
plt.subplots_adjust(wspace=0.3)
plt.savefig(path_results.replace("results.pkl", f"fista_convergence.pdf"), bbox_inches='tight')
#plt.savefig(path_results.replace("results.pkl", f"REPORT_fista_convergence.pdf"), bbox_inches='tight')


# PLOT FOR THE REPORT

plt.figure(figsize=(12, 4 * len(lambdas)))
cmap = cm.get_cmap('viridis', n)
colors = [cmap(j) for j in range(n)]

for i, best_lambda in enumerate(lambdas):
    # Format lambda as integer if possible
    significand = best_lambda / (10 ** np.floor(np.log10(best_lambda)))
    exponent = int(np.floor(np.log10(best_lambda)))
    if np.isclose(significand, round(significand)):
        lambda_label = rf"$\lambda = {int(significand)} \times 10^{{{exponent}}}$"
    else:
        lambda_label = rf"$\lambda = {significand:.2f} \times 10^{{{exponent}}}$"

    # First column: PSNR vs Runtime
    ax1 = plt.subplot(len(lambdas), 2, 2 * i + 1)
    if i == 0:
        ax1.set_title("PSNR (dB)", fontsize=17, pad=20)
    
    plt.text(-0.15, 0.5, lambda_label, fontsize=17,
             transform=ax1.transAxes, rotation=90,
             verticalalignment='center', horizontalalignment='right')
    
    for j, denoiser_name in enumerate(denoiser_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        psnr_final = result_dict["psnr_final"]
        runtime_final = result_dict["runtime_final"]
        ax1.semilogx(runtime_final, psnr_final, color=colors[j])
    
    plt.axhline(isnr, color='grey', linestyle='--')
    
    if i == len(lambdas) - 1:
        ax1.set_xlabel("Runtime (s)", fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.tick_params(axis='both', which='minor', labelsize=13)

    # Second column: Error vs Runtime
    ax2 = plt.subplot(len(lambdas), 2, 2 * i + 2)
    if i == 0:
        ax2.set_title(r"$\tilde{F}(z)$", fontsize=17, pad=20)
    
    for j, denoiser_name in enumerate(denoiser_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        error_final = result_dict["error_final"]
        runtime_final = result_dict["runtime_final"]
        ax2.loglog(runtime_final, error_final, label=f"{denoiser_name.split('_')[-1][:-1]}", color=colors[j])
    
    if i == len(lambdas) - 1:
        ax2.set_xlabel("Runtime (s)", fontsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='minor', labelsize=13)
    if i == 0:
        ax2.legend(title=r"$L$", fontsize=13, title_fontsize=13)

plt.subplots_adjust(wspace=0.2, hspace=0.15)
plt.savefig(path_results.replace("results.pkl", f"REPORT_fista_convergence.pdf"), bbox_inches='tight')
