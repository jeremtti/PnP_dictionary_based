import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

path_results = "convergence/SD_16K_50C_1L/results.pkl"

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
        plt.ylabel("Error", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)

        plt.subplot(n, 4, 4*i+4)
        plt.loglog(runtime_final, error_final, label="FISTA")
        plt.loglog(runtime_final_ista, error_final_ista, label="ISTA")
        plt.xlabel("Runtime (s)", fontsize=14)
        plt.ylabel("Error", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)
        
    significand = best_lambda / (10 ** np.floor(np.log10(best_lambda)))
    exponent = int(np.floor(np.log10(best_lambda)))
    plt.suptitle(rf"$\lambda = {significand:.2f} \times 10^{{{exponent}}}$", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    lambda_name = f"{best_lambda:.2e}"
    plt.savefig(path_results.replace("results.pkl", f"convergence_{lambda_name}.pdf"), bbox_inches='tight')
    #plt.savefig(path_results.replace("results.pkl", f"REPORT_convergence_{lambda_name}.pdf"), bbox_inches='tight')
    

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
    plt.ylabel("Error", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(title=r"$L$", fontsize=12)
    
plt.subplots_adjust(wspace=0.3)
plt.savefig(path_results.replace("results.pkl", f"fista_convergence.pdf"), bbox_inches='tight')
#plt.savefig(path_results.replace("results.pkl", f"REPORT_fista_convergence.pdf"), bbox_inches='tight')