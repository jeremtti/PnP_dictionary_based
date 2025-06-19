import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

path_results = "convergence/SD_10C_1L/results.pkl"

with open(path_results, "rb") as f:
    list_results = pickle.load(f)
    
lambdas = list_results["lambdas"]
denoiser_names = list_results["denoiser_names"]
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
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("PSNR (dB)", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.legend(fontsize=12)

        plt.subplot(n, 4, 4*i+2)
        plt.semilogx(runtime_final, psnr_final, label="FISTA")
        plt.semilogx(runtime_final_ista, psnr_final_ista, label="ISTA")
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
        
    # Scientific notation for lambda
    lambda_name = f"{best_lambda:.2e}"
    
    plt.suptitle(r"$\lambda = $" + lambda_name, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(path_results.replace("results.pkl", f"convergence_{lambda_name}.pdf"), bbox_inches='tight')
    

plt.figure(figsize=(12, 4*len(lambdas)))
cmap = cm.get_cmap('viridis', n)
colors = [cmap(j) for j in range(n)]

for i, best_lambda in enumerate(lambdas):
    
    lambda_name = f"{best_lambda:.2e}"
    
    plt.subplot(len(lambdas), 2, 2*i + 1)
    
    plt.text(-0.3, 0.5, rf"$\lambda$={lambda_name}", fontsize=16,
                 transform=plt.gca().transAxes, rotation=90,
                 verticalalignment='center', horizontalalignment='right')
    
    for j, denoiser_name in enumerate(denoiser_names):
        result_dict = list_results[(denoiser_name, best_lambda)]
        psnr_final = result_dict["psnr_final"]
        runtime_final = result_dict["runtime_final"]
        plt.semilogx(runtime_final, psnr_final, label=f"{denoiser_name.split('_')[-1][:-1]}", color=colors[j])
    
    plt.xlabel("Runtime (s)", fontsize=14)
    plt.ylabel("PSNR (dB)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(title=r"$L$", fontsize=12)
    
    
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
    
    
plt.savefig(path_results.replace("results.pkl", f"fista_convergence.pdf"), bbox_inches='tight')
    

# PRENDRE C = 20, 50

# JUSTE FISTA -> FAIRE VARIER L SUR MEME GRAPHE

# KERNEL_SIZE PLUS GRAND: 16x16 au lieu de 5x5

# CHOIX DE L PLUS RAPIDE -> TROUVER LAMBDA OPTIMAL