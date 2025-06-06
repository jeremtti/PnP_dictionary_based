import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import argparse
import yaml
from time import time

def plot_psnr_vs_lambda(path_results):

    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
    
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")]
    base_names = list(set([name.rsplit("_", 1)[0] for name in names]))
    base_names.sort()
    base_names.sort(key=lambda x: (x.split("_")[0],int(x.split("_")[1][:-1])))
    
    n_rep_list = list(set([int(name.split("_")[-1][:-1]) for name in names]))
    n_rep_list.sort()
    n_rep_list.remove(1)
    
    n_plots = len(base_names)
    n_img = len(list_results)

    fig_lambda, axs_lambda = plt.subplots(
        n_plots+1, n_img, sharey=True, figsize=(4*n_img, 3*(n_plots+1))
    )

    if (n_plots + 1 == 1) and (n_img == 1):
        axs_lambda = np.array([[axs_lambda]])
    elif n_plots + 1 == 1:
        axs_lambda = axs_lambda[np.newaxis, :]
    elif n_img == 1:
        axs_lambda = axs_lambda[:, np.newaxis]

    for i, results in enumerate(list_results):
        
        x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
        img = results["truth"].transpose(1, 2, 0).clip(0, 1)
        psnr = peak_signal_noise_ratio(img, x_observed.astype(np.float32))
        
        for k, base_name in enumerate(base_names):

            for rep in [1] + n_rep_list:
                
                name = f"{base_name}_{rep}R"
                res = results[name]
                axs_lambda[k, i].semilogx(res["lambda_list"], res['psnr'], label=rep)
            
            axs_lambda[k, i].set_title(f"{base_name}")
            axs_lambda[k, i].grid(True)
            axs_lambda[k, i].axhline(y=psnr, linestyle='--', alpha=0.5, color="grey")
            axs_lambda[k, i].legend(title="# iterations", fontsize=8, title_fontsize=8)
            axs_lambda[k, i].set_xlabel(rf"$\lambda$")
        
        name = "DRUNet"
        res = results[name]
        axs_lambda[-1, i].semilogx(res["lambda_list"], res['psnr'])
        axs_lambda[-1, i].set_title(f"{name}")
        axs_lambda[-1, i].grid(True)
        axs_lambda[-1, i].axhline(y=psnr, linestyle='--', alpha=0.5, color="grey")
        axs_lambda[-1, i].set_xlabel(rf"$\lambda$")

    for ax in axs_lambda[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    fig_lambda.tight_layout(rect=[0, 0, 1, 0.9])
    path_name = path_results.split(".")[0] + "_psnr_vs_lambda.pdf"
    fig_lambda.savefig(path_name, bbox_inches="tight")

def plot_error_vs_lambda(path_results):

    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
    
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")]
    base_names = list(set([name.rsplit("_", 1)[0] for name in names]))
    base_names.sort()
    base_names.sort(key=lambda x: (x.split("_")[0],int(x.split("_")[1][:-1])))
    
    n_rep_list = list(set([int(name.split("_")[-1][:-1]) for name in names]))
    n_rep_list.sort()
    n_rep_list.remove(1)
    
    n_plots = len(base_names)
    n_img = len(list_results)

    fig_lambda, axs_lambda = plt.subplots(
        n_plots, n_img, figsize=(4*n_img, 3*(n_plots))
    )

    if (n_plots == 1) and (n_img == 1):
        axs_lambda = np.array([[axs_lambda]])
    elif n_plots == 1:
        axs_lambda = axs_lambda[np.newaxis, :]
    elif n_img == 1:
        axs_lambda = axs_lambda[:, np.newaxis]

    for i, results in enumerate(list_results):
        
        x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
        img = results["truth"].transpose(1, 2, 0).clip(0, 1)
        
        for k, base_name in enumerate(base_names):

            for rep in [1] + n_rep_list:
                
                name = f"{base_name}_{rep}R"
                res = results[name]
                axs_lambda[k, i].semilogx(res["lambda_list"], res['error'], label=rep)
            
            axs_lambda[k, i].set_title(f"{base_name}")
            axs_lambda[k, i].grid(True)
            axs_lambda[k, i].legend(title="# iterations", fontsize=8, title_fontsize=8)
            axs_lambda[k, i].set_xlabel(rf"$\lambda$")

    for ax in axs_lambda[:, 0]:
        ax.set_ylabel("Functional error")
    
    fig_lambda.tight_layout(rect=[0, 0, 1, 0.9])
    path_name = path_results.split(".")[0] + "_error_vs_lambda.pdf"
    fig_lambda.savefig(path_name, bbox_inches="tight")

def plot_psnr_vs_inner_lambda(path_results, idx=0, max_rows=None):
    
    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
        
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")] + ["DRUNet"]
    n_lambda = len(list_results[0]["DRUNet"]["lambda_list"])
    
    if max_rows is not None:
        n_rows = min(len(names), max_rows)
        indices = np.random.choice(len(names), n_rows, replace=False)
    else:
        n_rows = len(names)
        indices = np.arange(n_rows)
    fig, axs = plt.subplots(n_rows, n_lambda, figsize=(4*n_lambda, 3.5*n_rows))

    for i in range(n_rows):
        axs[i, 0].set_ylabel(names[indices[i]])
        res = list_results[idx][names[indices[i]]]
        
        y_values = []  # Collect y-values for this row

        for j in range(n_lambda):
            y = res["psnr_inner"][res["stops"][j]:res["stops"][j+1]]
            axs[i, j].semilogy(y)
            axs[i, j].set_title(rf"$\lambda=${res['lambda_list'][j]:.2e}")
            y_values.extend(y)  # Gather for consistent y-axis
        
        # Set same y-limits for the entire row
        y_min, y_max = min(y_values), max(y_values)
        # for j in range(n_lambda):
        #     axs[i, j].set_ylim(y_min, y_max)

    for ax in axs[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    
    path_name = path_results.split(".")[0] + "_psnr_vs_inner_lambda.pdf"
    fig.savefig(path_name, bbox_inches="tight")

def plot_error_vs_inner_lambda(path_results, idx=0, max_rows=None):
    
    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
        
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")] + ["DRUNet"]
    n_lambda = len(list_results[0]["DRUNet"]["lambda_list"])
    
    if max_rows is not None:
        n_rows = min(len(names), max_rows)
        indices = np.random.choice(len(names), n_rows, replace=False)
    else:
        n_rows = len(names)
        indices = np.arange(n_rows)
    fig, axs = plt.subplots(n_rows, n_lambda, figsize=(4*n_lambda, 3.5*n_rows))

    for i in range(n_rows):
        axs[i, 0].set_ylabel(names[indices[i]])
        res = list_results[idx][names[indices[i]]]
        
        y_values = []  # Collect y-values for this row

        for j in range(n_lambda):
            y = res["error_inner"][res["stops"][j]:res["stops"][j+1]]
            axs[i, j].semilogy(y)
            axs[i, j].set_title(rf"$\lambda=${res['lambda_list'][j]:.2e}")
            y_values.extend(y)  # Gather for consistent y-axis
        
        # Set same y-limits for the entire row
        y_min, y_max = min(y_values), max(y_values)
        # for j in range(n_lambda):
        #     axs[i, j].set_ylim(y_min, y_max)

    for ax in axs[:, 0]:
        ax.set_ylabel("Functional error")

    path_name = path_results.split(".")[0] + "_error_vs_inner_lambda.pdf"
    fig.savefig(path_name, bbox_inches="tight")

def plot_best_psnr_vs_components(path_results):

    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
    
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")]
    components_list = list(set([int(name.split("_")[1][:-1]) for name in names]))
    components_list.sort()
    
    n_rep_list = list(set([int(name.split("_")[-1][:-1]) for name in names]))
    n_rep_list.sort()
    n_rep_list.remove(1)
    
    splitted_names = [name.split("_") for name in names]
    base_names = list(set([name[0]+"_"+name[2] for name in splitted_names]))
    base_names.sort()
    
    n_plots = len(base_names)
    n_img = len(list_results)

    fig_lambda, axs_lambda = plt.subplots(
        n_plots, n_img, sharey=True, figsize=(5*n_img, 4*(n_plots))
    )

    if (n_plots == 1) and (n_img == 1):
        axs_lambda = np.array([[axs_lambda]])
    elif n_plots == 1:
        axs_lambda = axs_lambda[np.newaxis, :]
    elif n_img == 1:
        axs_lambda = axs_lambda[:, np.newaxis]

    for i, results in enumerate(list_results):
        
        x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
        img = results["truth"].transpose(1, 2, 0).clip(0, 1)
        psnr = peak_signal_noise_ratio(img, x_observed.astype(np.float32))
        
        for k, base_name in enumerate(base_names):
            
            pair_name = base_name.split("_")
            
            for rep in [1] + n_rep_list:
                
                denoiser_names = [pair_name[0]+f"_{components}C_"+pair_name[1]+f"_{rep}R" for components in components_list]
                psnr_list = [results[name]['best_psnr'] for name in denoiser_names]
                axs_lambda[k, i].scatter(components_list, psnr_list, label=rep, s=20)
                axs_lambda[k, i].set_xscale('log')
                axs_lambda[k, i].set_xticks(components_list)
                axs_lambda[k, i].set_xticklabels([str(val) for val in components_list])#, rotation=45)
            
            axs_lambda[k, i].set_title(f"{base_name}")
            axs_lambda[k, i].grid(True)
            axs_lambda[k, i].axhline(y=psnr, linestyle='--', alpha=0.5, color="grey")
            psnr_drunet = results["DRUNet"]['best_psnr']
            axs_lambda[k, i].axhline(y=psnr_drunet, linestyle='--', alpha=0.5, color="black")
            axs_lambda[k, i].legend(title="# iterations", fontsize=8, title_fontsize=8)
            axs_lambda[k, i].set_xlabel(f"# atoms")

    for ax in axs_lambda[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    fig_lambda.tight_layout(rect=[0, 0, 1, 0.9])
    path_name = path_results.split(".")[0] + "_best_psnr_vs_components.pdf"
    fig_lambda.savefig(path_name, bbox_inches="tight")

def plot_final_psnr_vs_runtime(path_results):
    
    with open(path_results, "rb") as f:
        list_results = pickle.load(f)
    
    names = [key for key in list_results[0].keys() if key.startswith("AD") or key.startswith("SD")]
    components_list = list(set([int(name.split("_")[1][:-1]) for name in names]))
    components_list.sort()
    
    n_rep_list = list(set([int(name.split("_")[-1][:-1]) for name in names]))
    n_rep_list.sort()
    n_rep_list.remove(1)
    
    splitted_names = [name.split("_") for name in names]
    base_names = list(set([name[0]+"_"+name[2] for name in splitted_names]))
    base_names.sort()
    
    n_plots = len(base_names)
    n_img = len(list_results)

    fig_lambda, axs_lambda = plt.subplots(
        n_plots, n_img, sharey=True, figsize=(5*n_img, 4*(n_plots))
    )

    if (n_plots == 1) and (n_img == 1):
        axs_lambda = np.array([[axs_lambda]])
    elif n_plots == 1:
        axs_lambda = axs_lambda[np.newaxis, :]
    elif n_img == 1:
        axs_lambda = axs_lambda[:, np.newaxis]

    for i, results in enumerate(list_results):
        
        x_observed = results["observation"].transpose(1, 2, 0).clip(0, 1)
        img = results["truth"].transpose(1, 2, 0).clip(0, 1)
        psnr = peak_signal_noise_ratio(img, x_observed.astype(np.float32))
        
        for k, base_name in enumerate(base_names):
            
            pair_name = base_name.split("_")
            
            for rep in [1] + n_rep_list:
                
                denoiser_names = [pair_name[0]+f"_{components}C_"+pair_name[1]+f"_{rep}R" for components in components_list]
                psnr_list = [results[name]['best_psnr'] for name in denoiser_names]
                axs_lambda[k, i].scatter(components_list, psnr_list, label=rep, s=20)
                axs_lambda[k, i].set_xscale('log')
                axs_lambda[k, i].set_xticks(components_list)
                axs_lambda[k, i].set_xticklabels([str(val) for val in components_list])#, rotation=45)
            
            axs_lambda[k, i].set_title(f"{base_name}")
            axs_lambda[k, i].grid(True)
            axs_lambda[k, i].axhline(y=psnr, linestyle='--', alpha=0.5, color="grey")
            psnr_drunet = results["DRUNet"]['best_psnr']
            axs_lambda[k, i].axhline(y=psnr_drunet, linestyle='--', alpha=0.5, color="black")
            axs_lambda[k, i].legend(title="# iterations", fontsize=8, title_fontsize=8)
            axs_lambda[k, i].set_xlabel(f"# atoms")

    for ax in axs_lambda[:, 0]:
        ax.set_ylabel("PSNR (dB)")
    fig_lambda.tight_layout(rect=[0, 0, 1, 0.9])
    path_name = path_results.split(".")[0] + "_best_psnr_vs_components.pdf"
    fig_lambda.savefig(path_name, bbox_inches="tight")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot results from a given path.")
    parser.add_argument("-config", type=str, help="Path to the config.yaml file.")
    parser.add_argument("-path_results", type=str, help="Path to the results file.")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to plot for inner lambda plots.")
    args = parser.parse_args()

    # Load path_results from config.yaml or command-line argument
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            path_results = config.get("path_results")
    elif args.path_results:
        path_results = args.path_results
    else:
        raise ValueError("Either -config or -path_results must be provided.")

    # Call all plotting functions
    T_START = time()
    print("Plotting PSNR vs lambda...")
    plot_psnr_vs_lambda(path_results)
    print("Plotting error vs lambda...")
    plot_error_vs_lambda(path_results)
    print("Plotting PSNR vs inner lambda...")
    plot_psnr_vs_inner_lambda(path_results, max_rows=args.max_rows)
    print("Plotting error vs inner lambda...")
    plot_error_vs_inner_lambda(path_results, max_rows=args.max_rows)
    print("Plotting best PSNR vs components...")
    plot_best_psnr_vs_components(path_results)
    print(f"All plots generated in {time() - T_START:.2f} seconds.")

if __name__ == "__main__":
    main()