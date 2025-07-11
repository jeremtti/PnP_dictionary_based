from experiments_frugality_inpainting import *

def parse_args():
    parser = argparse.ArgumentParser(description="Read config from YAML file.")
    parser.add_argument('-config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    std_noise_train = config.get("std_noise_train", 0.05)
    std_noise = config.get("std_noise", 0.05)
    kernel_size = config.get("kernel_size", 5)
    components_list = config.get("components_list", [10])
    layers_list = config.get("layers_list", [1])
    n_rep_list = config.get("n_rep_list", [1, 2, 3, 4, 5, 10, 20, 100, 1000])
    denoiser_type_list = config.get("denoiser_type_list", ["SD"])
    prop_mask = config.get("prop_mask", 0.7)
    img_list = config.get("img_list", [0])
    n_lambda = config.get("n_lambda", 50)
    n_iter_per_lambda = config.get("n_iter_per_lambda", 200)
    eps_stop = float(config.get("eps_stop", 1e-6)) if config.get("eps_stop") is not None else None
    warm_restart = config.get("warm_restart", True)
    iter_grad = config.get("iter_grad", 500)
    lambda_unbiased = float(config.get("lambda_unbiased", 1e-3))
    save_path = config.get("save_path", "convergence_inpainting/SD_10C_1L/results_optimal_lambda.pkl")
    
    print(f"Will save in {save_path}")

    params_model = {
            "kernel_size": kernel_size,
            "lmbd": 1e-4,
            "color": COLOR,
            "device": DEVICE,
            "dtype": torch.float,
            "optimizer": "adam",
            "path_data": DATA_PATH,
            "max_sigma_noise": std_noise_train,
            "min_sigma_noise": std_noise_train,
            "mini_batch_size": 1,
            "max_batch": 10,
            "epochs": 50,
            "avg": False,
            "rescale": False,
            "fixed_noise": True,
            "D_shared": True,
            "step_size_scaling": 1.8,
            "lr": 1e-3,
            "dataset": DATASET,
        }

    DENOISERS = {} #"DRUNet": dict(model="drunet")}
    #DENOISERS["DRUNet"]["net"] = get_denoiser(**DENOISERS["DRUNet"])

    T_START = time.time()

    for denoiser_type in denoiser_type_list:
        for components in components_list:
            for layers in layers_list:
                params = {k: v for k, v in params_model.items()}
                params["n_layers"] = layers
                params["n_components"] = components
                
                # ----- #REPEAT = #LAYERS -----
                if denoiser_type == "SD":
                    base_name = f"SD_{kernel_size}K_{components}C_{layers}L"
                    model_type = "synthesis"
                elif denoiser_type == "AD":
                    base_name = f"AD_{kernel_size}K_{components}C_{layers}L"
                    model_type = "analysis"
                
                name = f"{base_name}_{layers}R"
                DENOISERS[name] = {"model": model_type, **params}
                print(f"Training {base_name}...")
                DENOISERS[name]["net"] = get_denoiser(**DENOISERS[name])
                
                # ----- #REPEAT = 1 -----
                if layers > 1:
                    denoiser = DENOISERS[name]
                    old_net = denoiser["net"]
                    net = UnrolledCDL(
                        type_unrolling=denoiser["model"],
                        **{k: v for k, v in denoiser.items() if k not in ["model", "net"]}
                    ).unrolled_net
                    # Replace the model with only the first layer of the trained model
                    net.parameter = old_net.parameter
                    net.model = torch.nn.ModuleList([old_net.model[0]])
                    DENOISERS[f"{base_name}_1R"] = dict(net=net, model=denoiser["model"], **params)
                    DENOISERS[f"{base_name}_1R"]["n_layers"] = 1
                
                # ----- #REPEAT = N_REP -----
                for n_rep in n_rep_list:
                    if n_rep == layers:
                        continue
                    denoiser = DENOISERS[f"{base_name}_1R"]
                    old_net = denoiser["net"]
                    net = UnrolledCDL(
                        type_unrolling=denoiser["model"],
                        **{k: v for k, v in denoiser.items() if k not in ["model", "net"]}
                    ).unrolled_net
                    assert len(net.model) == 1
                    net.parameter = old_net.parameter
                    net.model = torch.nn.ModuleList([old_net.model[0]] * n_rep)
                    DENOISERS[f"{base_name}_{n_rep}R"] = dict(
                        net=net, model=denoiser["model"], **params
                    )
                    DENOISERS[f"{base_name}_{n_rep}R"]["n_layers"] = n_rep

    print(f"\nSuccessfully prepared {len(DENOISERS.keys())} denoisers in {time.time() - T_START:.2f} seconds.")
    print(DENOISERS.keys())

    dataloader = create_dataloader(
        DATA_PATH,
        min_sigma_noise=std_noise,
        max_sigma_noise=std_noise,
        device=DEVICE,
        dtype=torch.float,
        mini_batch_size=1,
        train=False,
        color=COLOR,
        fixed_noise=True,
        crop=False,
    )

    img = dataloader.dataset[img_list[0]][1].cpu().numpy()
    x_truth = img

    np.random.seed(42)
    Phi = np.random.rand(*img[0].shape) < 0.7
    Phit = Phi
    nc, nxb, nyb = img.shape
    x_masked = img + std_noise * np.random.randn(nc, nxb, nyb)
    x_observed = Phi_channels(x_masked, Phi)
    gamma = 1.0

    denoiser_name = f"{denoiser_type_list[0]}_{kernel_size}K_{components_list[0]}C_{layers_list[0]}L_{n_rep_list[0]}R"
    print(f"\nRunning denoiser: {denoiser_name}")
    denoiser = DENOISERS[denoiser_name]
    net = denoiser["net"]
    model = "unrolled"
    model_type = denoiser["model"]
    parameter = net.parameter
    step_size = gamma * 1 / net.compute_lipschitz()

    lambda_max = lambda_max_synthesis(x_observed, denoiser["net"], Phit)
    lambda_list = np.logspace(np.log10(1*lambda_max), np.log10(1e-5 * lambda_max), n_lambda)
    
    results = {
        "observation": x_observed,
        "truth": img,
        "denoiser_name": denoiser_name,
        "lambda_list": lambda_list,
    }

    n_lambda = len(lambda_list)
    if lambda_list is None:
        lambda_list = [0.5 * std_noise] * n_lambda

    x_n = Phi_channels(x_observed, Phit)
    best_x_psnr = x_n.copy()
    best_x_error = x_n.copy()
    best_psnr = 0
    best_error = 0
    best_lambda = 0
    best_psnr_unbiased = 0
    best_lambda_unbiased = 0
    best_x_unbiased = x_n.copy()
    
    cvg = [1e10] * n_lambda*n_iter_per_lambda
    psnr = [0] * n_lambda
    error = [0] * n_lambda
    psnr_inner = [0] * n_lambda*n_iter_per_lambda
    error_inner = [0] * n_lambda*n_iter_per_lambda
    runtime = [0] * n_lambda*n_iter_per_lambda
    stops = [0] * (n_lambda+1)
    psnr_unbiased_inner = [[0 for _ in range(iter_grad)] for _ in range(n_lambda)]
    psnr_unbiased = [0] * n_lambda

    current_dual = None
    current_dual_fast = None

    best_current_dual, best_current_dual_fast = None, None

    t_iter = 0

    i = 0
    for k in tqdm(range(n_lambda)):
        if not warm_restart:
            x_n = Phi_channels(x_observed, Phit)
            current_dual = None
            current_dual_fast = None
        
        for t in tqdm(range(n_iter_per_lambda)):
            t_start = time.perf_counter()
            g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
            tmp = x_n - gamma * g_n
            x_old = x_n.copy()

            if model_type != "synthesis":
                x_n, current_dual = apply_model(
                    model, tmp, current_dual, lambda_list[k], net, True
                )
            else:
                alpha = t/(t+4)
                #alpha = 0
                x_n, current_dual, current_dual_fast = apply_model(
                    model, tmp, current_dual, lambda_list[k], net, True,
                    fast=True, dual_fast=current_dual_fast, alpha_fast=alpha
                )
            
            t_iter += time.perf_counter() - t_start
            cvg[i] = np.sum((x_n - x_old) ** 2)
            runtime[i] = t_iter
            psnr_inner[i] = peak_signal_noise_ratio(x_n, x_truth)
            criterion = np.sum((x_n - x_old) ** 2 / np.sum(x_old ** 2))
            
            if model_type == "analysis":
                error_inner[i] = error_analysis(x_n, x_observed, net, Phi, lambda_list[k])
            elif model_type == "synthesis":
                error_inner[i] = error_synthesis(current_dual[0], x_observed, net, Phi, lambda_list[k])
                
            if eps_stop is not None and criterion < eps_stop:
                break
            
            i += 1
        
        stops[k+1] = i
        
        if x_truth is not None:
            psnr[k] = psnr_inner[i-1]
            error[k] = error_inner[i-1]
            if psnr[k] > best_psnr:
                best_psnr = psnr[k]
                best_x_psnr = x_n.copy()
                best_lambda = lambda_list[k]
                best_current_dual = current_dual.copy() if current_dual is not None else None
                best_current_dual_fast = current_dual_fast.copy() if current_dual_fast is not None else None
            if error[k] < best_error:
                best_error = error[k]
                best_x_error = x_n.copy()
        
        # Unbiased projection
        support_z = current_dual != 0
        support_z = torch.tensor(support_z, dtype=torch.float, device=DEVICE)
        print(f"Proportion of zero elements in the dual: {np.mean(current_dual[0] == 0):.2%}")

        z_n_proj_fast = current_dual.copy()
        z_n_proj_fast = torch.tensor(z_n_proj_fast, device=DEVICE, dtype=torch.float)
        w_n = z_n_proj_fast.clone()
        psnr_unbiased_proj_fast = []
        
        for t in tqdm(range(iter_grad)):
            old_z_n = z_n_proj_fast.clone()
            grad = net.convt(w_n, net.parameter).detach().cpu().numpy()
            grad = Phi_channels(grad, Phi) - x_observed
            #error_unbiased_proj_fast.append(np.linalg.norm(grad) + lambda_unbiased
            grad = Phi_channels(grad, Phit)
            grad = torch.tensor(grad, device=DEVICE, dtype=torch.float)
            grad = net.conv(grad, net.parameter)
            w_n -= step_size * grad
            z_n_proj_fast = w_n * support_z * (1/(1+lambda_unbiased))
            x_unbiased_proj_fast = net.convt(z_n_proj_fast, net.parameter).detach().cpu().numpy()[0]
            alpha_t = t / (t + 4)
            w_n = z_n_proj_fast + alpha_t * (z_n_proj_fast - old_z_n)
            psnr_unbiased_inner[k][t] = peak_signal_noise_ratio(x_truth, x_unbiased_proj_fast)
        
        psnr_unbiased[k] = psnr_unbiased_inner[k][-1]
        if psnr_unbiased[k] > best_psnr_unbiased:
            best_psnr_unbiased = psnr_unbiased[k]
            best_lambda_unbiased = lambda_list[k]
            best_x_unbiased = x_unbiased_proj_fast.copy()
        
              
    results["cvg"] = cvg
    results["psnr"] = psnr
    results["error"] = error
    results["psnr_inner"] = psnr_inner
    results["error_inner"] = error_inner
    results["runtime"] = runtime
    results["stops"] = stops
    results["best_x_psnr"] = best_x_psnr
    results["best_x_error"] = best_x_error
    results["best_psnr"] = best_psnr
    results["best_error"] = best_error
    results["best_lambda"] = best_lambda
    results["best_current_dual"] = best_current_dual
    results["best_current_dual_fast"] = best_current_dual_fast
    results["psnr_unbiased"] = psnr_unbiased
    results["psnr_unbiased_inner"] = psnr_unbiased_inner
    results["best_lambda_unbiased"] = best_lambda_unbiased
    results["best_x_unbiased"] = best_x_unbiased
    results["best_psnr_unbiased"] = best_psnr_unbiased
    #results["error_unbiased_proj_fast"] = error_unbiased_proj_fast


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        