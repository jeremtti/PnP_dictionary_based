from experiments_inpainting import *

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
    lambdas = config.get("lambdas", [1e-2, 1e-3])
    lambdas = [float(x) for x in config.get("lambdas", [1e-2, 1e-3])]
    iter_final = config.get("iter_final", 1000)
    save_path = config.get("save_path", "convergence_inpainting/SD_10C_1L/results.pkl")
    
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
    
    Phi = np.random.rand(*img[0].shape) < 0.7
    Phit = Phi
    
    nc, nxb, nyb = img.shape
    x_masked = img + std_noise * np.random.randn(nc, nxb, nyb)
    x_observed = Phi_channels(x_masked, Phi)
    gamma = 1.0

    denoiser_names = list(DENOISERS.keys())
    #denoiser_names.remove("DRUNet")

    results = {
        "observation": x_observed,
        "truth": img,
        "denoiser_names": denoiser_names,
        "lambdas": lambdas,
    }

    n = len(lambdas)

    for denoiser_name in denoiser_names:
        
        print(f"\nRunning denoiser: {denoiser_name}")

        denoiser = DENOISERS[denoiser_name]
        net = denoiser["net"]
        model = "unrolled"
        
        i = 0
        
        for best_lambda in lambdas:
            
            x_n = Phi_channels(x_observed, Phit)
            current_dual = None
            current_dual_fast = None

            error_final = [0] * iter_final
            psnr_final = [0] * iter_final
            best_psnr = 0
            runtime_final = [0] * iter_final
            cvg_final = [0] * iter_final
            t_iter = 0
            update_dual = True

            for t in tqdm(range(iter_final)):
                
                t_start = time.perf_counter()

                g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
                tmp = x_n - gamma * g_n
                
                
                x_old = x_n.copy()
                alpha = t/(t+4)
                #alpha = 0
                x_n, current_dual, current_dual_fast = apply_model(
                    model, tmp, current_dual, best_lambda, net, update_dual,
                    fast=True, dual_fast=current_dual_fast, alpha_fast=alpha
                )

                t_iter += time.perf_counter() - t_start
                runtime_final[t] = t_iter
                cvg_final[t] = np.sum((x_n - x_old) ** 2)
                psnr_final[t] = peak_signal_noise_ratio(x_n, x_truth)
                error_final[t] = error_synthesis(current_dual[0], x_observed, net, Phi, best_lambda)
                
                if psnr_final[t] > best_psnr:
                    best_x = x_n.copy()
                    best_dual = current_dual.copy()
                    best_psnr = psnr_final[t]

            x_n = Phi_channels(x_observed, Phit)
            current_dual = None
            current_dual_fast = None

            error_final_ista = [0] * iter_final
            psnr_final_ista = [0] * iter_final
            best_psnr_ista = 0
            runtime_final_ista = [0] * iter_final
            cvg_final_ista = [0] * iter_final
            t_iter = 0
            update_dual = True

            for t in tqdm(range(iter_final)):
                
                t_start = time.perf_counter()

                g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
                tmp = x_n - gamma * g_n
                
                
                x_old = x_n.copy()
                #alpha = t/(t+4)
                alpha = 0
                x_n, current_dual, current_dual_fast = apply_model(
                    model, tmp, current_dual, best_lambda, net, update_dual,
                    fast=True, dual_fast=current_dual_fast, alpha_fast=alpha
                )

                t_iter += time.perf_counter() - t_start
                runtime_final_ista[t] = t_iter
                cvg_final_ista[t] = np.sum((x_n - x_old) ** 2)
                psnr_final_ista[t] = peak_signal_noise_ratio(x_n, x_truth)
                error_final_ista[t] = error_synthesis(current_dual[0], x_observed, net, Phi, best_lambda)
                
                if psnr_final_ista[t] > best_psnr_ista:
                    best_x_ista = x_n.copy()
                    best_dual_ista = current_dual.copy()
                    best_psnr_ista = psnr_final_ista[t]

            results[(denoiser_name, best_lambda)] = {
                # "best_x": best_x,
                # "best_dual": best_dual,
                "error_final": error_final,
                "error_final_ista": error_final_ista,
                "psnr_final": psnr_final,
                "psnr_final_ista": psnr_final_ista,
                "runtime_final": runtime_final,
                "runtime_final_ista": runtime_final_ista,
                "cvg_final": cvg_final,
                "cvg_final_ista": cvg_final_ista,
            }

    """
    denoiser = DENOISERS["DRUNet"]
    net = denoiser["net"]
    model = "drunet"

    print(f"\nRunning denoiser: DRUNet")

    for best_lambda in lambdas:
        
        x_n = Phi_channels(x_observed, Phit)
        current_dual = None
        current_dual_fast = None

        error_final = [0] * iter_final
        psnr_final = [0] * iter_final
        best_psnr = 0
        runtime_final = [0] * iter_final
        cvg_final = [0] * iter_final
        t_iter = 0
        update_dual = True

        for t in tqdm(range(iter_final)):
            
            t_start = time.perf_counter()

            g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
            tmp = x_n - gamma * g_n
            
            
            x_old = x_n.copy()
            x_n, current_dual = apply_model(
                    model, tmp, current_dual, best_lambda, net, update_dual
                )

            t_iter += time.perf_counter() - t_start
            runtime_final[t] = t_iter
            cvg_final[t] = np.sum((x_n - x_old) ** 2)
            psnr_final[t] = peak_signal_noise_ratio(x_n, x_truth)
            
            if psnr_final[t] > best_psnr:
                best_x = x_n.copy()
                best_psnr = psnr_final[t]

        results[("DRUNet", best_lambda)] = {
            # "best_x": best_x,
            "error_final": error_final,
            "psnr_final": psnr_final,
            "runtime_final": runtime_final,
            "cvg_final": cvg_final,
        }
    """
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
        