import argparse
import yaml
import time
import pickle
import os

import scipy
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn

import deepinv

from pnp_unrolling.unrolled_cdl import UnrolledCDL
from utils.measurement_tools import get_operators
from utils.tools import op_norm2
from pnp_unrolling.datasets import (
    create_imagenet_dataloader,
)



DATASET = "imagenet"
COLOR = True
DEVICE = "cuda:0"
create_dataloader = create_imagenet_dataloader
DATA_PATH = "./BSDS500/BSDS500/data/images"

def get_denoiser(model, **kwargs):

    if model == "drunet":
        nc = 3 if COLOR else 1
        net = deepinv.models.DRUNet(
            in_channels=nc,
            out_channels=nc,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
            pretrained="download",
        )
        net = nn.DataParallel(net, device_ids=[int(DEVICE[-1])])
    elif model in ["analysis", "synthesis"]:
        unrolled_cdl = UnrolledCDL(type_unrolling=model, **kwargs)
        # Training unrolled networks
        net, *_ = unrolled_cdl.fit()
    else:
        raise ValueError(
            f"Requested denoiser {model} which is not available."
        )
    return net

def apply_model(model, x, dual, reg_par, net=None, update_dual=False):

    if model == "unrolled":
        net.set_lmbd(reg_par)
        x_torch = torch.tensor(x, device=DEVICE, dtype=torch.float)[None, :]
        if dual is not None:
            dual = torch.tensor(dual, device=DEVICE, dtype=torch.float)
        with torch.no_grad():
            xnet, new_dual = net(x_torch, dual)
        if not update_dual:
            return xnet.detach().cpu().numpy()[0], None
        else:
            return (
                xnet.detach().cpu().numpy()[0],
                new_dual.detach().cpu().numpy()
            )
    elif model == "identity":
        return x, None
    elif model == "drunet":
        x_torch = torch.tensor(x, device=DEVICE, dtype=torch.float)[None, :]
        with torch.no_grad():
            xnet = net(x_torch, reg_par)
        return np.clip(xnet.detach().cpu().numpy()[0], 0, 1), None

def Phi_channels(x, Phi):

    new_x = np.concatenate(
        [Phi(x[i])[None, :] for i in range(x.shape[0])],
        axis=0
    )

    return new_x

def lambda_max_synthesis(y, net, Phit):
    
    parameter = net.parameter
    Astar_y = Phi_channels(y, Phit)
    Astar_y = torch.tensor(Astar_y, device=DEVICE, dtype=torch.float)[None, :]
    Dstar_Astar_y = net.conv(Astar_y, parameter).detach().cpu().numpy()
    return np.max(np.abs(Dstar_Astar_y))

def lambda_max_analysis(y, net, Phit):
    
    parameter = net.parameter.detach().cpu().numpy()   
    C, N1, N2 = y.shape
    Astar_y = Phi_channels(y, Phit)
    lambda_max = 0
    for c in range(C):
        norm_gamma_star_i = np.sum(np.abs(parameter[:, c, :, :]))
        norm_gamma_star_i = np.inf
        for k in range(parameter.shape[0]):
            norm_gamma_star_i = min(norm_gamma_star_i, np.sum(np.abs(parameter[k, c, :, :])))
        print(norm_gamma_star_i)
        for n1 in range(N1):
            for n2 in range(N2):
                current_lambda = np.abs(Astar_y[c, n1, n2] / norm_gamma_star_i)
                if current_lambda > lambda_max:
                    lambda_max = current_lambda
                    
    return lambda_max

def error_analysis(x, y, net, Phi, lamb):
    
    error1 = (1/2) * np.sum((Phi_channels(x, Phi) - y) ** 2)
    x = torch.tensor(x, device=DEVICE, dtype=torch.float)[None, :]
    gamma_star_x = net.conv(x, net.parameter).detach().cpu().numpy()
    error2 = np.sum(np.abs(gamma_star_x))
    
    return error1 + lamb * error2

def error_synthesis(z, y, net, Phi, lamb):
    
    error2 = np.sum(np.abs(z))
    z = torch.tensor(z, device=DEVICE, dtype=torch.float)[None, :]
    D_z = net.convt(z, net.parameter).detach().cpu().numpy()
    A_D_z = Phi_channels(D_z, Phi)
    error1 = (1/2) * np.sum((A_D_z - y) ** 2)
    
    return error1 + lamb * error2

def pnp_deblurring(
    model,
    pth_kernel,
    x_observed,
    normPhi2=None,
    n_iter_per_lambda=10,
    lambda_list=None,
    warm_restart=True,
    eps_stop=None,
    net=None,
    update_dual=False,
    x_truth=None,
    std_noise=0.1
):

    n_lambda = len(lambda_list)
    if lambda_list is None:
        lambda_list = [0.5 * std_noise] * n_lambda

    model_type = model
    if model in ["analysis", "synthesis"]:
        model = "unrolled"

    Phi, Phit = get_operators(type_op="deconvolution", pth_kernel=pth_kernel)
    if normPhi2 is None:
        normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    gamma = 1.0 / normPhi2

    x_n = Phi_channels(x_observed, Phit)
    best_x_psnr = x_n.copy()
    best_x_error = x_n.copy()
    best_psnr = 0
    best_error = 0
    best_lambda = 0
    
    cvg = [1e10] * n_lambda*n_iter_per_lambda
    psnr = [0] * n_lambda
    error = [0] * n_lambda
    psnr_inner = [0] * n_lambda*n_iter_per_lambda
    error_inner = [0] * n_lambda*n_iter_per_lambda
    runtime = [0] * n_lambda*n_iter_per_lambda
    stops = [0] * (n_lambda+1)
    
    current_dual = None
    t_iter = 0
    
    i = 0
    for k in tqdm(range(n_lambda)):
        if not warm_restart:
            x_n = Phi_channels(x_observed, Phit)
            current_dual = None
        
        for _ in range(n_iter_per_lambda):
            t_start = time.perf_counter()
            g_n = Phi_channels((Phi_channels(x_n, Phi) - x_observed), Phit)
            tmp = x_n - gamma * g_n
            x_old = x_n.copy()
            
            # if i==10:
            #     print(x_n[0,:3,:3])
            x_n, current_dual = apply_model(
                model, tmp, current_dual, lambda_list[k], net, update_dual
            )
            # if i==10:
            #     print(x_n[0,:3,:3])
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
            if error[k] < best_error:
                best_error = error[k]
                best_x_error = x_n.copy()
            
    return dict(img=np.clip(x_n, 0, 1),
                cvg=cvg,
                psnr=psnr,
                error=error,
                time=runtime,
                lambda_list=lambda_list,
                psnr_inner=psnr_inner,
                error_inner=error_inner,
                stops=stops,
                best_img_psnr=best_x_psnr,
                best_img_error=best_x_error,
                best_psnr=best_psnr,
                best_error=best_error,
                best_lambda=best_lambda)

def generate_results_pnp(pth_kernel,
                         std_noise,
                         DENOISERS,
                         img,
                         n_iter_per_lambda=10,
                         n_lambda=100,
                         lambda_start=10,
                         lambda_end=1e-5,
                         eps_stop=None,
                         warm_restart=True,
                         seed=42
):

    np.random.seed(seed)

    h = scipy.io.loadmat(pth_kernel)
    h = np.array(h["blur"])

    Phi, Phit = get_operators(type_op="deconvolution", pth_kernel=pth_kernel)
    x_blurred = Phi_channels(img, Phi)
    nc, nxb, nyb = x_blurred.shape
    x_observed = x_blurred + std_noise * np.random.randn(nc, nxb, nyb)
    normPhi2 = op_norm2(Phi, Phit, x_observed.shape)
    lambdas = np.logspace(np.log10(lambda_start), np.log10(lambda_end), n_lambda)

    results = {
        "observation": x_observed,
        "truth": img
    }    
    
    for name, denoiser in DENOISERS.items():
            
        if denoiser["model"] == "synthesis":
            lambda_max = lambda_max_synthesis(x_observed, denoiser["net"], Phit)
            lambda_list = np.logspace(np.log10(1*lambda_max), np.log10(1e-5 * lambda_max), n_lambda)
        elif denoiser["model"] == "drunet":
            lambda_list = np.logspace(np.log10(1e3), np.log10(1e-2), n_lambda)
        else:
            lambda_list = lambdas
            # lambda_max = lambda_max_analysis(x_observed, denoiser["net"], Phit)
            # lambda_list = np.logspace(np.log10(0.5*lambda_max), np.log10(1e-5 * lambda_max), n_lambda)
        
        results[name] = pnp_deblurring(
            denoiser["model"],
            pth_kernel,
            x_observed,
            normPhi2=normPhi2,
            n_iter_per_lambda=n_iter_per_lambda,
            lambda_list=lambda_list,
            eps_stop=eps_stop,
            warm_restart=warm_restart,
            update_dual=True,
            net=denoiser["net"],
            x_truth=img
        )

    return results

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-config", type=str)
    args, remaining_argv = pre_parser.parse_known_args()

    # Step 2: Load YAML config if provided
    config = {}
    if args.config:
        config = load_yaml_config(args.config)

    # Step 3: Define full parser with config defaults
    parser = argparse.ArgumentParser()

    parser.add_argument("-std_noise_train", type=float, default=config.get("std_noise_train", 0.05))
    parser.add_argument("-std_noise", type=float, default=config.get("std_noise", 0.1))
    parser.add_argument("-components_list", type=int, nargs='+', default=config.get("components_list", [10, 50, 100]))
    parser.add_argument("-layers_list", type=int, nargs='+', default=config.get("layers_list", [1, 20]))
    parser.add_argument("-n_rep_list", type=int, nargs='+', default=config.get("n_rep_list", [20, 50, 100]))
    parser.add_argument("-denoiser_type_list", type=str, nargs='+', default=config.get("denoiser_type_list", ["SD", "AD"]))
    parser.add_argument("-pth_kernel", type=str, default=config.get("pth_kernel", "blur_models/no_blur.mat"))
    parser.add_argument("-img_list", type=int, nargs='+', default=config.get("img_list", [0, 1, 2]))
    parser.add_argument("-n_iter_per_lambda", type=int, default=config.get("n_iter_per_lambda", 1))
    parser.add_argument("-n_lambda", type=int, default=config.get("n_lambda", 100))
    parser.add_argument("-lambda_start", type=float, default=config.get("lambda_start", 10))
    parser.add_argument("-lambda_end", type=float, default=config.get("lambda_end", 1e-5))
    parser.add_argument("-eps_stop", type=float, default=config.get("eps_stop", None))
    parser.add_argument("-warm_restart", type=lambda x: x.lower() == "true", default=config.get("warm_restart", True))
    parser.add_argument("-save_path", type=str, default=config.get("save_path", "results.pkl"))
    args = parser.parse_args(remaining_argv)
    
    params_model = {
        "kernel_size": 5,
        "lmbd": 1e-4,
        "color": COLOR,
        "device": DEVICE,
        "dtype": torch.float,
        "optimizer": "adam",
        "path_data": DATA_PATH,
        "max_sigma_noise": args.std_noise_train,
        "min_sigma_noise": args.std_noise_train,
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

    T_START = time.time()
    
    DENOISERS = {"DRUNet": dict(model="drunet")}
    DENOISERS["DRUNet"]["net"] = get_denoiser(**DENOISERS["DRUNet"])

    for denoiser_type in args.denoiser_type_list:
        for components in args.components_list:
            for layers in args.layers_list:
                params = {k: v for k, v in params_model.items()}
                params["n_layers"] = layers
                params["n_components"] = components
                
                # ----- #REPEAT = #LAYERS -----
                if denoiser_type == "SD":
                    base_name = f"SD_{components}C_{layers}L"
                    model_type = "synthesis"
                elif denoiser_type == "AD":
                    base_name = f"AD_{components}C_{layers}L"
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
                for n_rep in args.n_rep_list:
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

    dataloader = create_dataloader(
        DATA_PATH,
        min_sigma_noise=args.std_noise,
        max_sigma_noise=args.std_noise,
        device=DEVICE,
        dtype=torch.float,
        mini_batch_size=1,
        train=False,
        color=COLOR,
        fixed_noise=True,
        crop=False,
    )
    
    T_START = time.time()
    
    list_results = []
    for i in args.img_list:
        print(f"\nProcessing image {i}...")
        img = dataloader.dataset[i][1].cpu().numpy()
        results = generate_results_pnp(args.pth_kernel,
                                       args.std_noise,
                                       DENOISERS,
                                       img,
                                       n_iter_per_lambda=args.n_iter_per_lambda,
                                       n_lambda=args.n_lambda,
                                       lambda_start=args.lambda_start,
                                       lambda_end=args.lambda_end,
                                       eps_stop=args.eps_stop,
                                       warm_restart=args.warm_restart
                                       )
        list_results.append(results)

    print(f"\nSuccessfully processed {len(args.img_list)} images in {time.time() - T_START:.2f} seconds.")

    # If path folder does not exist, create it
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    with open(args.save_path, "wb") as f:
        pickle.dump(list_results, f)
    print("\nResults saved to", args.save_path, "successfully.")

if __name__ == "__main__":
    main()