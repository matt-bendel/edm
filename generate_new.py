# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

# 1k NFE partial denoise: LPIPS 0.0628; PSNR: 24.36

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import lpips

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch_utils import distributed as dist
from fire.fire_new import FIRE
from fire.forward_models import get_operator
from torchmetrics.functional import peak_signal_noise_ratio

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler_full_denoise(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=100, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    num_steps = 25
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    device = latents.device

    H = get_operator('inp_box', device)
    fire_runner = FIRE(net, latents, H, 'eta_scale/ffhq.npy', sigma_min ** 2)

    x_0 = PIL.Image.open('/storage/FFHQ/ffhq64/ffhq-64x64/00069/img00069001.png')
    x_0 = 2 * transforms.ToTensor()(x_0).unsqueeze(0).to(device) - 1
    y = H.H(x_0)

    # plt.imsave('tmp_x0.png', clear_color(x_0[0]))
    # plt.imsave('tmp_y.png', clear_color(H.Ht(y).view(1, 3, 64, 64)[0]))

    # fire_out = fire_runner.run_fire(latents * t_steps[0].float(), y, 1 / (t_steps[0] ** 2), 1e-3)
    # plt.imsave('tmp_fire_out.png', clear_color(fire_out[0]))


    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        if i < num_steps / 4:
            fire_runner.max_iters = 20
        elif i < num_steps / 2:
            fire_runner.max_iters = 10
        elif i < num_steps * 3 / 4:
            fire_runner.max_iters = 5
        else:
            fire_runner.max_iters = 1

        gamma_r = 1 / ((1 - t_next / t_hat) ** -1 * t_next * (2 * gamma + gamma ** 2) ** 1/2) ** 2
        gamma_r = gamma_r.unsqueeze(0).unsqueeze(0).repeat(x_hat.shape[0], 1).float()
        D_out = fire_runner.run_fire(i, x_hat.float(), y, 1e-3, 1 / (t_hat.unsqueeze(0).unsqueeze(0).repeat(x_hat.shape[0], 1).float() ** 2), gamma_r).to(torch.float64)
        # d_cur = (x_hat - denoised) / t_hat
        # x_next = x_hat + (t_next - t_hat) * d_cur
        x_next = (t_next / t_hat) * x_hat + (1 - t_next / t_hat) * D_out

        # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    plt.imsave('edm_fire_out.png', clear_color(x_next[0]))
    exit()
    return x_next

def edm_sampler_partial_denoise(
    net, latents, class_labels=None, randn_like=torch.randn_like, fname='',
    num_steps=100, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    device = latents.device

    H = get_operator('inp_box', device)
    fire_runner = FIRE(net, latents, H, 'eta_scale/ffhq.npy', sigma_min ** 2)

    x_0 = PIL.Image.open(fname)
    x_0 = 2 * transforms.ToTensor()(x_0).unsqueeze(0).to(device) - 1
    y = H.H(x_0)
    noise_sig = 1e-3
    gamma_w = 1 / (noise_sig ** 2)

    # plt.imsave('tmp_x0.png', clear_color(x_0[0]))
    # plt.imsave('tmp_y.png', clear_color(H.Ht(y).view(1, 3, 64, 64)[0]))

    # fire_out = fire_runner.run_fire(latents * t_steps[0].float(), y, 1 / (t_steps[0] ** 2), 1e-3)
    # plt.imsave('tmp_fire_out.png', clear_color(fire_out[0]))


    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    x_hat = x_next
    t_hat = t_steps[0]

    tunable_eta = 0.5

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # Denoise
        denoised_im, sigma_tilde_sq_inv = fire_runner.denoising(x_hat, 1 / (t_hat ** 2).unsqueeze(0).unsqueeze(0).repeat(x_hat.shape[0], 1).float())
        sigma_tilde_sq = 1 / sigma_tilde_sq_inv
        x_bar = denoised_im + torch.randn_like(denoised_im) * sigma_tilde_sq.sqrt()[:, 0, None, None, None]

        tr_approx = 0.
        num_samps = 50
        m = 0
        for k in range(num_samps):
            out = fire_runner.H.H(torch.randn_like(x_hat))
            m = out.shape[1]
            tr_approx += torch.sum(out ** 2, dim=1).unsqueeze(1)

        tr_approx = tr_approx / num_samps
        y_m_A_mu_2 = torch.sum((y - fire_runner.H.H(x_bar)) ** 2, dim=1).unsqueeze(1)
        sigma_bar_sq = (y_m_A_mu_2 - m / gamma_w) / tr_approx

        # Linear estimation
        x_swoop = fire_runner.linear_estimation(x_bar / sigma_bar_sq[:, 0, None, None, None], y, gamma_w, 1 / sigma_bar_sq.float())
        fire_runner.cg_initialization = x_swoop.clone()

        # Renoise
        kappa_sq = (1 + tunable_eta ** 2) * sigma_bar_sq
        n = fire_runner.renoising_edm(x_swoop, 1 / sigma_bar_sq.float(), 1 / kappa_sq.float(), gamma_w)

        # EDM update
        x_next = (t_next / t_hat) * x_hat + (1 - t_next / t_hat) * x_swoop
        x_hat = x_next + (1 - t_next / t_hat) * n
        t_hat = (t_next ** 2 + (1 - t_next / t_hat) ** 2 * kappa_sq[0, 0]).sqrt()
        print('---------------------------------')
        print(f'x_swoop_i; desired var: {(kappa_sq ** 2).cpu().numpy()}; actual var: {torch.mean((x_swoop + n - x_0) ** 2).cpu().numpy()}')
        print(f'x_(i+1); desired var: {(t_next ** 2).cpu().numpy()}; actual var: {torch.mean((x_next - x_0) ** 2).cpu().numpy()}')
        print(f'x_hat_(i+1); desired var: {(t_hat ** 2).cpu().numpy()}; actual var: {torch.mean((x_hat - x_0) ** 2).cpu().numpy()}')
        print('---------------------------------')


    # for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
    #     x_cur = x_next
    #
    #     # Increase noise temporarily.
    #     gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    #     gamma_next = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_next <= S_max else 0
    #     t_hat = net.round_sigma(t_cur + gamma * t_cur)
    #
    #     if i == 0:
    #         x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    #     else:
    #         x_hat = x_cur
    #
    #     t_hat_next = net.round_sigma(t_next + gamma_next * t_next)
    #
    #     # Euler step.
    #     kappa_i = (1 - t_next / t_hat) ** -1 * (t_hat_next ** 2 - t_next ** 2).sqrt()
    #     gamma_r = 1 / (kappa_i ** 2)
    #     gamma_r = gamma_r.unsqueeze(0).unsqueeze(0).repeat(x_hat.shape[0], 1).float()
    #     # print((1 - t_next / t_hat) ** -1)
    #     # print((t_hat_next ** 2 - t_next ** 2).sqrt())
    #     # print('----')
    #     # print(t_cur)
    #     # print(t_hat)
    #     # print('----')
    #     # print(t_next)
    #     # print(t_hat_next)
    #     # print('----')
    #     D_out_plus_kappa_i_noise = fire_runner.run_fire(kappa_i == 0, x_hat.float(), y, 1e-3, 1 / (t_hat.unsqueeze(0).unsqueeze(0).repeat(x_hat.shape[0], 1).float() ** 2), gamma_r).to(torch.float64)
    #     print(f'desired var: {(kappa_i ** 2).cpu().numpy()}; actual var: {torch.mean((D_out_plus_kappa_i_noise - x_0) ** 2).cpu().numpy()}')
    #     x_next = (t_next / t_hat) * x_hat + (1 - t_next / t_hat) * D_out_plus_kappa_i_noise
    #
    #     # Apply 2nd order correction.
    #     # if i < num_steps - 1:
    #     #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
    #     #     d_prime = (x_next - denoised) / t_next
    #     #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next, x_0

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=60., show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    lpips_vals = []
    psnr_vals = []

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        for i in range(100):
            fname = f'/storage/FFHQ/ffhq64/ffhq-64x64/00069/img000{69000 + i}.png'
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
            sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler_partial_denoise
            images, x = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, fname=fname, **sampler_kwargs)

            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for image_np in images_np:
                image_path = f'/storage/matt_models/edm_fire/partial_denoise/{sampler_kwargs["num_steps"]}/sample_{i}.png'
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            lpips_vals.append(loss_fn_vgg(images.float(), x.float()).mean().detach().cpu().numpy())
            psnr_vals.append(peak_signal_noise_ratio(images.float(), x.float()).mean().detach().cpu().numpy())

        print(f'Avg. LPIPS: {np.mean(lpips_vals)} +/- {np.std(lpips_vals) / len(lpips_vals)}')
        print(f'Avg. PSNR: {np.mean(psnr_vals)} +/- {np.std(psnr_vals) / len(psnr_vals)}')

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
