#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import random
import argparse
import matplotlib.pyplot as plt
import imageio
from tensorflow.keras.datasets import mnist

# Globals to track original MNIST images and metadata for each patch
_mnist_images = None
_patch_meta = None

def build_mnist_patch_library(kernel_size, filter_zeros=True):
    """
    Extract all kernel_size×kernel_size patches from the MNIST training set.
    Also saves the original images and patch metadata for seed location.
    Returns a NumPy array of shape (M, k, k).
    """
    global _mnist_images, _patch_meta
    (x_train, _), (_, _) = mnist.load_data()
    X = x_train.astype(np.float32) / 255.0  # Normalize to [0,1]
    _mnist_images = X.copy()               # Store the original dataset

    patches = []
    meta = []  # List of (image_index, row, col) for each patch
    N, H, W = X.shape
    for img_idx in range(N):
        img = X[img_idx]
        for i in range(H - kernel_size + 1):
            for j in range(W - kernel_size + 1):
                patches.append(img[i:i+kernel_size, j:j+kernel_size])
                meta.append((img_idx, i, j))
    patches = np.stack(patches, axis=0)
    _patch_meta = meta

    if filter_zeros:
        flat = patches.reshape(patches.shape[0], -1)
        nonzero_mask = np.any(flat != 0, axis=1)
        patches = patches[nonzero_mask]
        _patch_meta = [m for m, keep in zip(meta, nonzero_mask) if keep]

    print('Patches:', patches.shape)
    return patches


def match_patches_gpu(patches_mat, tpl_vec, mask_vec, gk_vec, err_thresh):
    """
    Compute normalized SSD between template and all patches on GPU using PyTorch.
    Returns indices of candidate patches.

    patches_mat: Tensor of shape (M, k*k)
    tpl_vec:     Tensor of shape (k*k,)
    mask_vec:    Tensor of shape (k*k,)
    gk_vec:      Tensor of shape (k*k,)
    err_thresh:  float error threshold
    """
    diff     = (patches_mat - tpl_vec.unsqueeze(0)) ** 2
    weighted = diff * gk_vec.unsqueeze(0) * mask_vec.unsqueeze(0)
    ssd_raw  = weighted.sum(dim=1)
    denom    = (mask_vec * gk_vec).sum()
    ssd      = ssd_raw / denom
    mn       = torch.min(ssd)
    thresh   = mn * (1.0 + err_thresh)
    cands    = torch.nonzero(ssd <= thresh, as_tuple=False).view(-1)
    return cands.cpu().numpy()


def init_window(window_size, kernel_size, patches):
    """
    Randomly selects a seed patch and places it at the center of the output window.
    Returns:
      window: float32 array of shape (H, W)
      mask:   uint8 array of shape (H, W), 1 for filled pixels
      seed_patch: float32 array of shape (k, k)
      seed_img:   float32 array of shape (28, 28), original MNIST image
      seed_meta:  tuple (image_index, row, col) of patch location
    """
    global _mnist_images, _patch_meta
    H, W = window_size
    half = kernel_size // 2

    window = np.zeros((H, W), dtype=np.float32)
    mask   = np.zeros((H, W), dtype=np.uint8)

    idx0 = np.random.randint(len(patches))
    seed_patch = patches[idx0]
    seed_meta = _patch_meta[idx0]
    img_idx, _, _ = seed_meta
    seed_img = _mnist_images[img_idx]

    cx, cy = H // 2, W // 2
    window[cx-half:cx+half+1, cy-half:cy+half+1] = seed_patch
    mask  [cx-half:cx+half+1, cy-half:cy+half+1] = 1

    return window, mask, seed_patch, seed_img, seed_meta


def synthesize_mnist(window_size, kernel_size, err_thresh, visualize,
                     seed_out=None, seed_img_out=None, gif_path=None, fps=10):
    """
    Perform non-parametric texture synthesis on MNIST using Efros & Leung method.
    Optionally save the seed patch, original seed image with bounding box, and an animation GIF.

    window_size:   tuple (H, W)
    kernel_size:   patch size k
    err_thresh:    error threshold epsilon
    visualize:     if True, show progress via matplotlib
    seed_out:      file path to save seed_patch image
    seed_img_out:  file path to save seed original image with red box
    gif_path:      file path to save synthesis animation
    fps:           frames per second for GIF
    """
    # Build patch library and prepare GPU tensors
    patches_np  = build_mnist_patch_library(kernel_size)
    k           = kernel_size
    M           = patches_np.shape[0]
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    patches_mat = torch.from_numpy(patches_np.reshape(M, -1)).float().to(device)
    g1d         = cv2.getGaussianKernel(k, k/6.4)
    gk_vec      = torch.from_numpy((g1d @ g1d.T).astype(np.float32).reshape(-1)).to(device)

    # Initialize window, mask, seed data
    window, mask, seed_patch, seed_img, seed_meta = \
        init_window(window_size, kernel_size, patches_np)
    H, W = window_size
    total_pixels = H * W
    half = k // 2

    # Save seed patch image
    if seed_out:
        cv2.imwrite(seed_out, (seed_patch * 255).astype(np.uint8))
    # Save original seed image with red bounding box
    if seed_img_out:
        img_color = cv2.cvtColor((seed_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        _, i, j = seed_meta
        cv2.rectangle(img_color, (j, i), (j+k-1, i+k-1), (0,0,255), 1)
        cv2.imwrite(seed_img_out, img_color)

    # Pad window and mask for border handling
    pad = half
    padded_window = np.pad(window, ((pad,pad),(pad,pad)), constant_values=0)
    padded_mask   = np.pad(mask,   ((pad,pad),(pad,pad)),   constant_values=0)

    frames = [] if gif_path else None

    # Prepare visualization
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(window, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    # Synthesis loop
    iter_count = 0
    while np.any(mask == 0):
        iter_count += 1
        filled = int(mask.sum())
        print(f"Iteration {iter_count}: {filled}/{total_pixels} filled, ε={err_thresh:.4f}", end="\r")

        # Find frontier pixels
        nb = cv2.dilate(mask, np.ones((3,3),np.uint8)) - mask
        coords = np.argwhere(nb == 1)

        for x, y in coords:
            if mask[x,y]: continue

            tpl = padded_window[x:x+k, y:y+k]
            msk = padded_mask[x:x+k, y:y+k]

            tpl_vec  = torch.from_numpy(tpl.ravel()).float().to(device)
            mask_vec = torch.from_numpy(msk.ravel()).float().to(device)

            cands = match_patches_gpu(patches_mat, tpl_vec, mask_vec, gk_vec, err_thresh)
            if cands.size == 0:
                err_thresh *= 1.1
                continue

            pick = np.random.choice(cands)
            val  = patches_np[pick, half, half]
            window[x,y] = val
            mask[x,y]   = 1

            padded_window[x+pad, y+pad] = val
            padded_mask[x+pad, y+pad]   = 1

            if visualize:
                im.set_data(window)
                fig.canvas.draw()
                plt.pause(0.001)
            if frames is not None:
                frames.append((window*255).astype(np.uint8))

    if visualize:
        plt.ioff()
        plt.show()
    if frames is not None:
        imageio.mimsave(gif_path, frames, fps=fps)

    print()
    return (window * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description='MNIST Non-Parametric Sampling Synthesis (Efros & Leung)')
    p.add_argument('--win_h',    type=int,   default=28, help='Output window height')
    p.add_argument('--win_w',    type=int,   default=28, help='Output window width')
    p.add_argument('--k',        type=int,   default=7,  help='Patch size k (k*k)')
    p.add_argument('--th',       type=float, default=0.1, help='Error threshold ε')
    p.add_argument('--viz',      action='store_true',    help='Enable dynamic visualization')
    p.add_argument('--out',      type=str,   default=None, help='Path to save synthesized result')
    p.add_argument('--seed',     type=int,   default=42,   help='Random seed')
    p.add_argument('--seed_out', type=str,   default=None, help='Path to save seed patch image')
    p.add_argument('--seed_img', type=str,   default=None, help='Path to save original seed image')
    p.add_argument('--gif',      type=str,   default=None, help='Path to save synthesis animation GIF')
    p.add_argument('--fps',      type=int,   default=30,   help='Frames per second for GIF')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate default filenames based on patch size if not provided
    if args.out is None:
        args.out = f"output/result_k{args.k}_seed{args.seed}.png"
    if args.seed_out is None:
        args.seed_out = f"output/seed_patch_k{args.k}_seed{args.seed}.png"
    if args.seed_img is None:
        args.seed_img = f"output/seed_original_k{args.k}_seed{args.seed}.png"
    if args.gif is None:
        args.gif = f"output/anim_k{args.k}_seed{args.seed}.gif"

    result = synthesize_mnist(
        window_size=(args.win_h, args.win_w),
        kernel_size=args.k,
        err_thresh=args.th,
        visualize=args.viz,
        seed_out=args.seed_out,
        seed_img_out=args.seed_img,
        gif_path=args.gif,
        fps=args.fps
    )
    if args.out:
        cv2.imwrite(args.out, result)

if __name__ == '__main__':
    main()
