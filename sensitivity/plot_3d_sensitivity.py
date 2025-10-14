#!/usr/bin/env python3
"""
3D Sensitivity Analysis for VQK, d_model, and MoE experts
Generates 3D surface plots showing how RankIC varies with these hyperparameters.
"""

import os
import re
import csv
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


def parse_folder_name(folder_name: str) -> Dict:
    """
    Parse folder name to extract hyperparameters.
    Example: VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_...
    """
    params = {}

    # Extract VQK (codebook size)
    vqk_match = re.search(r'VQK(\d+)', folder_name)
    if vqk_match:
        params['VQK'] = int(vqk_match.group(1))

    # Extract market
    if 'sp500' in folder_name:
        params['market'] = 'sp500'
    elif 'csi300' in folder_name:
        params['market'] = 'csi300'

    # Extract mo (number of experts)
    mo_match = re.search(r'_mo(\d+)', folder_name)
    if mo_match:
        params['mo'] = int(mo_match.group(1))

    # Extract dm (d_model)
    dm_match = re.search(r'_dm(\d+)', folder_name)
    if dm_match:
        params['dm'] = int(dm_match.group(1))

    # Extract nh (number of heads)
    nh_match = re.search(r'_nh(\d+)', folder_name)
    if nh_match:
        params['nh'] = int(nh_match.group(1))

    return params


def read_rank_ic(metric_file: str) -> float:
    """Read RankIC from metric CSV file."""
    try:
        with open(metric_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('') == 'RankIC':
                    return float(row['values'])
    except Exception as e:
        print(f"Error reading {metric_file}: {e}")
    return None


def collect_sensitivity_data(res_dir: str, market: str) -> List[Dict]:
    """
    Collect sensitivity data from experiment results.
    Returns list of dicts with keys: VQK, mo, dm, RankIC
    Only includes VQK=[128,256,512], dm=64 (best baseline), mo=[2,4,8]
    Only looks in specific directories: res/csi300/ (nh2) and res/sp500/ (nh4)
    Selects exactly one experiment per (VQK, mo, dm) combination.
    """
    # Define the hyperparameters we want to analyze
    VALID_VQK = [128, 256, 512]
    VALID_DM = [32, 64]  # Both dm=32 and dm=64 for comparison
    VALID_MO = [2, 4, 8]

    # Define the specific directory and nh requirement
    if market == 'csi300':
        search_dir = os.path.join(res_dir, 'csi300')
        required_nh = 2
    elif market == 'sp500':
        search_dir = os.path.join(res_dir, 'sp500')
        required_nh = 4
    else:
        return []

    if not os.path.exists(search_dir):
        print(f"Directory not found: {search_dir}")
        return []

    # Collect exactly one experiment per (VQK, mo, dm) combination
    experiments = {}

    # Only look at direct subdirectories (not recursive)
    for dir_name in os.listdir(search_dir):
        dir_path = os.path.join(search_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        if market not in dir_name:
            continue

        # Parse parameters from folder name
        params = parse_folder_name(dir_name)
        if 'VQK' not in params or 'mo' not in params or 'dm' not in params or 'nh' not in params:
            continue

        # Filter: only include desired hyperparameter values
        if params['VQK'] not in VALID_VQK:
            continue
        if params['dm'] not in VALID_DM:
            continue
        if params['mo'] not in VALID_MO:
            continue
        if params['nh'] != required_nh:
            continue

        # Read RankIC from all cross-validation folds
        rank_ics = []

        for i in range(5):  # 5-fold cross-validation
            metric_file = os.path.join(dir_path, f'{i}_metric.csv')
            if os.path.exists(metric_file):
                rank_ic = read_rank_ic(metric_file)
                if rank_ic is not None:
                    rank_ics.append(rank_ic)

        # Store this experiment (only first one per combination)
        if rank_ics:
            key = (params['VQK'], params['mo'], params['dm'])
            if key not in experiments:
                experiments[key] = {
                    'VQK': params['VQK'],
                    'mo': params['mo'],
                    'dm': params['dm'],
                    'RankIC': np.mean(rank_ics),
                    'RankIC_std': np.std(rank_ics),
                    'folder': dir_name
                }

    # Convert to list
    data = list(experiments.values())
    return data


def plot_3d_surface(
    data: List[Dict],
    market: str,
    save_dir: str = "sensitivity/plots",
    include_suptitle: bool = False,   # ← 전체 제목 기본 비표시
):
    if not data:
        print(f"No data available for {market}")
        return

    fig = plt.figure(figsize=(16, 4.5))

    BEST_DM = 64
    BEST_VQK = 512
    BEST_MO = 2 if market == 'csi300' else 8

    all_ric = np.array([d['RankIC'] for d in data])
    vmin, vmax = all_ric.min(), all_ric.max()

    ax1 = fig.add_subplot(131, projection='3d')
    filtered_data1 = [d for d in data if d['dm'] == BEST_DM]
    scatter1 = None
    if filtered_data1:
        vqk_vals = np.array([d['VQK'] for d in filtered_data1])
        mo_vals = np.array([d['mo'] for d in filtered_data1])
        ric_vals = np.array([d['RankIC'] for d in filtered_data1])
        scatter1 = plot_3d_subplot(
            ax1, vqk_vals, mo_vals, ric_vals,
            'Codebook Size', 'MoE Experts', 'RankIC',
            '(a) Codebook vs MoE', vmin, vmax, add_colorbar=False
        )
        ax1.set_title('(a) Codebook vs MoE', pad=6)  # ← 플롯과 더 붙임

    ax2 = fig.add_subplot(132, projection='3d')
    filtered_data2 = [d for d in data if d['mo'] == BEST_MO]
    if filtered_data2:
        vqk_vals = np.array([d['VQK'] for d in filtered_data2])
        dm_vals = np.array([d['dm'] for d in filtered_data2])
        ric_vals = np.array([d['RankIC'] for d in filtered_data2])
        scatter2 = plot_3d_subplot(
            ax2, vqk_vals, dm_vals, ric_vals,
            'Codebook Size', '$d_{model}$', 'RankIC',
            '(b) Codebook vs $d_{model}$', vmin, vmax, add_colorbar=False
        )
        ax2.set_title('(b) Codebook vs $d_{model}$', pad=6)  # ← 플롯과 더 붙임

    ax3 = fig.add_subplot(133, projection='3d')
    filtered_data3 = [d for d in data if d['VQK'] == BEST_VQK]
    if filtered_data3:
        dm_vals = np.array([d['dm'] for d in filtered_data3])
        mo_vals = np.array([d['mo'] for d in filtered_data3])
        ric_vals = np.array([d['RankIC'] for d in filtered_data3])
        scatter3 = plot_3d_subplot(
            ax3, dm_vals, mo_vals, ric_vals,
            '$d_{model}$', 'MoE Experts', 'RankIC',
            '(c) $d_{model}$ vs MoE', vmin, vmax, add_colorbar=False
        )
        ax3.set_title('(c) $d_{model}$ vs MoE', pad=6)  # ← 플롯과 더 붙임

    # 단일 컬러바
    if filtered_data1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(scatter1, cax=cbar_ax)
        cbar.set_label('RankIC', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # 전체 제목은 기본적으로 표시 안 함
    if include_suptitle:
        market_name = 'CSI300' if market == 'csi300' else 'S&P500'
        fig.suptitle(f'{market_name}: Hyperparameter Sensitivity Analysis',
                     fontsize=13, y=0.99, fontweight='normal')
        # suptitle이 있을 때만 서브플롯 상단을 약간 내림
        plt.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.12, wspace=0.22)
    else:
        # suptitle 없으니 상단 여백을 올려서 “(b)와 겹침” 자체를 원천 제거
        plt.subplots_adjust(left=0.05, right=0.90, top=0.96, bottom=0.12, wspace=0.22)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'3d_sensitivity_{market}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved 3D plot: {out_path}")
    plt.close(fig)


def plot_3d_subplot(ax, x_values, y_values, z_values, xlabel, ylabel, zlabel, title,
                    vmin=None, vmax=None, add_colorbar=True):
    """Plot 3D scatter with surface interpolation."""

    # Use a professional colormap suitable for papers
    cmap = 'Blues'  # Professional single-color gradient

    # Create scatter plot with unified color range
    scatter = ax.scatter(x_values, y_values, z_values, c=z_values,
                        cmap=cmap, s=120, alpha=0.9, edgecolors='darkblue',
                        linewidths=0.8, vmin=vmin, vmax=vmax)

    # Try to create surface interpolation if we have enough points
    if len(x_values) >= 4:
        try:
            # Create grid for interpolation
            xi = np.linspace(x_values.min(), x_values.max(), 20)
            yi = np.linspace(y_values.min(), y_values.max(), 20)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            zi = griddata((x_values, y_values), z_values, (xi, yi), method='cubic')

            # Plot surface with transparency
            surf = ax.plot_surface(xi, yi, zi, cmap=cmap, alpha=0.25,
                                  linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)
        except Exception as e:
            pass  # Silently skip if interpolation fails

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=11, pad=15, fontweight='normal')

    # Set ticks to actual values only
    x_unique = sorted(set(x_values))
    y_unique = sorted(set(y_values))
    ax.set_xticks(x_unique)
    ax.set_yticks(y_unique)

    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Adjust viewing angle for better visibility
    ax.view_init(elev=20, azim=45)

    # Add subtle grid
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    return scatter


def plot_3d_compact(data: List[Dict], market: str, save_dir: str = "sensitivity/plots"):
    """
    Create a compact 3D scatter plot showing all three dimensions at once.
    Color represents RankIC, with VQK, dm, and mo as the three axes.
    """
    if not data:
        print(f"No data available for {market}")
        return

    # Extract data
    vqk_values = np.array([d['VQK'] for d in data])
    dm_values = np.array([d['dm'] for d in data])
    mo_values = np.array([d['mo'] for d in data])
    rankic_values = np.array([d['RankIC'] for d in data])

    # Create figure
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot with color representing RankIC
    scatter = ax.scatter(vqk_values, dm_values, mo_values,
                        c=rankic_values, cmap='viridis',
                        s=250, alpha=0.8, edgecolors='k', linewidths=1.5)

    # Add value labels to each point
    for i, (vqk, dm, mo, ric) in enumerate(zip(vqk_values, dm_values, mo_values, rankic_values)):
        ax.text(vqk, dm, mo, f'{ric:.4f}', fontsize=8, ha='center', va='bottom')

    # Labels and title
    ax.set_xlabel('Codebook Size (VQK)', fontsize=11, labelpad=10)
    ax.set_ylabel('d_model', fontsize=11, labelpad=10)
    ax.set_zlabel('MoE Experts', fontsize=11, labelpad=10)
    ax.set_title(f'{market.upper()}: 3D Sensitivity Analysis\n(Color = RankIC)',
                 fontsize=13, pad=20)

    # Set ticks to actual values only
    vqk_unique = sorted(set(vqk_values))
    dm_unique = sorted(set(dm_values))
    mo_unique = sorted(set(mo_values))
    ax.set_xticks(vqk_unique)
    ax.set_yticks(dm_unique)
    ax.set_zticks(mo_unique)

    # Add colorbar with reduced width
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=12, pad=0.05)
    cbar.set_label('RankIC', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Adjust viewing angle for better visibility
    ax.view_init(elev=15, azim=45)

    # Add grid
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'3d_sensitivity_compact_{market}.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"Saved compact 3D plot: {out_path}")

    plt.close(fig)


def main():
    """Generate 3D sensitivity analysis plots for both markets."""
    res_dir = '/workspace/FVQ-VAE/res'

    for market in ['csi300', 'sp500']:
        print(f"\n{'='*60}")
        print(f"Processing {market.upper()} market...")
        print(f"{'='*60}")

        # Collect data
        data = collect_sensitivity_data(res_dir, market)
        print(f"Found {len(data)} experiments for {market}")

        if data:
            # Print summary
            print("\nData summary:")
            for d in sorted(data, key=lambda x: (x['VQK'], x['dm'], x['mo'])):
                print(f"  VQK={d['VQK']:3d}, dm={d['dm']:2d}, mo={d['mo']:1d} -> "
                      f"RankIC={d['RankIC']:.4f} ± {d['RankIC_std']:.4f}")

            # Generate plots
            print(f"\nGenerating 3D plots for {market}...")
            plot_3d_surface(data, market)
            plot_3d_compact(data, market)
        else:
            print(f"No data found for {market}")

    print(f"\n{'='*60}")
    print("All plots saved to sensitivity/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
