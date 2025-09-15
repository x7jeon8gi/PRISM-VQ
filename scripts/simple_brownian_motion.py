#!/usr/bin/env python3
"""
4개의 브라운 운동 시계열과 Reconstruction 시각화
"""

import numpy as np
import matplotlib.pyplot as plt

# 시드 설정
np.random.seed(42)

# 파라미터
n_steps = 100
dt = 0.01
n_paths = 4
noise_level = 0.1  # reconstruction noise 레벨

# 4개의 브라운 운동과 reconstruction 생성 및 시각화
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.subplots_adjust(hspace=0.7, wspace=0.3)  # 간격 더 늘리기

time = np.arange(n_steps) * dt

for i in range(n_paths):
    # 각각 다른 시드로 브라운 운동 생성
    np.random.seed(42 + i)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    W_original = np.cumsum(dW)
    
    # Reconstruction: 원본에 노이즈 추가
    reconstruction_noise = np.random.normal(0, noise_level, n_steps)
    W_reconstructed = W_original + reconstruction_noise
    
    # 원본 (왼쪽)
    ax_orig = axes[i, 0]
    ax_orig.plot(time, W_original, linewidth=4, color='blue')
    
    # 모든 장식 제거
    for spine in ax_orig.spines.values():
        spine.set_visible(False)
    
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    ax_orig.set_xticklabels([])
    ax_orig.set_yticklabels([])
    
    # Reconstruction (오른쪽)
    ax_recon = axes[i, 1]
    ax_recon.plot(time, W_reconstructed, linewidth=4, color='blue', alpha=0.8)
    
    # 모든 장식 제거
    for spine in ax_recon.spines.values():
        spine.set_visible(False)
    
    ax_recon.set_xticks([])
    ax_recon.set_yticks([])
    ax_recon.set_xticklabels([])
    ax_recon.set_yticklabels([])
    
    # 마지막이 아니면 점선으로 구분 (더 명확하게)
    if i < n_paths - 1:
        # 왼쪽 차트 아래 점선
        ax_orig.axhline(y=ax_orig.get_ylim()[0] - 0.1, color='black', 
                       linestyle='--', linewidth=2, alpha=0.8)
        # 오른쪽 차트 아래 점선
        ax_recon.axhline(y=ax_recon.get_ylim()[0] - 0.1, color='black', 
                        linestyle='--', linewidth=2, alpha=0.8)

# 저장
plt.tight_layout()
plt.savefig('plot_cache/brownian_motion_reconstruction.png', dpi=300, bbox_inches='tight')
plt.close()

print("브라운 운동 + Reconstruction 차트 저장 완료: plot_cache/brownian_motion_reconstruction.png")
