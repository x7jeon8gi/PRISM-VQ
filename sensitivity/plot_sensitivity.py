#!/usr/bin/env python3
"""
Simple sensitivity plotting for RankIC across hyperparameters.
Data is defined directly in the code for easy modification.
"""

import os
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


# 실험 데이터 정의 (여기서 직접 수정하세요)
SENSITIVITY_DATA = {
    "codebook_size": {
        "title": "Codebook Size",
        "x_label": "Codebook Size (K)",
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [128, 256, 512], "y": [0.0533, 0.0573, 0.0607], "marker": "o"},
            {"label": "S&P500", "x": [128, 256, 512], "y": [0.0096, 0.0035, 0.0078], "marker": "s"},
        ]
    },
    "experts": {
        "title": "Number of Experts", 
        "x_label": "Number of Experts",
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [2, 4, 8], "y": [0.0598, 0.0607, 0.0581], "marker": "^"},
            {"label": "S&P500", "x": [2, 4, 8], "y": [0.0053, 0.0048, 0.0096], "marker": "v"},
        ]
    },
    "transformer_hidden": {
        "title": "Transformer Hidden",
        "x_label": "Transformer Hidden (d_model)", 
        "y_label": "RankIC",
        "series": [
            {"label": "CSI300", "x": [128, 256, 512, 768], "y": [0.040, 0.045, 0.046, 0.044], "marker": "o"},
            {"label": "S&P500", "x": [128, 256, 512, 768], "y": [0.039, 0.043, 0.045, 0.043], "marker": "s"},
        ]
    },
    "transformer_heads": {
        "title": "Transformer Heads",
        "x_label": "Transformer Heads",
        "y_label": "RankIC", 
        "series": [
            {"label": "CSI300", "x": [4, 8, 16], "y": [0.043, 0.046, 0.045], "marker": "o"},
            {"label": "S&P500", "x": [4, 8, 16], "y": [0.041, 0.044, 0.043], "marker": "s"},
        ]
    }
}


def _sanitize_filename(title: str) -> str:
    """파일명으로 사용할 수 있게 제목을 정리"""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


def _annotate_points(ax: plt.Axes, x_values: List[float], y_values: List[float], color: str = None, fontsize: int = 8) -> None:
    """각 지점 위에 값(소수점 4자리)을 표시한다."""
    if not x_values or not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min
    offset = (y_range if y_range > 0 else 1e-6) * 0.02
    for xi, yi in zip(x_values, y_values):
        ax.text(xi, yi + offset, f"{yi:.4f}", ha="center", va="bottom", fontsize=fontsize, color=color)


def plot_sensitivity(data: Dict[str, Any], save_dir: str = "sensitivity/plots", show: bool = False, dpi: int = 180):
    """민감도 분석 플롯 생성"""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    # 각 시리즈 플롯
    for series in data["series"]:
        x = series["x"]
        y = series["y"] 
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                       linestyle=linestyle, color=color, capsize=3)
        else:
            ax.plot(x, y, label=label, marker=marker, 
                   linestyle=linestyle, color=color)

        # 값 라벨 표시
        _annotate_points(ax, x, y, color=color)
    
    # 축 및 제목 설정
    ax.set_xlabel(data["x_label"])
    ax.set_ylabel(data["y_label"])
    ax.set_title(data["title"])
    
    # x축 틱을 명시적으로 설정 (첫 번째 시리즈의 x값 사용)
    if data["series"]:
        x_values = data["series"][0]["x"]
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])
    
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    
    # 저장 또는 표시
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = _sanitize_filename(data["title"]) + ".png"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def main():
    """모든 민감도 분석 플롯을 생성합니다."""
    print("민감도 분석 플롯 생성 중...")
    
    # 각 실험에 대해 개별 플롯 생성
    for experiment_name, data in SENSITIVITY_DATA.items():
        print(f"플롯 생성 중: {data['title']}")
        plot_sensitivity(data, save_dir="sensitivity/plots", show=False, dpi=180)
    
    # Codebook Size와 Experts 합친 subplot 생성
    print("Codebook Size & Experts subplot 생성 중...")
    plot_codebook_experts_subplot(save_dir="sensitivity/plots", show=False, dpi=180)
    
    print("모든 플롯이 sensitivity/plots/ 폴더에 저장되었습니다.")


def plot_specific(experiment_name: str, show: bool = True):
    """특정 실험의 플롯만 생성합니다."""
    if experiment_name not in SENSITIVITY_DATA:
        print(f"사용 가능한 실험: {list(SENSITIVITY_DATA.keys())}")
        return
    
    data = SENSITIVITY_DATA[experiment_name]
    plot_sensitivity(data, save_dir="sensitivity/plots", show=show, dpi=180)


def plot_codebook_experts_subplot(save_dir: str = "sensitivity/plots", show: bool = False, dpi: int = 180):
    """Codebook Size와 Number of Experts를 1x2 subplot으로 생성합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Codebook Size 플롯
    codebook_data = SENSITIVITY_DATA["codebook_size"]
    for series in codebook_data["series"]:
        x = series["x"]
        y = series["y"]
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax1.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                        linestyle=linestyle, color=color, capsize=3)
        else:
            ax1.plot(x, y, label=label, marker=marker, 
                    linestyle=linestyle, color=color)

        # 값 라벨 표시
        _annotate_points(ax1, x, y, color=color)
    
    ax1.set_xlabel(codebook_data["x_label"])
    ax1.set_ylabel(codebook_data["y_label"])
    # ax1.set_title(codebook_data["title"])
    ax1.set_xticks([128, 256, 512])
    ax1.set_xticklabels(['128', '256', '512'])
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Experts 플롯
    experts_data = SENSITIVITY_DATA["experts"]
    for series in experts_data["series"]:
        x = series["x"]
        y = series["y"]
        label = series.get("label")
        marker = series.get("marker", "o")
        linestyle = series.get("linestyle", "-")
        color = series.get("color")
        yerr = series.get("yerr")
        
        if yerr is not None:
            ax2.errorbar(x, y, yerr=yerr, label=label, marker=marker, 
                        linestyle=linestyle, color=color, capsize=3)
        else:
            ax2.plot(x, y, label=label, marker=marker, 
                    linestyle=linestyle, color=color)

        # 값 라벨 표시
        _annotate_points(ax2, x, y, color=color)
    
    ax2.set_xlabel(experts_data["x_label"])
    ax2.set_ylabel(experts_data["y_label"])
    # ax2.set_title(experts_data["title"])
    ax2.set_xticks([2, 4, 8])
    ax2.set_xticklabels(['2', '4', '8'])
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # 전체 figure에 하나의 legend만 추가 (오른쪽 바깥, 작게 배치)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),  # 그래프에 더 가깝게
        frameon=False,
        fontsize=8,
        markerscale=0.9,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.3,
    )

    # subplot 간격 조정: 상단/우측 여백을 늘려 legend 공간 확보
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.92)
    
    # 저장 또는 표시
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = "codebook_experts_subplot.png"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved subplot: {out_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    main()
