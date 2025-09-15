import os
import sys
import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
from openTSNE import TSNE
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils import load_yaml_param_settings, seed_everything, get_root_dir
from dataset.dataset import init_data_loader
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from qlib.data import D
from qlib.constant import REG_US, REG_CN
import qlib

from trainer.train_vqvae import FactorVQVAE


def _get_region(code: str):
    return {"US": REG_US, "CN": REG_CN}.get(code, REG_US)


def build_datasets_from_config(config):
    universe = config['data']['universe']
    if universe == 'sp500':
        region_code = 'US'
        prefix = 'sp500'
    elif universe == 'csi300':
        region_code = 'CN'
        prefix = 'csi300'
    elif universe == 'csi500':
        region_code = 'CN'
        prefix = 'csi500'
    else:
        raise ValueError(f"Unsupported universe: {universe}")

    region = _get_region(region_code)

    pickle_base = config['data'].get('data_path', 'dataset/data')
    pred_horizon = config['vqvae']['predictor']['pred_len']
    step_len = config['data']['window_size']

    train_prepare = valid_prepare = test_prepare = None

    try:
        # Prefer prepared pickles if exist
        train_prepare = pickle.load(open(f"{pickle_base}/{region_code}/{prefix}_{step_len}_h{pred_horizon}_dl_train.pkl", 'rb'))
        valid_prepare = pickle.load(open(f"{pickle_base}/{region_code}/{prefix}_{step_len}_h{pred_horizon}_dl_valid.pkl", 'rb'))
        test_prepare  = pickle.load(open(f"{pickle_base}/{region_code}/{prefix}_{step_len}_h{pred_horizon}_dl_test.pkl",  'rb'))
        tag = 'pickle'
    except Exception:
        # Fallback to qlib Alpha158
        qlib.init(provider_uri=config['data'].get('provider_uri', str(ROOT/"qlib_data"/"us_data")), region=region)
        handler = Alpha158(**{
            'start_time': config['data']['train_period'][0],
            'end_time':   config['data']['test_period'][1],
            'fit_start_time': config['data']['train_period'][0],
            'fit_end_time':   config['data']['train_period'][1],
            'instruments': universe,
            'infer_processors': [
                {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                {'class': 'Fillna',           'kwargs': {'fields_group': 'feature'}},
                {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'prior', 'clip_outlier': True}},
                {'class': 'Fillna',           'kwargs': {'fields_group': 'prior'}},
            ],
            'learn_processors': [
                {'class': 'DropnaLabel'},
                {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}},
            ],
            'label': ["Ref($close, -5) / Ref($close, -1) - 1"],
        })
        segments = {
            'train': config['data']['train_period'],
            'valid': config['data']['valid_period'],
            'test':  config['data']['test_period'],
        }
        ds = TSDatasetH(handler=handler, segments=segments, step_len=step_len)
        train_prepare = ds.prepare(segments='train', col_set=["feature","prior","label"], data_key=DataHandlerLP.DK_L)
        valid_prepare = ds.prepare(segments='valid', col_set=["feature","prior","label"], data_key=DataHandlerLP.DK_L)
        test_prepare  = ds.prepare(segments='test',  col_set=["feature","prior","label"], data_key=DataHandlerLP.DK_I)
        for dl in (train_prepare, valid_prepare, test_prepare):
            dl.config(fillna_type='ffill+bfill')
        tag = 'qlib'
        print("========== Loading data from qlib ==========")
    return tag, train_prepare, valid_prepare, test_prepare


def select_batches_by_year_month(data_prepare, year_month_list):
    idx = data_prepare.get_index()  # MultiIndex with level 'datetime' and 'instrument'
    dt_level = idx.names.index('datetime')
    all_datetimes = pd.Index(idx.get_level_values(dt_level))

    target_dates = []
    for y, m in year_month_list:
        # 해당 연-월의 첫 거래일(최소 날짜) 선택
        mask = (all_datetimes.year == y) & (all_datetimes.month == m)
        if not mask.any():
            continue
        first_date = all_datetimes[mask].min()
        target_dates.append(first_date)

    # 각 날짜의 정수 위치 인덱스 배열을 수집 (dataset 내부 순서 기준)
    batch_indices = [np.where(all_datetimes == d)[0] for d in target_dates]
    return target_dates, batch_indices


@torch.no_grad()
def extract_embeddings_and_codes(model: FactorVQVAE, batch_tensor: torch.Tensor, config):
    # batch_tensor: shape (N_assets, T, C + P + H)
    vq_cfg = config['vqvae']
    n_feat = vq_cfg['num_features']
    n_prior = vq_cfg['num_prior_factors']

    x = batch_tensor.float()
    feature = x[:, :, 0:n_feat]
    prior   = x[:, -1, n_feat:n_feat+n_prior]
    future  = x[:, -1, n_feat+n_prior:]

    # 내부 모듈 접근: revin, spatial_encoder, quantizer
    feature_norm = model.vqvae.revin(feature, mode="norm")
    h_batch = model.vqvae.spatial_encoder(feature_norm)           # (B, d)
    z_q, _, (perplexity, min_encodings, vq_idx) = model.vqvae.quantizer(h_batch)
    return h_batch.detach().cpu().numpy(), vq_idx.detach().cpu().numpy()


@torch.no_grad()
def iterate_codes_over_loader(model: FactorVQVAE, data_prepare, config, device):
    all_codes = []
    # DailyBatchSamplerRandom를 흉내내기 위해 날짜 기준으로 그룹핑된 배치를 순회
    idx = data_prepare.get_index()
    dt_level = idx.names.index('datetime')
    all_datetimes = idx.get_level_values(dt_level)
    unique_dates = pd.Index(all_datetimes.unique())
    for d in unique_dates:
        indices = np.where(all_datetimes == d)[0]
        sampler = OneBatchSampler(indices)
        loader = DataLoader(data_prepare, batch_sampler=sampler)
        batch = next(iter(loader))
        _, k = extract_embeddings_and_codes(model, batch.to(device), config)
        all_codes.append(k)
    if len(all_codes) == 0:
        return np.array([], dtype=int)
    return np.concatenate(all_codes)


def plot_embeddings_combined(ax, emb_2d, code_idx, tickers, years, highlight_map, K_total=512):
    # 512개 코드북 전 범위에 대해 정규화/컬러맵 이산화
    cmap = plt.get_cmap('viridis', K_total)
    norm = BoundaryNorm(np.arange(-0.5, K_total + 0.5, 1), K_total)
    sc = ax.scatter(emb_2d[:,0], emb_2d[:,1], c=code_idx, cmap=cmap, norm=norm, s=12, alpha=0.65, edgecolors='none')

    # 대표 티커 주석: (TICKER, YEAR)
    for tkr, style in highlight_map.items():
        # 여러 연도 존재 가능 → 모두 표시
        matches = [i for i, name in enumerate(tickers) if name == tkr]
        for i in matches:
            ax.scatter(emb_2d[i,0], emb_2d[i,1], c='red', s=40, marker=style.get('marker','o'))
            label = f"{tkr}, {years[i]}"
            ax.text(emb_2d[i,0], emb_2d[i,1], label, fontsize=9, weight='bold', color='red')

    ax.set_title('Stage-1 Embeddings (openTSNE)', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    # 축 테두리(스파인) 제거
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.2)

    # 컬러바 (0..511)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Codebook Index (0..511)')
    cbar.set_ticks([0, 128, 256, 384, 511])


def plot_embeddings_plotly(emb_2d, code_idx, tickers, years, highlight_indices=None, K_total=512, out_png=None):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("Plotly is not installed. Please install with: pip install plotly kaleido")
        raise

    x = emb_2d[:, 0]
    y = emb_2d[:, 1]

    fig = go.Figure()
    # Base scatter for all assets
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=6,
            color=code_idx.astype(float),
            colorscale='Viridis',
            cmin=0,
            cmax=K_total-1,
            colorbar=dict(
                title='Codebook Index',
                tickmode='array',
                tickvals=[0, 128, 256, 384, 511]
            ),
            showscale=True,
            opacity=0.65,
        ),
        text=[f"{t}, {y_}" for t, y_ in zip(tickers, years)],
        hovertemplate="(%{x:.3f}, %{y:.3f})<br>Ticker-Year: %{text}<br>Code: %{marker.color}<extra></extra>",
        name='Assets',
    ))

    # Highlighted points (e.g., January only representative tickers)
    if highlight_indices:
        hi = sorted(set(highlight_indices))
        fig.add_trace(go.Scatter(
            x=[x[i] for i in hi],
            y=[y[i] for i in hi],
            mode='markers',  # markers+text 에서 텍스트 제거
            marker=dict(color='red', size=10, symbol='circle'),
            # text=[f"{tickers[i]}, {years[i]}" for i in hi],  # 주석: 라벨 텍스트는 그림을 지저분하게 하여 비활성화
            # textposition='top center',
            showlegend=False,
            hoverinfo='skip',
            name='Highlights',
        ))

    fig.update_layout(
        title='Stage-1 Embeddings (openTSNE)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    # Hide axes and grid (no border box)
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, showline=False)

    if out_png is None:
        out_png = 'plot.png'
    try:
        # requires kaleido
        fig.write_image(str(out_png), scale=2)
        print(f"Saved PNG to {out_png}")
    except Exception as e:
        print("Failed to write PNG with plotly. Install kaleido: pip install -U kaleido")
        raise


def plot_embeddings_hexbin(ax, emb_2d, code_idx, tickers, years, highlight_indices=None, K_total=512, gridsize=80, hex_cmap='Greys'):
    # 밑바탕: 밀도 헥스빈
    hb = ax.hexbin(emb_2d[:,0], emb_2d[:,1], gridsize=gridsize, bins='log', cmap=hex_cmap)
    # 상단: 옅은 산점도
    ax.scatter(emb_2d[:,0], emb_2d[:,1], s=6, alpha=0.25, edgecolors='none', c='black')

    # 코드북 색상 바는 생략(헥스빈 배경 우선). 필요시 별도 colorbar(hb) 가능
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('log10 density')

    # 하이라이트: 빨간 점만
    if highlight_indices:
        hi = sorted(set(highlight_indices))
        ax.scatter(emb_2d[hi,0], emb_2d[hi,1], s=30, c='red', alpha=0.9, edgecolors='none')

    # 미관 처리
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Stage-1 Embeddings (openTSNE) - Hexbin', fontsize=12)


def build_ticker_list_for_batch(data_prepare, date):
    idx = data_prepare.get_index()
    dt_level = idx.names.index('datetime')
    inst_level = idx.names.index('instrument') if 'instrument' in idx.names else 1
    mask = (idx.get_level_values(dt_level) == date)
    instruments = idx.get_level_values(inst_level)[mask]
    return list(instruments)


class OneBatchSampler(Sampler):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices)
    def __iter__(self):
        yield self.indices
    def __len__(self):
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(ROOT/ 'configs' / 'config.yaml'))
    parser.add_argument('--ckpt', type=str, default=str(ROOT/ 'checkpoints' / 'aaaisp500_h128_VQK512_C128_emb128_dl2p10_s42-epoch=15-val_loss=0.6465.ckpt'))
    parser.add_argument('--outdir', type=str, default=str(ROOT/ 'plot_cache'))
    parser.add_argument('--dr_method', type=str, choices=['tsne','pca'], default='tsne')
    parser.add_argument('--tsne_perplexity', type=float, default=20.0)
    parser.add_argument('--tsne_iter', type=int, default=2000)
    parser.add_argument('--tsne_n_jobs', type=int, default=-1)
    parser.add_argument('--tsne_ngm', type=str, choices=['bh','fft'], default='fft')
    parser.add_argument('--report_usage', action='store_true', help='valid/test 전체 코드북 활성화 통계 출력 및 저장')
    parser.add_argument('--months', type=str, default='1,4,7,10', help='포함할 월(1-12) 리스트, 예: 1,4,7,10')
    parser.add_argument('--style', type=str, choices=['plotly','hexbin'], default='plotly', help='출력 스타일 선택')
    parser.add_argument('--whiten', type=str, choices=['none','pre','post'], default='none', help='전역 화이트닝 위치')
    parser.add_argument('--hexbin_gridsize', type=int, default=80)
    parser.add_argument('--hexbin_cmap', type=str, default='Greys')
    parser.add_argument('--include_codebook', action='store_true', help='코드북 벡터 자체를 함께(또는 단독으로) 투영/표시')
    parser.add_argument('--codebook_subset', type=int, default=0, help='코드북에서 임의 선택할 개수(0이면 전체)')
    parser.add_argument('--codebook_seed', type=int, default=42)
    parser.add_argument('--codebook_mark', type=int, default=5, help='붉은 점으로 표시할 코드북 개수')
    parser.add_argument('--highlight_random_codes', type=int, default=0, help='무작위 선택 코드북 개수(>0이면 사용)')
    parser.add_argument('--highlight_seed', type=int, default=42, help='랜덤 코드북 선택 시드')
    parser.add_argument('--highlight_per_code', type=int, default=0, help='코드별 하이라이트 샘플 수(0이면 해당 코드 전부)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml_param_settings(args.config)
    seed_everything(config['train'].get('seed', 42))

    # 데이터 준비
    tag, train_p, valid_p, test_p = build_datasets_from_config(config)

    # 지정한 월만 선택 (기본: 분기별 1,4,7,10)
    try:
        months_sel = [int(m.strip()) for m in args.months.split(',') if m.strip()]
        months_sel = [m for m in months_sel if 1 <= m <= 12]
        if len(months_sel) == 0:
            months_sel = [1]
    except Exception:
        months_sel = [1,4,7,10]

    valid_targets = [(2020, m) for m in months_sel] + [(2021, m) for m in months_sel]
    test_targets  = [(2022, m) for m in months_sel] + [(2023, m) for m in months_sel]
    valid_dates, valid_batches = select_batches_by_year_month(valid_p, valid_targets)
    test_dates,  test_batches  = select_batches_by_year_month(test_p,  test_targets)

    # Stage1 로드
    # Lightning ckpt에서 내부 vqvae만 필요하므로 전체 로드 후 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_Tmax = 1
    stage1 = FactorVQVAE(config, dummy_Tmax)
    sd = torch.load(args.ckpt, map_location=device)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    stage1.load_state_dict(sd, strict=False)
    stage1.eval()
    stage1.to(device)

    # 대표 티커 하이라이트
    highlight = {
        'AAPL': {'marker': 'X', 'label': 'AAPL'},
        'NVDA': {'marker': 'P', 'label': 'NVDA'},
        'GOOG': {'marker': 's', 'label': 'GOOG'},
        'MSFT': {'marker': 'D', 'label': 'MSFT'},
        'AMZN': {'marker': '^', 'label': 'AMZN'},
    }

    # 모든 날짜를 하나의 임베딩 공간으로 통합
    all_dates = valid_dates + test_dates
    all_batches = valid_batches + test_batches

    H_list, K_list, Ticker_list, Year_list, Month_list = [], [], [], [], []

    for date, indices in zip(all_dates, all_batches):
        if len(indices) == 0:
            continue
        data_prepare = valid_p if date in valid_dates else test_p
        sampler = OneBatchSampler(indices)
        loader = DataLoader(data_prepare, batch_sampler=sampler)
        batch = next(iter(loader))
        h, k = extract_embeddings_and_codes(stage1, batch.to(device), config)
        tickers = build_ticker_list_for_batch(data_prepare, date)
        ts = pd.Timestamp(date)
        year_arr = np.full(len(k), ts.year)
        month_arr = np.full(len(k), ts.month)
        H_list.append(h)
        K_list.append(k)
        Ticker_list.extend(tickers)
        Year_list.extend(list(year_arr))
        Month_list.extend(list(month_arr))

    if len(H_list) == 0:
        print("No data for selected dates.")
        return

    H = np.vstack(H_list)

    # 코드북 벡터 추출 (K, d)
    codebook = stage1.vqvae.quantizer.embedding.weight.detach().cpu().numpy()
    # 코드북 서브셋
    if args.codebook_subset and args.codebook_subset > 0:
        rng = np.random.RandomState(args.codebook_seed)
        idx_all = np.arange(codebook.shape[0])
        sel = rng.choice(idx_all, size=min(args.codebook_subset, len(idx_all)), replace=False)
        codebook = codebook[sel]

    if args.whiten == 'pre':
        # 인코더 임베딩 전역 화이트닝(표준화) - 코드북과 함께 맞추기
        eps = 1e-6
        H_cat = np.vstack([H, codebook])
        H_mean = H_cat.mean(axis=0, keepdims=True)
        H_std  = H_cat.std(axis=0, keepdims=True) + eps
        H = (H - H_mean) / H_std
        codebook = (codebook - H_mean) / H_std
    K_idx = np.concatenate(K_list)
    Years = np.array(Year_list)
    Months = np.array(Month_list)

    # openTSNE 2D 임베딩
    if args.dr_method == 'tsne':
        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            learning_rate='auto',
            metric='euclidean',
            initialization='pca',
            n_jobs=args.tsne_n_jobs,
            negative_gradient_method=args.tsne_ngm,
            random_state=0,
            n_iter=args.tsne_iter,
        )
        if args.include_codebook:
            X = np.vstack([H, codebook])
            X2 = tsne.fit(X)
            H2 = X2[:H.shape[0]]
            CB2 = X2[H.shape[0]:]
        else:
            H2 = tsne.fit(H)
            CB2 = None
    else:
        # 안전장치: PCA로 폴백 (scikit-learn 사용은 제거했으므로 간단 구현)
        H_center = H - H.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_center, full_matrices=False)
        H2 = H_center @ Vt[:2].T
        if args.include_codebook:
            CB2 = (codebook - H.mean(axis=0, keepdims=True)) @ Vt[:2].T
        else:
            CB2 = None

    if args.whiten == 'post':
        # 2D 좌표 전역 화이트닝(표준화)
        eps = 1e-9
        if CB2 is not None:
            X2 = np.vstack([H2, CB2])
            X2_mean = X2.mean(axis=0, keepdims=True)
            X2_std  = X2.std(axis=0, keepdims=True) + eps
            X2 = (X2 - X2_mean) / X2_std
            H2 = X2[:H2.shape[0]]
            CB2 = X2[H2.shape[0]:]
        else:
            H2_mean = H2.mean(axis=0, keepdims=True)
            H2_std  = H2.std(axis=0, keepdims=True) + eps
            H2 = (H2 - H2_mean) / H2_std

    # 하이라이트 인덱스 구성 (기본값 제거하고, 코드북 표시가 있으면 샘플 하이라이트 비활성화)
    highlight_indices = [] if args.include_codebook else [
        i for i,(t,m) in enumerate(zip(Ticker_list, Months)) if (t in {'AAPL','NVDA','GOOG','MSFT','AMZN'} and m==1)
    ]

    # 출력: 스타일에 따라 분기
    out_path = Path(args.outdir)/'stage1_embeddings_sp500_2020_2023_tsne.png'

    if args.style == 'plotly':
        # Plotly 배경 산점도 저장
        plot_embeddings_plotly(
            H2, K_idx, Ticker_list, Years,
            highlight_indices=highlight_indices,
            K_total=config['vqvae'].get('num_embed', 512),
            out_png=out_path,
        )
        # 코드북 중 일부만 붉은 점으로 오버레이(별도 이미지 저장)
        if args.include_codebook and CB2 is not None:
            rng = np.random.RandomState(args.codebook_seed)
            n = CB2.shape[0]
            k = min(max(1, args.codebook_mark), n)
            sel = rng.choice(np.arange(n), size=k, replace=False)
            # Matplotlib을 사용해 기존 PNG 위에 추가 저장(간단 구현)
            fig, ax = plt.subplots(1,1, figsize=(10,8))
            ax.scatter(H2[:,0], H2[:,1], s=1, alpha=0.0)  # 축 스케일 맞춤용 빈 플롯
            ax.scatter(CB2[sel,0], CB2[sel,1], s=40, c='red', alpha=0.9, edgecolors='none')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            # 원본 좌표계와 동일 저장명 뒤에 언더스코어로 별도 저장
            out_path_cb = Path(args.outdir)/'stage1_embeddings_sp500_2020_2023_tsne_.png'
            plt.tight_layout(); plt.savefig(out_path_cb, dpi=300, bbox_inches='tight'); plt.close(fig)
            print(f"Saved codebook overlay to {out_path_cb}")
    else:
        # Matplotlib hexbin + scatter 오버레이
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_embeddings_hexbin(
            ax, H2, K_idx, Ticker_list, Years,
            highlight_indices=highlight_indices,
            K_total=config['vqvae'].get('num_embed', 512),
            gridsize=args.hexbin_gridsize,
            hex_cmap=args.hexbin_cmap,
        )
        if args.include_codebook and CB2 is not None:
            ax.scatter(CB2[:,0], CB2[:,1], s=30, c='red', alpha=0.9, edgecolors='none')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved PNG to {out_path}")

    # 코드북 사용량 리포트
    if args.report_usage:
        valid_codes = iterate_codes_over_loader(stage1, valid_p, config, device)
        test_codes  = iterate_codes_over_loader(stage1, test_p,  config, device)
        K_total = int(config['vqvae'].get('num_embed', 512))
        valid_unique = np.unique(valid_codes)
        test_unique  = np.unique(test_codes)
        print(f"Valid 사용 코드북 개수: {len(valid_unique)} / {K_total}")
        print(f"Test  사용 코드북 개수: {len(test_unique)} / {K_total}")

        # 빈도 그래프 저장
        def save_hist(codes, tag):
            fig2, ax2 = plt.subplots(1,1, figsize=(12,3))
            bins = np.arange(-0.5, K_total+0.5, 1)
            ax2.hist(codes, bins=bins, color='#4e79a7')
            ax2.set_xlim([-0.5, K_total-0.5])
            ax2.set_xlabel('Codebook Index (0..{})'.format(K_total-1))
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Codebook Utilization: {tag} (unique={len(np.unique(codes))})')
            outp = Path(args.outdir)/f'codebook_usage_{tag}.png'
            plt.tight_layout()
            plt.savefig(outp, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"Saved {outp}")
        if valid_codes.size > 0:
            save_hist(valid_codes, 'valid')
        if test_codes.size > 0:
            save_hist(test_codes, 'test')


if __name__ == '__main__':
    main()


