import time
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
import gc
import qlib
from qlib.contrib.data.handler import Alpha158
import pickle
from utils import load_yaml_param_settings, load_args, get_root_dir, seed_everything
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP
from dataset.dataset import init_data_loader
from pathlib import Path
import pickle as pkl


def load_group_meta(base_path: str, prefix: str, universe: str):
    """
    Load dataframe metadata to infer group sizes and column order.
    Tries vqvae first, then others.
    Returns: (tag, order, sizes)
        tag: 'vqvae' or 'others'
        order: list like ['feature','prior','label']
        sizes: dict with counts per group
    """
    base = Path(base_path) if base_path else Path('dataset/data')
    candidates = [
        base / prefix / f"{universe}_vqvae_dataframe_learn.pkl",
        base / prefix / f"{universe}_others_dataframe_learn.pkl",
    ]
    for df_path in candidates:
        if df_path.exists():
            tag = 'vqvae' if 'vqvae' in df_path.name else 'others'
            df = pd.read_pickle(df_path)
            if not isinstance(df.columns, pd.MultiIndex):
                raise ValueError(f"Expected MultiIndex columns, got {type(df.columns)} at {df_path}")
            top = [c[0] for c in df.columns]
            order = []
            sizes = {}
            for g in top:
                if g not in order:
                    order.append(g)
                sizes[g] = sizes.get(g, 0) + 1
            return tag, order, sizes
    raise FileNotFoundError("No dataframe pickle found for vqvae/others under " + str(base / prefix))


def load_prepared_datasets(base_path: str, prefix: str, universe: str, window_size: int):
    """
    Load prepared TSDataSampler pickles. Prefer vqvae, fallback to others.
    Returns: (tag, train, valid, test)
    """
    base = Path(base_path) if base_path else Path('dataset/data')
    for tag in ['vqvae', 'others']:
        try:
            with open(base / prefix / f"{universe}_{tag}_{window_size}_dl_train.pkl", 'rb') as f:
                train = pkl.load(f)
            with open(base / prefix / f"{universe}_{tag}_{window_size}_dl_valid.pkl", 'rb') as f:
                valid = pkl.load(f)
            with open(base / prefix / f"{universe}_{tag}_{window_size}_dl_test.pkl", 'rb') as f:
                test = pkl.load(f)
            return tag, train, valid, test
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No prepared dataset pickles found for {prefix}/{universe} (vqvae/others)")


def check_prior_consistency(batch: torch.Tensor, n_feature: int, n_prior: int, atol: float = 1e-6):
    """
    Verify that for each timestep, all stocks share the same prior values.
    batch shape: (N_stocks, T, F_total)
    """
    if n_prior <= 0:
        return True, 0.0
    prior = batch[:, :, n_feature:n_feature + n_prior]
    # std across stock dimension; ignore NaNs (use numpy for nan-aware ops)
    prior_np = prior.detach().cpu().numpy()
    # Check only the last time step (batch sampler groups by same date)
    last_step = prior_np[:, -1, :]  # (N, n_prior)
    diffs = np.nanstd(last_step, axis=0)  # (n_prior,)
    max_diff = np.nanmax(diffs) if diffs.size else 0.0
    is_consistent = bool(max_diff <= atol)
    return is_consistent, float(max_diff)


def check_column_order(batch: torch.Tensor, order: list, sizes: dict):
    """
    Confirm that concatenation order is feature, prior, label.
    We verify sizes sum and that label is last slice.
    """
    total = batch.shape[-1]
    n_feature = sizes.get('feature', 0)
    n_prior = sizes.get('prior', 0)
    n_label = sizes.get('label', 0)
    # sum check
    if n_feature + n_prior + n_label != total:
        return False, f"Size mismatch: feature({n_feature}) + prior({n_prior}) + label({n_label}) != total({total})"
    # order check
    expected = ['feature', 'prior', 'label']
    if order[:3] != expected:
        return False, f"Column group order {order[:3]} != {expected}"
    # sanity: label last slice has small size (typically 1)
    if n_label < 1:
        return False, f"Invalid label size: {n_label}"
    return True, "OK"


def detect_prior_columns(batch: torch.Tensor, atol: float = 1e-6):
    """
    Heuristically detect prior columns: columns whose values are identical across stocks at each timestep.
    Returns sorted list of prior column indices.
    """
    N, T, F = batch.shape
    # compute std across N for each (t, f)
    batch_np = batch.detach().cpu().numpy()
    std_nf = np.nanstd(batch_np, axis=0)  # (T, F)
    # columns where std == 0 for all timesteps
    mask = np.all(std_nf <= atol, axis=0)  # (F,)
    prior_idx = list(np.where(mask)[0])
    return sorted(prior_idx)


def check_nan_rates(batch: torch.Tensor, n_feature: int, n_prior: int, n_label: int):
    feat = batch[:, :, :n_feature]
    prior = batch[:, :, n_feature:n_feature + n_prior]
    label = batch[:, :, n_feature + n_prior:]
    def rate(x):
        total = x.numel()
        n_nan = torch.isnan(x).sum().item()
        return n_nan / max(total, 1)
    return rate(feat), rate(prior), rate(label)

def find_inconsistent_market_data(data_loader, dataset_name="Unknown"):
    """
    DataLoader에서 첫 번째 배치를 가져와 market 데이터의 불일치를 확인합니다.
    """
    print(f"\n=== {dataset_name} 데이터 분석 ===")
    
    # 데이터 로더에서 배치 하나를 가져옴
    batch = next(iter(data_loader))
    
    print(f"원본 배치 형태: {batch.shape}")
    
    # 제공해주신 로직으로 데이터 슬라이싱
    batch = batch.squeeze(0)
    firm_char = batch[:, : ,0:158]
    y = batch[:, :, -1]
    market = batch[:, :, 158:158+13]
    
    print(f"squeeze 후 배치 형태 (회사 수, 시점, 피처 수): {batch.shape}")
    print(f"회사별 특성 데이터 형태: {firm_char.shape}")
    print(f"타겟 데이터 형태: {y.shape}")
    print(f"마켓 데이터 형태: {market.shape}")
    
    num_firms = market.shape[0]
    num_timesteps = market.shape[1]
    
    if num_firms < 2:
        print("회사가 1개뿐이라 비교할 수 없습니다.")
        return

    # 첫 번째 회사의 market 데이터를 기준점으로 설정
    base_market_data = market[0] 
    
    print(f"\n기준 회사(인덱스 0)의 market 데이터 샘플:")
    print(f"  첫 번째 시점: {base_market_data[0, :5]}")
    print(f"  마지막 시점: {base_market_data[-1, :5]}")
    
    # 다른 모든 회사와 market 데이터 비교
    is_consistent = True
    inconsistent_firms = []
    
    for i in range(1, num_firms):
        # torch.allclose는 부동소수점 오차를 감안하여 비교해줍니다.
        if not torch.allclose(market[i], base_market_data, atol=1e-6):
            is_consistent = False
            inconsistent_firms.append(i)
            
            print(f"\n❗️ 회사 {i}번의 market 데이터가 기준과 다릅니다:")
            print(f"  첫 번째 시점: {market[i, 0, :5]}")
            print(f"  마지막 시점: {market[i, -1, :5]}")
            
            # 어느 시점에서 차이가 나는지 확인
            diff_mask = ~torch.isclose(market[i], base_market_data, atol=1e-6)
            diff_timesteps = torch.any(diff_mask, dim=1).nonzero().flatten()
            
            if len(diff_timesteps) > 0:
                print(f"  차이가 나는 시점들: {diff_timesteps[:10].tolist()}")  # 처음 10개만 출력
                
            # 첫 번째 차이점에서 상세 분석
            if len(diff_timesteps) > 0:
                t = diff_timesteps[0].item()
                diff_features = diff_mask[t].nonzero().flatten()
                print(f"  시점 {t}에서 차이나는 feature들: {diff_features[:5].tolist()}")
                print(f"    기준값: {base_market_data[t, diff_features[:5]]}")
                print(f"    현재값: {market[i, t, diff_features[:5]]}")
            
            if len(inconsistent_firms) >= 3:  # 너무 많이 출력하지 않도록 제한
                print(f"  ... (총 {num_firms-1}개 회사 중 더 많은 불일치 있음)")
                break
            
    if is_consistent:
        print(f"\n✅ 해당 배치에서 market 데이터는 모두 일치합니다.")
    else:
        print(f"\n❌ 총 {len(inconsistent_firms)}개 회사에서 market 데이터 불일치 발견")
        
    return is_consistent, inconsistent_firms

def analyze_data_loader_sampling(data_loader, dataset_name="Unknown"):
    """
    데이터 로더의 샘플링 로직을 분석합니다.
    """
    print(f"\n=== {dataset_name} 데이터 로더 샘플링 분석 ===")
    
    # 샘플러 정보 확인
    sampler = data_loader.sampler
    batch_sampler = data_loader.batch_sampler
    
    print(f"샘플러 타입: {type(sampler).__name__}")
    print(f"배치 샘플러 타입: {type(batch_sampler).__name__ if batch_sampler else 'None'}")
    
    # 실제 사용되는 샘플러 확인 (batch_sampler가 있으면 그것을 사용)
    effective_sampler = batch_sampler if batch_sampler else sampler
    
    # 셔플 여부 확인 (다양한 sampler 타입에 대응)
    if hasattr(effective_sampler, 'shuffle'):
        print(f"셔플 여부: {effective_sampler.shuffle}")
    elif hasattr(effective_sampler, 'replacement'):
        print(f"샘플러 정보: replacement={effective_sampler.replacement}")
    else:
        print("셔플 정보를 확인할 수 없습니다.")
    
    print(f"총 배치 수: {len(data_loader)}")
    
    # 첫 번째 배치의 인덱스 정보 확인
    if batch_sampler:
        # batch_sampler가 있는 경우
        sampler_iter = iter(batch_sampler)
        first_batch_indices = next(sampler_iter)
        print(f"첫 번째 배치 인덱스 수: {len(first_batch_indices)}")
        print(f"첫 번째 배치 인덱스 샘플: {first_batch_indices[:10]}")
        
        # DailyBatchSamplerRandom인 경우 추가 정보
        if hasattr(batch_sampler, 'data_source'):
            data_source = batch_sampler.data_source
            if hasattr(data_source, 'get_index'):
                index_df = data_source.get_index()
                print(f"데이터 소스 인덱스 형태: {index_df.shape}")
                print(f"인덱스 레벨: {index_df.names}")
                
                # 날짜별 그룹 정보
                datetime_level = index_df.names.index('datetime')
                dates = index_df.get_level_values(datetime_level).unique()
                print(f"고유 날짜 수: {len(dates)}")
                print(f"날짜 범위: {dates[0]} ~ {dates[-1]}")
                
                # 첫 번째 배치에 해당하는 실제 인덱스들 확인
                all_datetimes = index_df.get_level_values(datetime_level)
                batch_dates = all_datetimes[first_batch_indices]
                unique_dates = batch_dates.unique()
                print(f"첫 번째 배치의 고유 날짜 수: {len(unique_dates)}")
                print(f"첫 번째 배치 날짜: {unique_dates}")
                
                if len(unique_dates) > 1:
                    print("⚠️  한 배치에 여러 날짜의 데이터가 섞여있습니다!")
                else:
                    print("✅ 한 배치는 단일 날짜의 데이터로만 구성되어 있습니다.")
    else:
        # 일반 sampler인 경우
        print("일반 샘플러를 사용 중입니다. 배치별 날짜 분석을 생략합니다.")

def load_and_check_data():
    """
    stage2_gpt_sweep.py와 동일한 방식으로 데이터를 로드하고 검증합니다.
    """
    # Load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    # Load dataset (prefer vqvae, fallback to others)
    pickle_path = config['data']['data_path']
    prefix = 'US'
    universe = 'sp500'
    window_size = 20

    try:
        tag, train_prepare, valid_prepare, test_prepare = load_prepared_datasets(pickle_path, prefix, universe, window_size)
        print(f"========== Loaded prepared datasets: tag={tag} ==========")
    except FileNotFoundError:
        print(f"Using Alpha158 handler with qlib data")
        data_handler_config = load_yaml_param_settings(args.data_handler_config)
        qlib.init(provider_uri=config['data'].get('provider_uri', "./qlib_data/cn_data"), region=data_handler_config['region'])
        dataset = Alpha158(**data_handler_config)

        segments = {
            'train': ['2009-01-01', '2019-12-31'],
            'valid': ['2020-01-01', '2021-12-31'],
            'test': ['2022-01-01', '2024-12-31'],
        }

        TsDataset = TSDatasetH(
            handler=dataset, 
            segments=segments, 
            step_len=window_size, 
        )

        train_prepare = TsDataset.prepare(segments='train')
        valid_prepare = TsDataset.prepare(segments='valid')
        test_prepare = TsDataset.prepare(segments='test')
        
        train_prepare.config(fillna_type='ffill+bfill')
        valid_prepare.config(fillna_type='ffill+bfill')
        test_prepare.config(fillna_type='ffill+bfill')

    # Create data loaders
    train_loader, _ = init_data_loader(train_prepare, shuffle=True, num_workers=0)
    valid_loader, _ = init_data_loader(valid_prepare, shuffle=False, num_workers=0)
    test_loader, _ = init_data_loader(test_prepare, shuffle=False, num_workers=0)
    
    # 데이터 로더 샘플링 로직 분석
    print("\n" + "="*70)
    print("=== 데이터 로더 샘플링 분석 ===")
    analyze_data_loader_sampling(train_loader, "훈련")
    analyze_data_loader_sampling(valid_loader, "검증") 
    analyze_data_loader_sampling(test_loader, "테스트")
    
    # 그룹 메타 로딩 (order, sizes)
    try:
        tag_meta, order, sizes = load_group_meta(pickle_path, prefix, universe)
        print(f"그룹 순서: {order}, 크기: {sizes}")
    except Exception as e:
        print(f"그룹 메타 로딩 실패: {e}")
        order, sizes = None, None

    def evaluate_loader(loader, name, order, sizes):
        # 첫 배치 로드
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if batch.dim() == 4 and batch.shape[0] == 1:
            batch = batch.squeeze(0)

        if sizes is not None:
            n_feature = sizes.get('feature', 0)
            n_prior = sizes.get('prior', 0)
            n_label = sizes.get('label', 0)
        else:
            # Fallback: detect prior columns heuristically, assume 1 label col
            prior_idx = detect_prior_columns(batch)
            total = batch.shape[-1]
            n_label = 1 if total - len(prior_idx) >= 1 else 0
            prior_start = min(prior_idx) if prior_idx else 0
            prior_end = max(prior_idx) if prior_idx else -1
            if prior_idx and prior_end - prior_start + 1 == len(prior_idx):
                n_feature = prior_start
                n_prior = len(prior_idx)
            else:
                n_feature = total - n_label
                n_prior = 0

        print("\n" + "="*70)
        print(f"=== {name} Prior 일관성 검증 ===")
        ok_prior, max_diff = check_prior_consistency(batch, n_feature, n_prior, atol=1e-5)
        print(f"Prior 일관성: {'✅' if ok_prior else '❌'} (최대 표준편차: {max_diff:.3e})")

        print("\n" + "="*70)
        print(f"=== {name} 컬럼 순서 검증 ===")
        if order is not None and sizes is not None:
            ok_order, msg = check_column_order(batch, order, sizes)
        else:
            total = batch.shape[-1]
            label_ok = n_label >= 1 and (n_feature + n_prior + n_label == total)
            ok_order = label_ok and (n_prior > 0)
            msg = 'Heuristic check' if ok_order else 'Heuristic order failed'
        print(f"컬럼 순서: {'✅' if ok_order else '❌'} - {msg}")

        print("\n" + "="*70)
        print(f"=== {name} NaN 비율 점검 ===")
        feat_nan, prior_nan, label_nan = check_nan_rates(batch, n_feature, n_prior, n_label)
        print(f"feature NaN 비율: {feat_nan:.5f}")
        print(f"prior   NaN 비율: {prior_nan:.5f}")
        print(f"label   NaN 비율: {label_nan:.5f}")

        ok_all = ok_prior and ok_order and prior_nan == 0.0
        return ok_all, ok_prior, ok_order, prior_nan, msg

    # 각 데이터셋 검증 수행
    ok_train, p_train, o_train, prior_nan_train, msg_train = evaluate_loader(train_loader, '훈련', order, sizes)
    ok_valid, p_valid, o_valid, prior_nan_valid, msg_valid = evaluate_loader(valid_loader, '검증', order, sizes)
    ok_test, p_test, o_test, prior_nan_test, msg_test = evaluate_loader(test_loader, '테스트', order, sizes)

    # 종합 결과
    print("\n" + "="*50)
    print("=== 전체 요약 ===")
    def line(name, ok, p, o, n, msg):
        print(f"{name}: {'✅' if ok else '❌'} | prior={'OK' if p else 'NG'}, order={'OK' if o else 'NG'}, prior_NaN={n:.5f} {'' if o else msg}")
    line('훈련', ok_train, p_train, o_train, prior_nan_train, msg_train)
    line('검증', ok_valid, p_valid, o_valid, prior_nan_valid, msg_valid)
    line('테스트', ok_test, p_test, o_test, prior_nan_test, msg_test)

if __name__ == "__main__":
    load_and_check_data()
