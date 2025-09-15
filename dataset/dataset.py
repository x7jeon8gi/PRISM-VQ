import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np 
import logging
import pandas as pd
import copy
from torch.utils.data.dataloader import default_collate

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

        self.index_df = self.data_source.get_index()
        datetime_level = self.index_df.names.index('datetime') # 'datetime' 레벨 위치 찾기
        daily_groups = pd.Series(self.index_df.values).groupby(self.index_df.get_level_values(datetime_level))

        self.daily_count = daily_groups.size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        # 날짜 순서 보장을 위해 unique dates 저장
        self.dates = daily_groups.groups.keys() # 정렬된 날짜 리스트

    def __iter__(self):
        date_indices = np.arange(len(self.dates))
        if self.shuffle:
            np.random.shuffle(date_indices)

        # 전체 데이터셋에서의 실제 인덱스 번호(정수 위치)를 yield 해야 함
        # self.index_df 를 기준으로 각 날짜에 해당하는 정수 인덱스를 찾아야 함
        datetime_level = self.index_df.names.index('datetime')
        all_datetimes = self.index_df.get_level_values(datetime_level)

        for i in date_indices:
            target_date = list(self.dates)[i] # 접근 방식 수정 필요할 수 있음 (dates 타입 확인)
            # 해당 날짜를 가진 모든 샘플의 *정수 위치* 인덱스 찾기
            indices_for_date = np.where(all_datetimes == target_date)[0]
            if len(indices_for_date) != self.daily_count[i]:
                print(f"Warning: Index count mismatch for date {target_date}. Expected {self.daily_count[i]}, Found {len(indices_for_date)}")
            yield indices_for_date # 해당 날짜의 인덱스 배열 자체를 yield

    def __len__(self):
        return len(self.daily_count) # len(self.data_source)
 

def init_data_loader(handler, shuffle, num_workers=0, index=False):
    sampler = DailyBatchSamplerRandom(handler, shuffle)
    num_batches_per_epoch = len(sampler)
    # 모든 데이터를 float 타입으로 변환하는 collate 함수
    def float_collate_fn(batch):
        batch = default_collate(batch)
        if isinstance(batch, torch.Tensor):
            return batch.float()
        return batch
    
    data_loader = DataLoader(handler,
                             batch_sampler=sampler,
                             pin_memory=True,
                             num_workers=num_workers,
                             drop_last=False,
                             collate_fn=float_collate_fn)  # float 변환 collate 함수 적용
    
    if index == True:
        return data_loader, handler, num_batches_per_epoch
    else:
        return data_loader, num_batches_per_epoch

    # data_loader.dataset 으로 확인 가능

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

def drop_duplicates(x, tolerance=1e-10):
    """
    각 주식의 8일치 특성 데이터에서 중복 행(완전히 동일한 행)이 존재하면 해당 주식을 삭제할 수 있도록 마스크를 반환.
    벡터화된 방식으로 각 행에 대해 해시 값을 계산하여 중복 검사를 수행합니다.
    
    Args:
        x: Tensor, shape (N, T, F)
            N: 주식 수
            T: 시계열 길이 (예: 8)
            F: 특성 차원 (예: 221)
        tolerance: float, 부동소수점 연산의 오차를 고려해 반올림할 단위
        
    Returns:
        mask: Tensor of bool, shape (N,)
            중복 행이 없는 주식은 True, 중복이 있으면 False
    """
    N, T, F = x.shape
    device = x.device
    
    # 1. 각 주식의 각 행에 대해 튜플 형태의 해시를 구하는 대신, 부동소수점 값의 특성을 반올림하여 벡터화된 해시값을 계산합니다.
    # weights: 고정 가중치 벡터(예: 1,2,...,F)를 사용해서 각 행을 하나의 스칼라 값으로 매핑합니다.
    weights = torch.arange(1, F+1, device=device, dtype=x.dtype)
    
    # 먼저, tolerance를 고려해 반올림합니다.
    x_rounded = torch.round(x / tolerance)  # shape (N, T, F)
    # 각 행의 해시값: row_hash의 shape는 (N, T)
    row_hash = torch.sum(x_rounded * weights, dim=-1)
    
    # 2. 각 주식별로 row_hash를 정렬한 후, 인접한 값이 같으면 중복 있는 것으로 판단합니다.
    sorted_hash, _ = torch.sort(row_hash, dim=1)  # shape (N, T)
    # 인접한 해시값의 차이가 0이면 중복 (여기서는 정수 비교이므로 tolerance 문제가 없음)
    dup_flags = (sorted_hash[:, 1:] == sorted_hash[:, :-1])
    # 만약 한 주식 내에서 하나라도 중복이 있다면, 그 주식은 False로 처리
    mask = ~torch.any(dup_flags, dim=1)
    
    return mask