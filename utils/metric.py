import wandb
import pandas as pd
import numpy as np
import hashlib

def log_metrics_as_bar_chart(metrics_dict, model_name=None):
    """
    metrics_dict: 예) {
        'IC': 0.12, 
        'IC_IR': 0.08,
        'RankIC': 0.10,
        'RankIC_IR': 0.07
    }
    model_name: 모델을 구별하기 위한 이름 (예: "VQ128_rank0", "baseline_model" 등)
    
    이 딕셔너리 내용을 Table로 만들어 바 차트로 wandb에 로깅합니다.
    """
    
    # 1) 딕셔너리를 (Metric, Value) 형태의 2차원 리스트로 변환
    data = []
    for key, value in metrics_dict.items():
        # wandb가 float/int 등을 기대하므로, value가 numpy 타입이면 float()로 변환 권장
        data.append([key, float(value)])
    
    # 2) wandb.Table 객체 생성
    table = wandb.Table(data=data, columns=["Metric", "Value"])
    
    # 3) 모델 이름에 따라 차트 제목과 (짧은) 로그 키 설정
    if model_name:
        chart_title = f"Metrics Bar Chart - {model_name}"
        # W&B는 내부적으로 run-<id>-<key> 형태로 아티팩트를 생성하므로,
        # 128자 제한을 피하기 위해 key는 짧은 해시로 축약한다.
        short_token = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8]
        log_key = f"mbar_{short_token}"
    else:
        chart_title = "Metrics Bar Chart"
        log_key = "mbar"
    
    # 4) wandb.plot.bar를 통해 바 차트 생성
    bar_chart = wandb.plot.bar(
        table,      # wandb.Table 객체
        "Metric",   # x축 (카테고리)
        "Value",    # y축 (수치)
        title=chart_title
    )
    
    # 5) 최종 wandb 로깅 (모델별로 구별되는 키 사용)
    wandb.log({log_key: bar_chart})


def calculate_table_metrics(series, period, name, target_return=0):

    if period is not None:
        if type(period) == int:
            series = series[series.index.year == int(period)].copy()
            # series['return'] = series['return'] / series['return'].iloc[0]  
        elif type(period) == list:
            series = series.loc[period[0]:period[1]].copy()
    try:  
        daily_log_returns = series['return']
        cum_return = series['return'].cumsum()
    except:
        daily_log_returns = series
        cum_return = series.cumsum()
    normal_cum_return = np.exp(cum_return)
    
    # MDD 계산을 위해 누적 일반 리턴 사용
    max_cumulative_returns = normal_cum_return.cummax()
    drawdown = (normal_cum_return - max_cumulative_returns) / (max_cumulative_returns + 1e-9) 
    mdd = drawdown.min()

    # 연간 수익률 및 기타 지표 계산
    annual_return = daily_log_returns.mean() * 252
    annual_std = daily_log_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std

    # Sortino Ratio
    # Calculate downside deviation
    downside_returns = daily_log_returns[daily_log_returns < target_return]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - target_return) / downside_std if downside_std != 0 else np.nan
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(mdd) if mdd != 0 else np.nan

    # Turnover
    turnover = series['turnover'].mean()
    turnover = round(turnover, 4)
    
    result = {
        'Annualized Return': round(annual_return, 4),
        'Annual Std': round(annual_std, 4),
        'MDD': round(mdd, 4),
        'Sharpe Ratio': round(sharpe_ratio, 4),
        'Sortino Ratio': round(sortino_ratio, 4),
        'Calmar Ratio': round(calmar_ratio, 4),
        'Cumulative Returns': round(cum_return.iloc[-1], 4),
        'Turnover': turnover
    }

    return pd.DataFrame.from_dict(result, orient='index', columns=[f'{name}'])