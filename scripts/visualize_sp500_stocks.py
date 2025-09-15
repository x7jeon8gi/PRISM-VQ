#!/usr/bin/env python3
"""
S&P 500 주요 종목 시각화 스크립트
논문 품질의 차트를 생성합니다.

주요 기능:
- S&P 500 주요 5개 종목 (AAPL, NVDA, GOOG, MSFT, AMZN) 차트
- 가격 차트, 수익률 차트, 변동성 차트
- 논문용 고품질 시각화
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import yfinance as yf

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# 한국어 폰트 설정 (논문용 영어 차트이므로 주석 처리)
# plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# 색상 팔레트 설정 (논문용) - 다양한 섹터
COLORS = {
    'AAPL': '#1f77b4',  # 블루 - 기술
    'JPM': '#ff7f0e',   # 오렌지 - 금융
    'JNJ': '#2ca02c',   # 그린 - 헬스케어
    'XOM': '#d62728',   # 레드 - 에너지
    'PG': '#9467bd',    # 퍼플 - 소비재
}

STOCK_NAMES = {
    'AAPL': 'Apple Inc.',           # 기술 (Technology)
    'JPM': 'JPMorgan Chase & Co.',  # 금융 (Financials)
    'JNJ': 'Johnson & Johnson',     # 헬스케어 (Healthcare)
    'XOM': 'Exxon Mobil Corp.',     # 에너지 (Energy)
    'PG': 'Procter & Gamble Co.'   # 소비재 (Consumer Goods)
}


def load_sp500_data_from_yahoo(tickers, start_date='2019-01-01', end_date='2024-12-31'):
    """
    Yahoo Finance에서 S&P 500 데이터를 로드합니다.
    """
    print(f"Yahoo Finance에서 데이터 다운로드 중: {tickers}")
    print(f"기간: {start_date} ~ {end_date}")
    
    try:
        # yfinance로 데이터 다운로드
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if len(tickers) == 1:
            # 단일 종목인 경우 컬럼 구조 조정
            price_df = pd.DataFrame({tickers[0]: data['Close']})
        else:
            # 다중 종목인 경우 Adjusted Close 가격만 사용
            price_df = data['Close']
        
        # NaN 값 제거
        price_df = price_df.dropna()
        
        print(f"데이터 로드 완료: {len(price_df)}개 거래일")
        print(f"종목: {list(price_df.columns)}")
        
        return price_df
        
    except Exception as e:
        print(f"Yahoo Finance 데이터 로드 중 오류: {e}")
        return None


def load_sp500_data(data_path):
    """
    로컬 데이터 파일에서 S&P 500 데이터를 로드합니다. (백업용)
    """
    try:
        # 먼저 pickle 파일 시도
        us_data_path = Path(data_path) / "US"
        
        # 가격 데이터 로드 시도
        price_file = us_data_path / "us_close_prices.pkl"
        if price_file.exists():
            print(f"로컬 가격 데이터 로드: {price_file}")
            price_df = pd.read_pickle(price_file)
            return price_df
        
        # CSV 파일에서 로드 시도
        csv_file = Path(data_path) / "[usa]_[all_themes]_[daily]_[vw_cap].csv"
        if csv_file.exists():
            print(f"로컬 CSV 데이터 로드: {csv_file}")
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # 필요한 종목만 필터링
            target_stocks = ['AAPL', 'NVDA', 'GOOG', 'MSFT', 'AMZN']
            df_filtered = df[df['symbol'].isin(target_stocks)].copy()
            
            # 피벗 테이블로 변환 (날짜 x 종목)
            price_df = df_filtered.pivot(index='date', columns='symbol', values='close')
            return price_df
            
    except Exception as e:
        print(f"로컬 데이터 로드 중 오류: {e}")
        return None
    
    return None


def generate_sample_data():
    """
    샘플 데이터를 생성합니다 (실제 데이터가 없는 경우).
    """
    print("샘플 데이터를 생성합니다...")
    
    # 2020-01-01부터 2024-12-31까지의 날짜 범위
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2024-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 주말 제외 (영업일만)
    dates = dates[dates.dayofweek < 5]
    
    np.random.seed(42)  # 재현 가능한 결과를 위해
    
    # 각 종목의 초기 가격과 특성
    stock_params = {
        'AAPL': {'initial': 300, 'drift': 0.0003, 'volatility': 0.025},
        'NVDA': {'initial': 240, 'drift': 0.0008, 'volatility': 0.035},
        'GOOG': {'initial': 1400, 'drift': 0.0002, 'volatility': 0.022},
        'MSFT': {'initial': 160, 'drift': 0.0004, 'volatility': 0.020},
        'AMZN': {'initial': 1800, 'drift': 0.0001, 'volatility': 0.028},
    }
    
    data = {}
    
    for symbol, params in stock_params.items():
        # 기하 브라운 운동으로 주가 시뮬레이션
        n_days = len(dates)
        dt = 1/252  # 1 영업일
        
        # 랜덤 워크 생성
        random_walk = np.random.normal(0, 1, n_days)
        
        # 가격 경로 계산
        log_returns = params['drift'] * dt + params['volatility'] * np.sqrt(dt) * random_walk
        log_returns[0] = 0  # 첫날은 변화 없음
        
        # 누적합으로 로그 가격 계산
        log_prices = np.log(params['initial']) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        
        # 일부 트렌드와 이벤트 추가
        if symbol == 'NVDA':
            # NVIDIA의 AI 붐 시뮬레이션 (2023년부터 급상승)
            ai_boom_start = pd.Timestamp('2023-01-01')
            ai_mask = dates >= ai_boom_start
            prices[ai_mask] *= np.exp(np.linspace(0, 1.2, ai_mask.sum()))
        
        if symbol == 'AAPL':
            # Apple의 주기적 제품 출시 효과
            for year in [2020, 2021, 2022, 2023, 2024]:
                event_date = pd.Timestamp(f'{year}-09-15')  # 9월 이벤트
                event_idx = np.where(dates >= event_date)[0]
                if len(event_idx) > 0:
                    idx = event_idx[0]
                    if idx < len(prices):
                        prices[idx:idx+30] *= 1.05  # 5% 상승 효과
        
        data[symbol] = prices
    
    # DataFrame 생성
    price_df = pd.DataFrame(data, index=dates)
    return price_df


def calculate_returns(price_df):
    """
    가격 데이터로부터 로그 수익률을 계산합니다.
    """
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return log_returns


def calculate_rolling_volatility(returns, window=20):
    """
    롤링 변동성을 계산합니다.
    """
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 연율화
    return volatility


def plot_price_charts(price_df, save_dir):
    """
    가격 차트를 생성합니다.
    """
    stocks = list(price_df.columns)
    
    # 개별 가격 차트 - 정확히 (N,1) 구성
    fig, axes = plt.subplots(len(stocks), 1, figsize=(10, 3*len(stocks)))
    # fig.suptitle('S&P 500 Major Stocks - Individual Price Charts', 
    #            fontsize=16, fontweight='bold', y=0.98)
    
    if len(stocks) == 1:
        axes = [axes]  # 단일 종목인 경우 리스트로 변환
    
    for i, stock in enumerate(stocks):
        ax = axes[i]
        
        # 가격 차트
        ax.plot(price_df.index, price_df[stock], 
                color=COLORS[stock], linewidth=2.5, alpha=0.9)
        
        # 영역 채우기
        ax.fill_between(price_df.index, price_df[stock].min(), price_df[stock],
                       color=COLORS[stock], alpha=0.3)
        
        ax.set_title(f'{STOCK_NAMES[stock]} ({stock})', fontweight='bold', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=10)
        
        # 테두리 제거
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 그리드 (매우 연하게)
        ax.grid(True, alpha=0.2)
        
        # x축 날짜 포맷팅
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # 최고점과 최저점 표시
        # max_idx = price_df[stock].idxmax()
        # min_idx = price_df[stock].idxmin()
        # max_price = price_df[stock].max()
        # min_price = price_df[stock].min()
        
        # ax.scatter(max_idx, max_price, color='green', s=50, zorder=5, alpha=0.8)
        # ax.scatter(min_idx, min_price, color='red', s=50, zorder=5, alpha=0.8)
        
        # 가격 범위 텍스트
        # price_range = f"Range: ${min_price:.1f} - ${max_price:.1f}"
        # ax.text(0.02, 0.98, price_range, transform=ax.transAxes, 
        #         verticalalignment='top', fontsize=9, 
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 마지막 종목이 아닌 경우 x축 레이블 숨기기
        if i < len(stocks) - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_individual_price_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 전체 비교 차트
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # 정규화된 가격 (시작점을 100으로)
    normalized_prices = price_df.div(price_df.iloc[0]) * 100
    
    for stock in stocks:
        ax2.plot(normalized_prices.index, normalized_prices[stock],
                color=COLORS[stock], linewidth=2, label=f'{stock}')
    
    ax2.set_title('S&P 500 Major Stocks - Normalized Price Comparison (Base=100)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # 테두리 제거
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left', fontsize=10)
    
    # x축 포맷팅
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_normalized_price_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"가격 차트 저장 완료: {save_dir}")


def plot_returns_analysis(returns_df, save_dir):
    """
    수익률 분석 차트를 생성합니다.
    """
    # 1. 일일 수익률 히스토그램
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('S&P 500 Major Stocks - Daily Returns Distribution', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    stocks = list(returns_df.columns)
    for i, stock in enumerate(stocks):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        # 히스토그램과 KDE
        returns_stock = returns_df[stock].dropna()
        ax.hist(returns_stock, bins=50, alpha=0.7, color=COLORS[stock], density=True)
        
        # 통계 정보
        mean_ret = returns_stock.mean()
        std_ret = returns_stock.std()
        skew = returns_stock.skew()
        kurt = returns_stock.kurtosis()
        
        ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.4f}')
        ax.axvline(mean_ret + std_ret, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(mean_ret - std_ret, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_title(f'{stock} - Daily Returns', fontweight='bold')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 통계 텍스트
        stats_text = f'Std: {std_ret:.4f}\nSkew: {skew:.2f}\nKurt: {kurt:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 마지막 빈 subplot 제거
    if len(stocks) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 누적 로그 수익률 차트 - 개별 차트 (N,1) 구성
    cumulative_log_returns = returns_df.cumsum()  # 로그 수익률의 누적합
    
    # 개별 누적 로그 수익률 차트 - 정확히 (N,1) 구성
    fig2, axes = plt.subplots(len(stocks), 1, figsize=(10, 3*len(stocks)))
    fig2.suptitle('S&P 500 Major Stocks - Individual Cumulative Log Returns', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    if len(stocks) == 1:
        axes = [axes]  # 단일 종목인 경우 리스트로 변환
    
    for i, stock in enumerate(stocks):
        ax = axes[i]
        
        # 누적 로그 수익률 플롯
        ax.plot(cumulative_log_returns.index, cumulative_log_returns[stock],
                color=COLORS[stock], linewidth=2.5, alpha=0.9)
        
        # 영역 채우기 (0 기준)
        ax.fill_between(cumulative_log_returns.index, 0, cumulative_log_returns[stock],
                       color=COLORS[stock], alpha=0.3)
        
        # 기준선 (0.0) 표시
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # 제목과 레이블
        final_log_return = cumulative_log_returns[stock].iloc[-1]
        total_return_pct = (np.exp(final_log_return) - 1) * 100
        ax.set_title(f'{STOCK_NAMES[stock]} ({stock}) - Total Return: {total_return_pct:+.1f}%', 
                    fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Log Return', fontsize=10)
        
        # 테두리 제거
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 그리드 (매우 연하게)
        ax.grid(True, alpha=0.2)
        
        # x축 포맷팅
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # 최고점과 최저점 표시
        max_idx = cumulative_log_returns[stock].idxmax()
        min_idx = cumulative_log_returns[stock].idxmin()
        max_val = cumulative_log_returns[stock].max()
        min_val = cumulative_log_returns[stock].min()
        
        ax.scatter(max_idx, max_val, color='green', s=50, zorder=5, alpha=0.8)
        ax.scatter(min_idx, min_val, color='red', s=50, zorder=5, alpha=0.8)
        
        # 통계 정보 텍스트
        stats_text = f'Max: {max_val:.3f}\nMin: {min_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 마지막 종목이 아닌 경우 x축 레이블 숨기기
        if i < len(stocks) - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Date', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_individual_cumulative_log_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 추가: 전체 비교 차트도 별도로 저장
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    for stock in stocks:
        ax3.plot(cumulative_log_returns.index, cumulative_log_returns[stock],
                color=COLORS[stock], linewidth=2, label=f'{stock}')
    
    ax3.set_title('S&P 500 Major Stocks - Cumulative Log Returns Comparison', 
                  fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative Log Return', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    
    # 테두리 제거
    for spine in ax3.spines.values():
        spine.set_visible(False)
    
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper left', fontsize=10)
    
    # x축 포맷팅
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_cumulative_log_returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"수익률 분석 차트 저장 완료: {save_dir}")


def plot_volatility_analysis(returns_df, save_dir):
    """
    변동성 분석 차트를 생성합니다.
    """
    # 롤링 변동성 계산 (20일, 60일)
    vol_20 = calculate_rolling_volatility(returns_df, 20)
    vol_60 = calculate_rolling_volatility(returns_df, 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('S&P 500 Major Stocks - Rolling Volatility (Annualized)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    stocks = list(returns_df.columns)
    for i, stock in enumerate(stocks):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        # 20일, 60일 롤링 변동성
        ax.plot(vol_20.index, vol_20[stock], color=COLORS[stock], 
                linewidth=1.5, label='20-day', alpha=0.8)
        ax.plot(vol_60.index, vol_60[stock], color=COLORS[stock], 
                linewidth=2, linestyle='--', label='60-day', alpha=0.9)
        
        ax.set_title(f'{stock} - Rolling Volatility', fontweight='bold')
        ax.set_ylabel('Annualized Volatility')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # x축 포맷팅
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
        
        # 평균 변동성 표시
        mean_vol = vol_20[stock].mean()
        ax.axhline(mean_vol, color='red', linestyle=':', alpha=0.7)
        ax.text(0.02, 0.98, f'Avg: {mean_vol:.2f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 마지막 빈 subplot 제거
    if len(stocks) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_rolling_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"변동성 분석 차트 저장 완료: {save_dir}")


def plot_correlation_analysis(returns_df, save_dir):
    """
    상관관계 분석 차트를 생성합니다.
    """
    # 상관관계 매트릭스
    corr_matrix = returns_df.corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 히트맵
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Daily Returns Correlation Matrix', fontweight='bold', fontsize=12)
    
    # 시간에 따른 롤링 상관관계 (AAPL vs others)
    base_stock = 'AAPL'
    rolling_window = 60
    
    for stock in returns_df.columns:
        if stock != base_stock:
            rolling_corr = returns_df[base_stock].rolling(rolling_window).corr(returns_df[stock])
            ax2.plot(rolling_corr.index, rolling_corr, 
                    color=COLORS[stock], linewidth=2, label=f'{base_stock} vs {stock}')
    
    ax2.set_title(f'Rolling Correlation with {base_stock} (60-day window)', 
                  fontweight='bold', fontsize=12)
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # x축 포맷팅
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sp500_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"상관관계 분석 차트 저장 완료: {save_dir}")


def create_summary_dashboard(price_df, returns_df, save_dir):
    """
    종합 대시보드를 생성합니다.
    """
    # 5개 종목을 위해 5컬럼으로 설정
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.2)
    
    # 제목
    fig.suptitle('S&P 500 Major Stocks - Comprehensive Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. 정규화된 가격 비교 (상단 전체)
    ax1 = fig.add_subplot(gs[0, :])
    normalized_prices = price_df.div(price_df.iloc[0]) * 100
    
    for stock in price_df.columns:
        ax1.plot(normalized_prices.index, normalized_prices[stock],
                color=COLORS[stock], linewidth=2.5, label=f'{stock}')
    
    ax1.set_title('Normalized Price Comparison (Base=100)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left', fontsize=11, ncol=5)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 2. 개별 주식 통계 (중간 행)
    stats_data = []
    for i, stock in enumerate(price_df.columns):
        ax = fig.add_subplot(gs[1, i])
        
        # 기본 통계
        returns_stock = returns_df[stock].dropna()
        total_return = (price_df[stock].iloc[-1] / price_df[stock].iloc[0] - 1) * 100
        annual_return = returns_stock.mean() * 252 * 100
        annual_vol = returns_stock.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        max_drawdown = ((price_df[stock] / price_df[stock].cummax()) - 1).min() * 100
        
        # 바 차트로 주요 지표 표시
        metrics = ['Total\nReturn', 'Annual\nReturn', 'Annual\nVol', 'Sharpe\nRatio', 'Max\nDrawdown']
        values = [total_return, annual_return, annual_vol, sharpe_ratio, max_drawdown]
        colors_bar = ['green' if v > 0 else 'red' for v in values[:-1]] + ['red']
        
        bars = ax.bar(range(len(metrics)), values, color=colors_bar, alpha=0.7)
        ax.set_title(f'{stock}', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')
        
        stats_data.append({
            'Stock': stock,
            'Total Return (%)': f'{total_return:.1f}',
            'Annual Return (%)': f'{annual_return:.1f}',
            'Annual Vol (%)': f'{annual_vol:.1f}',
            'Sharpe Ratio': f'{sharpe_ratio:.2f}',
            'Max Drawdown (%)': f'{max_drawdown:.1f}'
        })
    
    # 3. 상관관계 히트맵 (하단 왼쪽 3컬럼)
    ax3 = fig.add_subplot(gs[2, :3])
    corr_matrix = returns_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Daily Returns Correlation', fontweight='bold', fontsize=12)
    
    # 4. 변동성 비교 (하단 오른쪽 2컬럼)
    ax4 = fig.add_subplot(gs[2, 3:])
    vol_data = []
    for stock in returns_df.columns:
        vol_20 = calculate_rolling_volatility(returns_df[[stock]], 20)
        vol_data.append(vol_20[stock].dropna())
    
    ax4.boxplot(vol_data, labels=returns_df.columns, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax4.set_title('Volatility Distribution (20-day Rolling)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Annualized Volatility')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(save_dir / 'sp500_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 통계 테이블 저장
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(save_dir / 'sp500_summary_statistics.csv', index=False)
    
    print(f"종합 대시보드 저장 완료: {save_dir}")
    print("\n=== 주요 통계 ===")
    print(stats_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='S&P 500 주요 종목 시각화')
    parser.add_argument('--data_path', type=str, 
                       default=str(ROOT / 'dataset' / 'data'),
                       help='로컬 데이터 경로 (백업용)')
    parser.add_argument('--output_dir', type=str,
                       default=str(ROOT / 'plot_cache' / 'sp500_stocks'),
                       help='출력 디렉토리')
    parser.add_argument('--use_sample', action='store_true',
                       help='샘플 데이터 사용')
    parser.add_argument('--use_local', action='store_true',
                       help='로컬 데이터 파일 우선 사용')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-12-31',
                       help='종료 날짜 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("S&P 500 주요 종목 시각화 시작")
    print("=" * 60)
    
    # 타겟 종목 정의 - 다양한 섹터
    target_stocks = ['AAPL', 'JPM', 'JNJ', 'XOM', 'PG']
    
    # 데이터 로드 우선순위
    price_df = None
    
    if args.use_sample:
        print("샘플 데이터를 사용합니다.")
        price_df = generate_sample_data()
    elif args.use_local:
        print("로컬 데이터를 우선 사용합니다.")
        price_df = load_sp500_data(args.data_path)
        if price_df is None:
            print("로컬 데이터를 찾을 수 없습니다. Yahoo Finance에서 다운로드합니다.")
            price_df = load_sp500_data_from_yahoo(target_stocks, args.start_date, args.end_date)
    else:
        # 기본: Yahoo Finance 사용
        print("Yahoo Finance에서 실시간 데이터를 가져옵니다.")
        price_df = load_sp500_data_from_yahoo(target_stocks, args.start_date, args.end_date)
        
        # Yahoo Finance 실패시 로컬 데이터 시도
        if price_df is None:
            print("Yahoo Finance 접근 실패. 로컬 데이터를 시도합니다.")
            price_df = load_sp500_data(args.data_path)
    
    # 최종 백업: 샘플 데이터
    if price_df is None:
        print("모든 실제 데이터 로드 실패. 샘플 데이터를 생성합니다.")
        price_df = generate_sample_data()
    
    # 타겟 종목만 필터링
    available_stocks = [stock for stock in target_stocks if stock in price_df.columns]
    
    if not available_stocks:
        print("타겟 종목을 찾을 수 없습니다. 사용 가능한 모든 종목을 사용합니다.")
        available_stocks = list(price_df.columns)
    
    price_df = price_df[available_stocks].dropna()
    print(f"\n분석 대상 종목: {available_stocks}")
    print(f"데이터 기간: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"총 데이터 포인트: {len(price_df)}")
    
    # 수익률 계산
    returns_df = calculate_returns(price_df)
    
    # 차트 생성
    print("\n차트 생성 중...")
    plot_price_charts(price_df, save_dir)
    plot_returns_analysis(returns_df, save_dir)
    plot_volatility_analysis(returns_df, save_dir)
    plot_correlation_analysis(returns_df, save_dir)
    create_summary_dashboard(price_df, returns_df, save_dir)
    
    print(f"\n모든 차트가 {save_dir}에 저장되었습니다.")
    print("\n생성된 파일:")
    for file in save_dir.glob('*.png'):
        print(f"  - {file.name}")
    for file in save_dir.glob('*.csv'):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
