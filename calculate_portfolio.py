import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
from utils import Cal_IC_IR, RankIC, calculate_portfolio_metrics
from utils import top_k_backtest

def top_k_backtest_clipped(pred_df, price_df, top_k=50, n_drop=5, rebalance_period=5, 
                          rebalance_strategy='dropout', open_cost=0.0005, close_cost=0.0015,
                          clip_return_min=-0.1, clip_return_max=0.1):
    """
    Implements a configurable Top-K backtesting strategy with return clipping.
    
    Parameters
    ----------
    ...
    clip_return_min : float
        Minimum daily return threshold (default: -20%)
    clip_return_max : float  
        Maximum daily return threshold (default: 20%)
    """
    dates = sorted(pred_df.index.get_level_values('datetime').unique())
    returns_list = []
    holdings = set()
    
    # Ensure price_df has the correct index for fast lookups
    price_df_indexed = price_df.set_index(['datetime', 'instrument'])

    for i, today in enumerate(dates[:-1]): # Go up to the second to last day
        tomorrow = dates[i+1]
        daily_return = 0.0

        if holdings:
            # Get today's and tomorrow's prices for current holdings
            try:
                today_prices = price_df_indexed.loc[today].reindex(list(holdings))['close']
                tomorrow_prices = price_df_indexed.loc[tomorrow].reindex(list(holdings))['close']
                
                # Calculate 1-day return
                valid_returns = (tomorrow_prices / today_prices - 1).dropna()
                if not valid_returns.empty:
                    daily_return = valid_returns.mean()
                    
                    # Apply return clipping
                    daily_return = np.clip(daily_return, clip_return_min, clip_return_max)
                    
            except (KeyError, IndexError):
                # Handle cases where price data might be missing for a specific day
                pass

        turnover = 0
        
        # Rebalance only on rebalancing days
        if i % rebalance_period == 0:
            daily_preds = pred_df.loc[today].sort_values('score', ascending=False)
            stocks_to_buy = set()
            stocks_to_sell = set()
            
            if rebalance_strategy == 'dropout':
                if not holdings:
                    stocks_to_buy = set(daily_preds.head(top_k).index)
                else:
                    held_stock_scores = daily_preds.reindex(list(holdings)).dropna()
                    stocks_to_sell = set(held_stock_scores.nsmallest(n_drop, 'score').index)
                    
                    buy_candidates_preds = daily_preds.drop(index=list(holdings), errors='ignore')
                    stocks_to_buy = set(buy_candidates_preds.head(len(stocks_to_sell)).index)
            
            elif rebalance_strategy == 'full':
                new_portfolio = set(daily_preds.head(top_k).index)
                stocks_to_buy = new_portfolio - holdings
                stocks_to_sell = holdings - new_portfolio
            
            else:
                raise ValueError(f"Unknown rebalance_strategy: {rebalance_strategy}")
            
            # Update holdings for the next day
            holdings.difference_update(stocks_to_sell)
            holdings.update(stocks_to_buy)
            turnover = len(stocks_to_buy) + len(stocks_to_sell)

        # Calculate transaction costs for today's trades
        cost = (turnover / (2 * top_k)) * (open_cost + close_cost) if top_k > 0 and turnover > 0 else 0
        net_return = daily_return - cost
        returns_list.append({'datetime': today, 'return': net_return})
        
    return pd.DataFrame(returns_list).set_index('datetime')['return'] if returns_list else pd.Series(dtype=float)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    # 데이터 로드
    pred_df = pd.read_pickle('benchmarks_output/FactorVAE/42/pred_label_42.pkl')
    price_df = pd.read_pickle('stock_data/features/kr_close_prices.pkl')
    
    # price_df 포맷 변환
    price_df = price_df.stack().reset_index()
    price_df.columns = ['datetime', 'instrument', 'close']
    price_df['datetime'] = pd.to_datetime(price_df['datetime'])
    price_df['instrument'] = price_df['instrument'].astype(str)
    
    # 백테스트용 score만 추출
    pred_df_for_backtest = pred_df[['score']].copy()
    
    print("=== FactorVAE 포트폴리오 메트릭 계산 (Return Clipping 적용) ===")
    print(f"예측 데이터 형태: {pred_df.shape}")
    print(f"가격 데이터 형태: {price_df.shape}")
    
    # IC 메트릭 계산
    pred_df_clean = pred_df.dropna()
    ic_metrics = Cal_IC_IR(pred_df_clean)
    rank_ic_metrics = RankIC(pred_df_clean)
    
    print("\n=== IC 메트릭 ===")
    print(ic_metrics)
    print("\n=== Rank IC 메트릭 ===")
    print(rank_ic_metrics)
    
    # 백테스트 실행 (기존)
    portfolio_returns_original = top_k_backtest(
        pred_df=pred_df_for_backtest,
        price_df=price_df,
        top_k=50,
        n_drop=5,
        rebalance_period=1,
        rebalance_strategy='dropout'
    )
    
    # 백테스트 실행 (Return Clipping 적용)
    portfolio_returns_clipped = top_k_backtest_clipped(
        pred_df=pred_df_for_backtest,
        price_df=price_df,
        top_k=50,
        n_drop=5,
        rebalance_period=5,
        rebalance_strategy='dropout',
        clip_return_min=-0.1,  # -10%로 제한
        clip_return_max=0.1    # +10%로 제한
    )
    
    # 포트폴리오 메트릭 계산
    portfolio_metrics_original = calculate_portfolio_metrics(portfolio_returns_original)
    portfolio_metrics_clipped = calculate_portfolio_metrics(portfolio_returns_clipped)
    
    print("\n=== 포트폴리오 메트릭 (기존) ===")
    print(portfolio_metrics_original)
    print("\n=== 포트폴리오 메트릭 (Return Clipping 적용) ===")
    print(portfolio_metrics_clipped)
    
    # 결과 저장
    output_path = Path('benchmarks_output/FactorVAE/42')
    ic_metrics.to_csv(output_path / "ic_metrics_new.csv")
    rank_ic_metrics.to_csv(output_path / "rank_ic_metrics_new.csv")
    portfolio_metrics_original.to_csv(output_path / "portfolio_metrics_original.csv")
    portfolio_metrics_clipped.to_csv(output_path / "portfolio_metrics_clipped.csv")
    portfolio_returns_original.to_pickle(output_path / "portfolio_returns_original.pkl")
    portfolio_returns_clipped.to_pickle(output_path / "portfolio_returns_clipped.pkl")
    
    # 누적 수익률 비교 플롯 저장
    try:
        cumulative_returns_original = (1 + portfolio_returns_original).cumprod()
        cumulative_returns_clipped = (1 + portfolio_returns_clipped).cumprod()
        
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        cumulative_returns_original.plot(title='Original: FactorVAE Cumulative Returns', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (1 + R)')
        plt.grid(True)
        plt.legend(['Original'])
        
        plt.subplot(2, 1, 2)
        cumulative_returns_clipped.plot(title='Clipped: FactorVAE Cumulative Returns (Return Clipped: ±10%)', color='red')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (1 + R)')
        plt.grid(True)
        plt.legend(['Return Clipped'])
        
        plt.tight_layout()
        plot_path = output_path / "cumulative_returns_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"누적 수익률 비교 플롯이 {plot_path}에 저장되었습니다.")
    except Exception as e:
        print(f"플롯 저장 실패: {e}")

    print(f"\n결과가 {output_path}에 저장되었습니다.")
    print("- ic_metrics_new.csv")
    print("- rank_ic_metrics_new.csv") 
    print("- portfolio_metrics_original.csv")
    print("- portfolio_metrics_clipped.csv")
    print("- portfolio_returns_original.pkl")
    print("- portfolio_returns_clipped.pkl")
    print("- cumulative_returns_comparison.png")