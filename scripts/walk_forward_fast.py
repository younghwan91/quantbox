"""
간소화된 Walk-Forward 분석

복잡한 백테스트 대신 Z-Score 시그널 기반의 빠른 분석
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from itertools import combinations, product

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from quantbox.core.config import Config
from quantbox.data.cache import DataCache
from quantbox.utils.numba_ops import calculate_rolling_zscore, calculate_half_life


def load_cached_data(cache: DataCache, symbols: List[str], interval: str = "15m") -> Dict[str, pd.DataFrame]:
    """캐시에서 데이터 로드"""
    data = {}
    
    for symbol in symbols:
        for key in cache.disk_cache._index.keys():
            if symbol in key and interval in key:
                entry = cache.disk_cache.get(key)
                if entry and entry.data is not None:
                    df = entry.data
                    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(c in df.columns for c in cols):
                        data[symbol] = df[cols]
                    break
    
    return data


def calculate_spread_stats(prices1: np.ndarray, prices2: np.ndarray, formation_period: int) -> Dict:
    """스프레드 통계 계산"""
    log_prices1 = np.log(prices1)
    log_prices2 = np.log(prices2)
    
    # OLS 헤지 비율
    X = log_prices2[:formation_period]
    y = log_prices1[:formation_period]
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    beta = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    
    # 스프레드
    spread = log_prices1 - beta * log_prices2
    
    # 형성 기간의 통계
    spread_formation = spread[:formation_period]
    mean = np.mean(spread_formation)
    std = np.std(spread_formation)
    
    # Z-Score
    zscore = (spread - mean) / std if std > 0 else np.zeros_like(spread)
    
    return {
        'hedge_ratio': beta,
        'spread': spread,
        'zscore': zscore,
        'mean': mean,
        'std': std
    }


def simulate_pair_trading(
    zscore: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_loss_z: float,
    max_holding: int,
    fee: float = 0.001  # 수수료+슬리피지
) -> Dict:
    """간단한 페어 트레이딩 시뮬레이션"""
    
    n = len(zscore)
    position = 0  # 0: no position, 1: long spread, -1: short spread
    entry_z_val = 0
    entry_idx = 0
    
    trades = []
    
    for i in range(n):
        z = zscore[i]
        
        if position == 0:
            # Entry
            if z >= entry_z:
                position = -1  # Short spread
                entry_z_val = z
                entry_idx = i
            elif z <= -entry_z:
                position = 1  # Long spread
                entry_z_val = z
                entry_idx = i
                
        else:
            holding_bars = i - entry_idx
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if position == 1:  # Long spread
                if z >= -exit_z:  # Mean reversion
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z <= entry_z_val - stop_loss_z:  # Stop loss
                    should_exit = True
                    exit_reason = "stop_loss"
            else:  # Short spread
                if z <= exit_z:  # Mean reversion
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z >= entry_z_val + stop_loss_z:  # Stop loss
                    should_exit = True
                    exit_reason = "stop_loss"
            
            if holding_bars >= max_holding:
                should_exit = True
                exit_reason = "max_holding"
            
            if should_exit:
                # PnL (Z-Score 변화가 대략적인 수익)
                z_change = (entry_z_val - z) * position
                pnl = z_change * 0.01 - fee  # 대략 z-score 1당 1% 변화
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_z': entry_z_val,
                    'exit_z': z,
                    'position': position,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'holding': holding_bars
                })
                
                position = 0
    
    # 열린 포지션 청산
    if position != 0:
        z = zscore[-1]
        z_change = (entry_z_val - z) * position
        pnl = z_change * 0.01 - fee
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': n - 1,
            'entry_z': entry_z_val,
            'exit_z': z,
            'position': position,
            'pnl': pnl,
            'reason': 'end_of_data',
            'holding': n - 1 - entry_idx
        })
    
    return trades


def evaluate_params(
    data: Dict[str, pd.DataFrame],
    params: Dict,
    formation_period: int = 960  # 10일 (96 * 10)
) -> Dict:
    """파라미터 조합 평가"""
    
    symbols = list(data.keys())
    all_trades = []
    
    # 모든 페어 조합
    for s1, s2 in combinations(symbols, 2):
        prices1 = data[s1]['Close'].values
        prices2 = data[s2]['Close'].values
        
        if len(prices1) < formation_period + 100:
            continue
        
        # 스프레드 계산
        stats = calculate_spread_stats(prices1, prices2, formation_period)
        
        # 트레이딩 기간 (형성 기간 이후)
        trading_zscore = stats['zscore'][formation_period:]
        
        # 시뮬레이션
        trades = simulate_pair_trading(
            zscore=trading_zscore,
            entry_z=params['entry_zscore'],
            exit_z=params['exit_zscore'],
            stop_loss_z=params['stop_loss_zscore'],
            max_holding=params.get('max_holding', 96)
        )
        
        for t in trades:
            t['pair'] = (s1, s2)
        
        all_trades.extend(trades)
    
    if not all_trades:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'sharpe': 0
        }
    
    # 메트릭스 계산
    returns = [t['pnl'] for t in all_trades]
    total_return = sum(returns) * 100  # 퍼센트
    wins = [r for r in returns if r > 0]
    
    return {
        'total_return': total_return,
        'num_trades': len(all_trades),
        'win_rate': len(wins) / len(all_trades) * 100 if all_trades else 0,
        'avg_win': np.mean([r for r in returns if r > 0]) * 100 if wins else 0,
        'avg_loss': abs(np.mean([r for r in returns if r <= 0])) * 100 if len(returns) > len(wins) else 0,
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'trades': all_trades
    }


def walk_forward_fast(
    data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List],
    train_days: int = 30,
    test_days: int = 7,
) -> Dict:
    """빠른 Walk-Forward 분석"""
    
    first_df = next(iter(data.values()))
    total_bars = len(first_df)
    bars_per_day = 96  # 15분봉
    
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    
    print(f"\n📊 Fast Walk-Forward Analysis")
    print(f"   Total: {total_bars} bars ({total_bars/bars_per_day:.0f} days)")
    print(f"   Train/Test: {train_days}/{test_days} days")
    
    # 단순 50/50 분할
    split_idx = total_bars // 2
    
    train_data = {s: df.iloc[:split_idx].copy() for s, df in data.items()}
    test_data = {s: df.iloc[split_idx:].copy() for s, df in data.items()}
    
    print(f"   Train: {split_idx} bars")
    print(f"   Test: {total_bars - split_idx} bars")
    
    # Grid Search on training data
    print(f"\n🔍 Grid Search on training data...")
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations_list = list(product(*param_values))
    
    best_params = None
    best_score = float('-inf')
    all_results = []
    
    for i, values in enumerate(combinations_list):
        params = dict(zip(param_names, values))
        
        results = evaluate_params(train_data, params)
        
        # Sharpe ratio 기준
        score = results['sharpe']
        
        all_results.append({
            'params': params.copy(),
            'train_return': results['total_return'],
            'train_trades': results['num_trades'],
            'train_win_rate': results['win_rate'],
            'train_sharpe': results['sharpe'],
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
        
        print(f"   [{i+1}/{len(combinations_list)}] entry={params['entry_zscore']}, exit={params['exit_zscore']}, stop={params['stop_loss_zscore']}: Sharpe={score:.2f}, Return={results['total_return']:.1f}%")
    
    print(f"\n✨ Best params: {best_params}")
    
    # Test on out-of-sample data
    print(f"\n📈 Testing on out-of-sample data...")
    train_results = evaluate_params(train_data, best_params)
    test_results = evaluate_params(test_data, best_params)
    
    print(f"   Train: Return={train_results['total_return']:.1f}%, WinRate={train_results['win_rate']:.1f}%, Trades={train_results['num_trades']}")
    print(f"   Test:  Return={test_results['total_return']:.1f}%, WinRate={test_results['win_rate']:.1f}%, Trades={test_results['num_trades']}")
    
    # Efficiency ratio
    efficiency = 0
    if train_results['total_return'] != 0:
        efficiency = test_results['total_return'] / train_results['total_return']
    
    print(f"   Efficiency Ratio: {efficiency:.2f}")
    
    return {
        'best_params': best_params,
        'train_results': {
            'return': train_results['total_return'],
            'trades': train_results['num_trades'],
            'win_rate': train_results['win_rate'],
            'avg_win': train_results['avg_win'],
            'avg_loss': train_results['avg_loss'],
            'sharpe': train_results['sharpe'],
        },
        'test_results': {
            'return': test_results['total_return'],
            'trades': test_results['num_trades'],
            'win_rate': test_results['win_rate'],
            'avg_win': test_results['avg_win'],
            'avg_loss': test_results['avg_loss'],
            'sharpe': test_results['sharpe'],
        },
        'efficiency': efficiency,
        'all_results': all_results,
    }


def calculate_kelly(results: Dict) -> Dict:
    """Kelly Criterion 계산"""
    
    test = results['test_results']
    
    win_rate = test['win_rate'] / 100
    avg_win = test['avg_win']
    avg_loss = test['avg_loss']
    
    if avg_loss == 0 or win_rate == 0:
        return {'kelly': 0, 'half_kelly': 0}
    
    # b = win/loss ratio
    b = avg_win / avg_loss
    
    # Kelly formula: f* = (p*b - q) / b
    # where p = win rate, q = 1-p
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    
    return {
        'win_rate': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': b,
        'kelly': kelly * 100,
        'half_kelly': kelly * 50,
        'quarter_kelly': kelly * 25,
    }


def main():
    print("🔍 Fast Walk-Forward Analysis for Parameter Generalization")
    print("=" * 60)
    
    config = Config()
    cache = DataCache(
        cache_dir=str(config.data.cache_dir),
        memory_size=100,
        compress=True,
        default_expiry_hours=24 * 365
    )
    
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
        "SUIUSDT", "XLMUSDT", "DOTUSDT", "LTCUSDT",
    ]
    
    print(f"\n📊 Loading cached 15m data...")
    data = load_cached_data(cache, SYMBOLS, "15m")
    print(f"   Loaded: {len(data)} symbols")
    
    if len(data) < 2:
        print("❌ Not enough data")
        return
    
    # 공통 인덱스
    common_index = None
    for df in data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    
    data = {s: df.loc[common_index] for s, df in data.items()}
    print(f"   Common bars: {len(common_index)}")
    
    # 파라미터 그리드
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5, 3.0],
        'exit_zscore': [0.3, 0.5, 0.7],
        'stop_loss_zscore': [3.0, 3.5, 4.0],
    }
    
    print(f"\n🎯 Parameter Grid: {len(param_grid['entry_zscore'])} x {len(param_grid['exit_zscore'])} x {len(param_grid['stop_loss_zscore'])} = {len(param_grid['entry_zscore']) * len(param_grid['exit_zscore']) * len(param_grid['stop_loss_zscore'])} combinations")
    
    # Walk-Forward
    results = walk_forward_fast(data, param_grid)
    
    # Kelly 계산
    kelly = calculate_kelly(results)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    print(f"\n✨ Recommended Parameters:")
    for k, v in results['best_params'].items():
        print(f"   {k}: {v}")
    
    print(f"\n📈 Performance Summary:")
    print(f"   Train Return: {results['train_results']['return']:.1f}%")
    print(f"   Test Return:  {results['test_results']['return']:.1f}%")
    print(f"   Efficiency:   {results['efficiency']:.2f}")
    
    if results['efficiency'] > 0.5:
        print("   ✅ Good generalization")
    elif results['efficiency'] > 0.3:
        print("   ⚠️  Moderate overfitting")
    else:
        print("   ❌ High overfitting risk")
    
    print(f"\n📊 Kelly Criterion (from OOS test):")
    print(f"   Win Rate: {kelly['win_rate']:.1f}%")
    print(f"   Avg Win:  {kelly['avg_win']:.2f}%")
    print(f"   Avg Loss: {kelly['avg_loss']:.2f}%")
    print(f"   Win/Loss Ratio: {kelly['win_loss_ratio']:.2f}")
    print(f"   Full Kelly:    {kelly['kelly']:.1f}%")
    print(f"   Half Kelly:    {kelly['half_kelly']:.1f}% (recommended)")
    print(f"   Quarter Kelly: {kelly['quarter_kelly']:.1f}% (conservative)")
    
    # 결과 저장
    output = {
        'best_params': results['best_params'],
        'train': results['train_results'],
        'test': results['test_results'],
        'efficiency': results['efficiency'],
        'kelly': kelly,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    output_path = Path(__file__).parent.parent / "walk_forward_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
