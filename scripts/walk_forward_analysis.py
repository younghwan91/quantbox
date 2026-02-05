"""
Walk-Forward 분석을 통한 일반화 가능한 파라미터 탐색

목적:
- 과최적화(overfitting) 방지
- Out-of-Sample 성과 검증
- 안정적인 파라미터 범위 도출
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from itertools import product

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from quantbox.core.config import Config
from quantbox.data.cache import DataCache
from quantbox.backtest.runner import BacktestRunner, BacktestConfig
from quantbox.backtest.metrics import calculate_metrics, PerformanceMetrics
from quantbox.strategy.pair_selector import PairSelector
from quantbox.strategy.filters import FilterPipeline, CorrelationFilter, CointegrationFilter, HalfLifeFilter


def load_cached_data(cache: DataCache, symbols: List[str], interval: str = "15m") -> Dict[str, pd.DataFrame]:
    """캐시에서 데이터 로드"""
    data = {}
    
    for symbol in symbols:
        # 캐시 인덱스에서 해당 심볼의 15m 데이터 찾기
        for key in cache.disk_cache._index.keys():
            if symbol in key and interval in key:
                entry = cache.disk_cache.get(key)
                if entry and entry.data is not None:
                    df = entry.data
                    # OHLCV 컬럼만 선택
                    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(c in df.columns for c in cols):
                        data[symbol] = df[cols]
                    break
    
    return data


def run_single_backtest(
    data: Dict[str, pd.DataFrame],
    params: Dict,
    initial_capital: float = 10000.0
) -> Dict:
    """단일 파라미터 조합으로 백테스트 실행"""
    
    config = BacktestConfig(
        initial_capital=initial_capital,
        trading_fee=0.0005,  # 0.05% 바이낸스 수수료
        slippage=0.001,      # 0.1% 슬리피지
        
        # Window settings (15분봉 기준)
        formation_periods=params.get('formation_periods', 96 * 10),  # 10일
        trading_periods=params.get('trading_periods', 96),          # 1일
        window_shift=params.get('window_shift', 96),                # 1일
        
        # Strategy parameters
        entry_zscore=params.get('entry_zscore', 2.0),
        exit_zscore=params.get('exit_zscore', 0.5),
        stop_loss_zscore=params.get('stop_loss_zscore', 3.5),
        max_holding_periods=params.get('max_holding_periods', 96),  # 1일
        
        # Pair selection
        max_pairs_per_window=5,
        min_correlation=params.get('min_correlation', 0.6),
        max_cointegration_pvalue=0.05,
    )
    
    # 필터 파이프라인
    pipeline = FilterPipeline([
        CorrelationFilter(min_correlation=config.min_correlation, max_correlation=0.98),
        CointegrationFilter(max_pvalue=config.max_cointegration_pvalue),
        HalfLifeFilter(min_half_life=3, max_half_life=200),
    ])
    
    selector = PairSelector(filter_pipeline=pipeline, max_pairs=50)
    
    runner = BacktestRunner(config)
    
    try:
        results = runner.run_backtest(data, pair_selector=selector, progress=False)
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return {'metrics': PerformanceMetrics()}


def grid_optimize(
    data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List],
    maximize: str = 'sharpe_ratio'
) -> Tuple[Dict, PerformanceMetrics, List[Dict]]:
    """Grid Search 최적화"""
    
    best_params = None
    best_metrics = None
    best_score = float('-inf')
    all_results = []
    
    # 파라미터 조합 생성
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"  Testing {len(combinations)} parameter combinations...")
    
    for values in combinations:
        params = dict(zip(param_names, values))
        
        results = run_single_backtest(data, params)
        metrics = results.get('metrics', PerformanceMetrics())
        
        score = getattr(metrics, maximize, 0) or 0
        
        all_results.append({
            'params': params.copy(),
            'metrics': metrics,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = metrics
    
    return best_params, best_metrics, all_results


def walk_forward_analysis(
    data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List],
    train_days: int = 30,
    test_days: int = 7,
    step_days: int = 7,
) -> Dict:
    """Walk-Forward 분석 실행"""
    
    # 첫 번째 심볼에서 타임스탬프 추출
    first_df = next(iter(data.values()))
    timestamps = first_df.index.tolist()
    
    # 15분봉 기준 bars 계산
    bars_per_day = 96
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    step_bars = step_days * bars_per_day
    total_required = train_bars + test_bars
    
    print(f"\n📊 Walk-Forward Configuration:")
    print(f"   Train: {train_days} days ({train_bars} bars)")
    print(f"   Test: {test_days} days ({test_bars} bars)")
    print(f"   Step: {step_days} days ({step_bars} bars)")
    print(f"   Total data: {len(timestamps)} bars ({len(timestamps)/bars_per_day:.1f} days)")
    
    # 윈도우 생성
    windows = []
    i = 0
    while i + total_required <= len(timestamps):
        train_start_idx = i
        train_end_idx = i + train_bars
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + test_bars, len(timestamps))
        
        windows.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        i += step_bars
    
    print(f"   Windows: {len(windows)}")
    
    if len(windows) == 0:
        print("❌ Not enough data for walk-forward analysis")
        return {}
    
    # 각 윈도우 처리
    window_results = []
    
    for w_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n📈 Window {w_id + 1}/{len(windows)}:")
        print(f"   Train: {timestamps[train_start].strftime('%Y-%m-%d')} ~ {timestamps[train_end-1].strftime('%Y-%m-%d')}")
        print(f"   Test: {timestamps[test_start].strftime('%Y-%m-%d')} ~ {timestamps[test_end-1].strftime('%Y-%m-%d')}")
        
        # 데이터 분할
        train_data = {
            symbol: df.iloc[train_start:train_end].copy()
            for symbol, df in data.items()
        }
        test_data = {
            symbol: df.iloc[test_start:test_end].copy()
            for symbol, df in data.items()
        }
        
        # 학습 데이터로 최적화
        print("   Optimizing on training data...")
        best_params, train_metrics, _ = grid_optimize(train_data, param_grid)
        
        if best_params is None:
            print("   ⚠️  No valid parameters found")
            continue
        
        print(f"   Best params: entry={best_params.get('entry_zscore')}, exit={best_params.get('exit_zscore')}, stop={best_params.get('stop_loss_zscore')}")
        print(f"   Train return: {train_metrics.total_return_pct:.2f}%")
        
        # 테스트 데이터로 검증
        print("   Testing on out-of-sample data...")
        test_results = run_single_backtest(test_data, best_params)
        test_metrics = test_results.get('metrics', PerformanceMetrics())
        
        print(f"   Test return: {test_metrics.total_return_pct:.2f}%")
        
        # 효율성 비율 계산
        efficiency = 0.0
        if train_metrics.total_return_pct != 0:
            efficiency = test_metrics.total_return_pct / train_metrics.total_return_pct
        
        print(f"   Efficiency ratio: {efficiency:.2f}")
        
        window_results.append({
            'window_id': w_id,
            'train_start': timestamps[train_start],
            'train_end': timestamps[train_end-1],
            'test_start': timestamps[test_start],
            'test_end': timestamps[test_end-1],
            'best_params': best_params,
            'train_return': train_metrics.total_return_pct,
            'train_sharpe': train_metrics.sharpe_ratio,
            'train_trades': train_metrics.total_trades,
            'test_return': test_metrics.total_return_pct,
            'test_sharpe': test_metrics.sharpe_ratio,
            'test_trades': test_metrics.total_trades,
            'efficiency': efficiency,
        })
    
    return compile_wf_results(window_results, param_grid)


def compile_wf_results(window_results: List[Dict], param_grid: Dict) -> Dict:
    """Walk-Forward 결과 컴파일"""
    
    if not window_results:
        return {'error': 'No valid windows'}
    
    # 통계
    test_returns = [w['test_return'] for w in window_results]
    train_returns = [w['train_return'] for w in window_results]
    efficiencies = [w['efficiency'] for w in window_results]
    
    summary = {
        'num_windows': len(window_results),
        'avg_train_return': np.mean(train_returns),
        'avg_test_return': np.mean(test_returns),
        'std_test_return': np.std(test_returns),
        'avg_efficiency': np.mean(efficiencies),
        'min_efficiency': min(efficiencies),
        'max_efficiency': max(efficiencies),
        'positive_windows': sum(1 for r in test_returns if r > 0),
        'negative_windows': sum(1 for r in test_returns if r <= 0),
        'win_rate': sum(1 for r in test_returns if r > 0) / len(test_returns) * 100,
    }
    
    # 파라미터별 빈도 (가장 자주 선택된 값)
    param_frequency = {}
    for param_name in param_grid.keys():
        values = [w['best_params'][param_name] for w in window_results if param_name in w['best_params']]
        if values:
            unique, counts = np.unique(values, return_counts=True)
            param_frequency[param_name] = {
                'most_common': unique[np.argmax(counts)],
                'frequency': counts[np.argmax(counts)] / len(values) * 100,
                'mean': np.mean(values),
                'std': np.std(values),
            }
    
    # 권장 파라미터 (테스트 수익률 가중 평균)
    recommended_params = {}
    for param_name in param_grid.keys():
        weighted_sum = 0
        weight_total = 0
        for w in window_results:
            if w['test_return'] > 0:  # 양수 수익 윈도우만
                weight = w['test_return']
                weighted_sum += w['best_params'][param_name] * weight
                weight_total += weight
        
        if weight_total > 0:
            recommended_params[param_name] = weighted_sum / weight_total
        else:
            # 단순 평균 사용
            recommended_params[param_name] = np.mean([
                w['best_params'][param_name] for w in window_results
            ])
    
    return {
        'summary': summary,
        'windows': window_results,
        'param_frequency': param_frequency,
        'recommended_params': recommended_params,
    }


def print_results(results: Dict):
    """결과 출력"""
    
    print("\n" + "="*60)
    print("📊 WALK-FORWARD ANALYSIS RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    summary = results['summary']
    print(f"\n📈 Summary:")
    print(f"   Windows analyzed: {summary['num_windows']}")
    print(f"   Avg Train Return: {summary['avg_train_return']:.2f}%")
    print(f"   Avg Test Return: {summary['avg_test_return']:.2f}%")
    print(f"   Test Return Std: {summary['std_test_return']:.2f}%")
    print(f"   Positive Windows: {summary['positive_windows']}/{summary['num_windows']} ({summary['win_rate']:.1f}%)")
    
    print(f"\n📉 Efficiency Ratio (Test/Train):")
    print(f"   Average: {summary['avg_efficiency']:.2f}")
    print(f"   Range: [{summary['min_efficiency']:.2f}, {summary['max_efficiency']:.2f}]")
    
    if summary['avg_efficiency'] > 0.5:
        print("   ✅ Good - Low overfitting risk")
    elif summary['avg_efficiency'] > 0.3:
        print("   ⚠️  Moderate - Some overfitting possible")
    else:
        print("   ❌ Poor - High overfitting risk")
    
    print(f"\n🎯 Parameter Stability:")
    for param, stats in results.get('param_frequency', {}).items():
        print(f"   {param}:")
        print(f"      Most common: {stats['most_common']} ({stats['frequency']:.0f}% of windows)")
        print(f"      Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    print(f"\n✨ Recommended Parameters (weighted by test return):")
    for param, value in results.get('recommended_params', {}).items():
        print(f"   {param}: {value:.2f}")
    
    # Kelly 계산을 위한 통계
    test_returns = [w['test_return'] for w in results['windows']]
    wins = [r for r in test_returns if r > 0]
    losses = [abs(r) for r in test_returns if r <= 0]
    
    if wins and losses:
        win_rate = len(wins) / len(test_returns)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        print(f"\n📊 Statistics for Kelly Criterion:")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        print(f"   Avg Win: {avg_win:.2f}%")
        print(f"   Avg Loss: {avg_loss:.2f}%")
        
        # Kelly formula: f* = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss if avg_loss > 0 else 1
        kelly = (win_rate * b - (1 - win_rate)) / b
        half_kelly = kelly / 2
        
        print(f"   Win/Loss Ratio: {b:.2f}")
        print(f"   Full Kelly: {kelly*100:.1f}%")
        print(f"   Half Kelly (recommended): {half_kelly*100:.1f}%")


def main():
    """메인 실행 함수"""
    
    print("🔍 Walk-Forward Analysis for Parameter Generalization")
    print("="*60)
    
    # 설정
    config = Config()
    cache = DataCache(
        cache_dir=str(config.data.cache_dir),
        memory_size=100,
        compress=True,
        default_expiry_hours=24 * 365
    )
    
    # 분석할 심볼 (라이브 트레이딩과 동일)
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
        "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
        "SUIUSDT", "XLMUSDT", "DOTUSDT", "LTCUSDT",
        "BCHUSDT", "UNIUSDT", "NEARUSDT", "APTUSDT", "PEPEUSDT",
    ]
    
    print(f"\n📊 Loading cached 15m data for {len(SYMBOLS)} symbols...")
    data = load_cached_data(cache, SYMBOLS, "15m")
    print(f"   Loaded: {len(data)} symbols")
    
    if len(data) < 2:
        print("❌ Not enough data. Please run download_15m_data.py first.")
        return
    
    # 데이터 정렬 (공통 타임스탬프만)
    common_index = None
    for symbol, df in data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    print(f"   Common timeframe: {len(common_index)} bars ({len(common_index)/96:.1f} days)")
    
    # 공통 인덱스로 필터링
    data = {symbol: df.loc[common_index] for symbol, df in data.items()}
    
    # 파라미터 그리드 (간소화)
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.3, 0.5, 0.7],
        'stop_loss_zscore': [3.0, 3.5, 4.0],
    }
    
    print(f"\n🎯 Parameter Grid:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    # Walk-Forward 분석
    results = walk_forward_analysis(
        data=data,
        param_grid=param_grid,
        train_days=30,   # 30일 학습
        test_days=7,     # 7일 테스트
        step_days=7,     # 7일씩 이동
    )
    
    # 결과 출력
    print_results(results)
    
    # 결과 저장
    output_path = Path(__file__).parent.parent / "walk_forward_results.json"
    
    # datetime 객체를 문자열로 변환
    results_serializable = results.copy()
    if 'windows' in results_serializable:
        for w in results_serializable['windows']:
            for key in ['train_start', 'train_end', 'test_start', 'test_end']:
                if key in w and hasattr(w[key], 'isoformat'):
                    w[key] = w[key].isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
