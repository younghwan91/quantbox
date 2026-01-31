"""
Pair selector for identifying tradeable pairs.

Uses the filter pipeline to find cointegrated pairs
from a universe of assets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from quantbox.strategy.filters import FilterPipeline, create_default_pipeline, FilterResult
from quantbox.data.preprocessor import DataPreprocessor, PairData


logger = logging.getLogger('PairSelector')


@dataclass
class PairCandidate:
    """Candidate trading pair with scores and statistics."""
    symbol1: str
    symbol2: str
    
    # Filter results
    passed: bool = False
    combined_score: float = 0.0
    filter_results: Dict[str, FilterResult] = field(default_factory=dict)
    
    # Statistics
    correlation: float = 0.0
    cointegration_pvalue: float = 1.0
    half_life: float = 0.0
    hedge_ratio: float = 1.0
    hurst: float = 0.5
    
    # Current state
    current_z_score: float = 0.0
    spread_std: float = 1.0
    
    @property
    def pair_key(self) -> Tuple[str, str]:
        return (self.symbol1, self.symbol2)
    
    def to_dict(self) -> Dict:
        return {
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'passed': self.passed,
            'score': self.combined_score,
            'correlation': self.correlation,
            'cointegration_pvalue': self.cointegration_pvalue,
            'half_life': self.half_life,
            'hedge_ratio': self.hedge_ratio,
            'current_z_score': self.current_z_score,
        }


class PairSelector:
    """
    Selector for identifying tradeable pairs.
    
    Uses a filter pipeline to evaluate all possible pairs
    and rank them by quality scores.
    """
    
    def __init__(
        self,
        filter_pipeline: FilterPipeline = None,
        preprocessor: DataPreprocessor = None,
        max_pairs: int = 50,
        n_jobs: int = 1
    ):
        self.pipeline = filter_pipeline or create_default_pipeline()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.max_pairs = max_pairs
        self.n_jobs = n_jobs
        
        # Cache
        self._candidates_cache: Dict[Tuple, PairCandidate] = {}
    
    def select_pairs(
        self,
        data: Dict[str, pd.DataFrame],
        progress: bool = True,
        min_data_points: int = 100
    ) -> List[PairCandidate]:
        """
        Select tradeable pairs from universe.
        
        Args:
            data: Dict of symbol -> OHLCV DataFrame
            progress: Show progress bar
            min_data_points: Minimum data points required
            
        Returns:
            List of PairCandidate sorted by score
        """
        symbols = list(data.keys())
        logger.info(f"Selecting pairs from {len(symbols)} symbols")
        
        # Generate all combinations
        all_pairs = list(combinations(symbols, 2))
        logger.info(f"Evaluating {len(all_pairs)} possible pairs")
        
        candidates = []
        
        # Evaluate pairs
        if self.n_jobs == 1:
            # Sequential processing
            iterator = tqdm(all_pairs, desc="Evaluating pairs") if progress else all_pairs
            for symbol1, symbol2 in iterator:
                candidate = self._evaluate_pair(
                    symbol1, symbol2, data, min_data_points
                )
                if candidate and candidate.passed:
                    candidates.append(candidate)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_pair, s1, s2, data, min_data_points
                    ): (s1, s2)
                    for s1, s2 in all_pairs
                }
                
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Evaluating pairs"
                ) if progress else as_completed(futures)
                
                for future in iterator:
                    candidate = future.result()
                    if candidate and candidate.passed:
                        candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Limit number of pairs
        selected = candidates[:self.max_pairs]
        
        logger.info(f"Selected {len(selected)} pairs from {len(candidates)} candidates")
        
        return selected
    
    def _evaluate_pair(
        self,
        symbol1: str,
        symbol2: str,
        data: Dict[str, pd.DataFrame],
        min_data_points: int
    ) -> Optional[PairCandidate]:
        """Evaluate a single pair."""
        try:
            df1 = data[symbol1]
            df2 = data[symbol2]
            
            # Align data
            common_idx = df1.index.intersection(df2.index)
            if len(common_idx) < min_data_points:
                return None
            
            close1 = df1.loc[common_idx, 'Close']
            close2 = df2.loc[common_idx, 'Close']
            
            # Get volumes if available
            vol1 = df1.loc[common_idx, 'Volume'] if 'Volume' in df1.columns else None
            vol2 = df2.loc[common_idx, 'Volume'] if 'Volume' in df2.columns else None
            
            # Apply filters
            passed, score, results = self.pipeline.apply(
                close1, close2,
                volume1=vol1,
                volume2=vol2
            )
            
            # Create candidate
            candidate = PairCandidate(
                symbol1=symbol1,
                symbol2=symbol2,
                passed=passed,
                combined_score=score,
                filter_results=results
            )
            
            # Extract statistics from filter results
            if 'correlation' in results:
                candidate.correlation = results['correlation'].metadata.get('correlation', 0)
            
            if 'cointegration' in results:
                candidate.cointegration_pvalue = results['cointegration'].metadata.get('p_value', 1)
            
            if 'half_life' in results:
                candidate.half_life = results['half_life'].metadata.get('half_life', 0)
                candidate.hedge_ratio = results['half_life'].metadata.get('hedge_ratio', 1)
            
            if 'hurst' in results:
                candidate.hurst = results['hurst'].metadata.get('hurst', 0.5)
            
            # Calculate current z-score
            if passed:
                log1 = np.log(close1)
                log2 = np.log(close2)
                spread = log1 - candidate.hedge_ratio * log2
                candidate.spread_std = spread.std()
                candidate.current_z_score = (spread.iloc[-1] - spread.mean()) / candidate.spread_std
            
            return candidate
            
        except Exception as e:
            logger.debug(f"Error evaluating {symbol1}/{symbol2}: {e}")
            return None
    
    def select_tradeable_pairs(
        self,
        candidates: List[PairCandidate],
        min_z_score: float = 2.0,
        max_z_score: float = 5.0
    ) -> List[PairCandidate]:
        """
        Filter candidates to pairs ready for trading.
        
        Args:
            candidates: List of pair candidates
            min_z_score: Minimum absolute z-score for entry
            max_z_score: Maximum absolute z-score (filter outliers)
            
        Returns:
            List of tradeable pairs
        """
        tradeable = []
        
        for candidate in candidates:
            z = abs(candidate.current_z_score)
            if min_z_score <= z <= max_z_score:
                tradeable.append(candidate)
        
        # Sort by absolute z-score (prefer stronger signals)
        tradeable.sort(key=lambda x: abs(x.current_z_score), reverse=True)
        
        return tradeable
    
    def rank_pairs(
        self,
        candidates: List[PairCandidate],
        weights: Dict[str, float] = None
    ) -> List[PairCandidate]:
        """
        Rank pairs by weighted scoring.
        
        Args:
            candidates: List of pair candidates
            weights: Dict of metric -> weight (default equal weights)
            
        Returns:
            Sorted list of candidates
        """
        if not weights:
            weights = {
                'combined_score': 0.3,
                'half_life_score': 0.2,
                'correlation_score': 0.15,
                'coint_score': 0.2,
                'z_score_strength': 0.15
            }
        
        def calculate_rank_score(c: PairCandidate) -> float:
            score = 0.0
            
            # Combined filter score
            score += weights.get('combined_score', 0) * c.combined_score
            
            # Half-life score (prefer lower, bounded)
            if 0 < c.half_life < 100:
                hl_score = 1 - (c.half_life / 100)
                score += weights.get('half_life_score', 0) * hl_score
            
            # Correlation score
            score += weights.get('correlation_score', 0) * c.correlation
            
            # Cointegration score
            coint_score = 1 - c.cointegration_pvalue
            score += weights.get('coint_score', 0) * coint_score
            
            # Z-score strength (prefer values around 2-3)
            z_abs = abs(c.current_z_score)
            z_strength = max(0, min(1, (z_abs - 1) / 2))
            score += weights.get('z_score_strength', 0) * z_strength
            
            return score
        
        # Calculate and sort
        for c in candidates:
            c.combined_score = calculate_rank_score(c)
        
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        
        return candidates
    
    def get_pair_report(
        self,
        candidates: List[PairCandidate],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Generate report DataFrame for top pairs.
        """
        records = []
        
        for i, c in enumerate(candidates[:top_n], 1):
            records.append({
                'rank': i,
                'pair': f"{c.symbol1}/{c.symbol2}",
                'score': f"{c.combined_score:.3f}",
                'correlation': f"{c.correlation:.3f}",
                'coint_pvalue': f"{c.cointegration_pvalue:.4f}",
                'half_life': f"{c.half_life:.1f}",
                'hedge_ratio': f"{c.hedge_ratio:.3f}",
                'z_score': f"{c.current_z_score:.2f}",
            })
        
        return pd.DataFrame(records)
    
    def clear_cache(self):
        """Clear the candidates cache."""
        self._candidates_cache.clear()


def select_pairs_from_data(
    data: Dict[str, pd.DataFrame],
    config: Dict = None,
    progress: bool = True
) -> List[PairCandidate]:
    """
    Convenience function to select pairs from data.
    
    Args:
        data: Dict of symbol -> OHLCV DataFrame
        config: Optional configuration for filter thresholds
        progress: Show progress bar
        
    Returns:
        List of selected pair candidates
    """
    from quantbox.strategy.filters import (
        FilterPipeline, CorrelationFilter, CointegrationFilter,
        HalfLifeFilter, HedgeRatioFilter, StationarityFilter
    )
    
    config = config or {}
    
    # Create pipeline with config
    pipeline = FilterPipeline([
        CorrelationFilter(
            min_correlation=config.get('min_correlation', 0.5),
            max_correlation=config.get('max_correlation', 0.99)
        ),
        CointegrationFilter(
            max_pvalue=config.get('coint_pvalue', 0.05)
        ),
        HalfLifeFilter(
            min_half_life=config.get('min_half_life', 1.0),
            max_half_life=config.get('max_half_life', 100.0)
        ),
        HedgeRatioFilter(
            min_ratio=config.get('min_hedge_ratio', 0.25),
            max_ratio=config.get('max_hedge_ratio', 4.0)
        ),
        StationarityFilter(
            max_pvalue=config.get('stationarity_pvalue', 0.05)
        ),
    ])
    
    selector = PairSelector(
        filter_pipeline=pipeline,
        max_pairs=config.get('max_pairs', 50)
    )
    
    return selector.select_pairs(data, progress=progress)
