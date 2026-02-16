"""
15분봉 3개월 데이터 다운로드 스크립트

실제 데이터만 사용 - 합성 데이터 금지
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from binance.client import Client
from quantbox.core.config import Config
from quantbox.data.downloader import DataDownloader
from quantbox.data.cache import DataCache

# 설정
INTERVAL = "15m"  # 라이브 트레이딩과 동일
DAYS = 90  # 3개월

# 상위 30 코인 (거래량 기준, 스테이블코인 제외)
TOP_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    "SUIUSDT", "TONUSDT", "XLMUSDT", "DOTUSDT", "HBARUSDT",
    "LTCUSDT", "SHIBUSDT", "BCHUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "PEPEUSDT", "ICPUSDT", "AAVEUSDT", "FETUSDT",
    "FILUSDT", "ARBUSDT", "ATOMUSDT", "OPUSDT", "INJUSDT"
]

def main():
    # pydantic-settings: .env 검증 (필수 변수 누락 시 즉시 에러)
    from quantbox.core.settings import get_settings
    settings = get_settings()
    
    # 클라이언트 초기화
    print("🔌 Connecting to Binance...")
    client = Client(settings.binance_api_key, settings.binance_api_secret)
    
    # 설정 및 캐시
    config = Config()
    cache = DataCache(
        cache_dir=str(config.data.cache_dir),
        memory_size=100,
        compress=True,
        default_expiry_hours=24 * 365  # 1년 (사실상 만료 없음)
    )
    
    # 다운로더 초기화
    downloader = DataDownloader(client, config, cache)
    
    # 날짜 범위 (3개월)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS)
    
    print(f"\n📊 Data Download Configuration:")
    print(f"   Interval: {INTERVAL}")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"   Days: {DAYS}")
    print(f"   Expected bars per coin: ~{DAYS * 24 * 4:,}")  # 15분 = 하루 96봉
    
    # 상위 코인 (미리 정의된 리스트 사용)
    coins = TOP_COINS
    print(f"\n📋 Using predefined top {len(coins)} coins:")
    print(f"   {', '.join(coins[:10])}...")
    
    # 데이터 다운로드
    print(f"\n⬇️  Downloading {INTERVAL} data for {len(coins)} coins...")
    data = downloader.download_multiple(
        symbols=coins,
        interval=INTERVAL,
        start=start_date,
        end=end_date,
        use_cache=True,
        progress=True
    )
    
    # 결과 요약
    print(f"\n✅ Download Complete!")
    print(f"   Downloaded: {len(data)}/{len(coins)} coins")
    
    total_bars = sum(len(df) for df in data.values())
    print(f"   Total bars: {total_bars:,}")
    
    # 캐시 정보
    cache_dir = Path(config.data.cache_dir)
    cache_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl.gz"))
    print(f"   Cache size: {cache_size / 1024 / 1024:.1f} MB")
    
    # 각 코인별 정보
    print(f"\n📋 Per-coin Summary:")
    for symbol, df in sorted(data.items()):
        start = df.index.min()
        end = df.index.max()
        print(f"   {symbol}: {len(df):,} bars ({start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')})")
    
    # 페어 수 계산
    n_pairs = len(coins) * (len(coins) - 1) // 2
    print(f"\n📈 Pairs available for analysis: {n_pairs}")
    
    return data


if __name__ == "__main__":
    main()
