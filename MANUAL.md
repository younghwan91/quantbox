# QuantBox 운영 매뉴얼

**v14.4.2 — Dual-Axis Squeeze (VR + MA Cluster)**

> 전략 설계·파라미터: `.github/agents.md` 참조  
> 파라미터 source of truth: `quantbox/strategy/vol_straddle/config.py`

---

## 1. 설치

```bash
git clone https://github.com/younghwan91/quantbox.git
cd quantbox

# uv (권장)
uv sync

# 또는 pip
pip install -e .
```

---

## 2. 환경변수 설정 (.env)

`.env.example`을 복사해서 시작:

```bash
cp .env.example .env
```

**.env 항목:**

```dotenv
# 🔑 API Keys — 미설정 시 실행 불가
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=false           # 테스트넷 사용 시 true

# Discord (선택)
# DISCORD_WEBHOOK_URL=          # 단방향 알림
# DISCORD_BOT_TOKEN=            # 양방향 원격 제어 (Bot)
# DISCORD_CHANNEL_ID=           # Bot 명령 수신 채널 ID (숫자)
```

> ⚠️ 전략 파라미터(SL, TP, leverage 등)는 `.env`가 아닌  
> `quantbox/strategy/vol_straddle/config.py`에서 관리.  
> `.env`는 인프라 설정(API 키, Discord)만 담당.

---

## 3. 실행

### 3.1 VolStraddle (현재 프로덕션 — v14.4.2)

```bash
# Paper trading (시뮬레이션, 실제 주문 없음)
python -m quantbox.scalper --strategy vol_straddle --paper

# 한 사이클 후 종료
python -m quantbox.scalper --strategy vol_straddle --paper --once

# LIVE (실거래 — 주의!)
python -m quantbox.scalper --strategy vol_straddle
```

### 3.2 데이터 캐시

```bash
python -m quantbox.data.downloader --symbols BTCUSDT ETHUSDT --tf 15m 1h 4h
```

### 3.3 백테스트

```bash
# 기준 백테스트 (scaling_r38 baseline)
python research/scaling_r38.py
```

---

## 4. 테스트

```bash
uv run pytest tests/ -v
```

---

## 5. Discord Bot 원격 제어

Bot을 활성화하려면 `.env`에 `DISCORD_BOT_TOKEN`과 `DISCORD_CHANNEL_ID`를 모두 설정.

**Bot 생성 방법:**

1. <https://discord.com/developers/applications> → New Application
2. Bot → Reset Token → **토큰 복사**
3. "Privileged Gateway Intents" → **Message Content Intent** 활성화
4. OAuth2 → URL Generator → Scopes: `bot`, Permissions: `Send Messages` + `Read Message History`
5. 생성된 URL로 서버에 초대
6. 사용할 채널 우클릭 → ID 복사 (Discord 개발자 모드 필요)

**사용 가능한 명령어:**

| 명령어 | 설명 |
|--------|------|
| `!status` | 전체 시스템 상태 + 슬롯별 PnL |
| `!pnl` | 손익 요약 |
| `!pause` | 신규 진입 일시 정지 (기존 포지션 유지) |
| `!resume` | 일시 정지 해제 |
| `!close all` | 전체 포지션 즉시 청산 |
| `!close N` | N번 슬롯만 청산 |
| `!stop` | 전체 청산 후 시스템 종료 |
| `!help` | 명령어 목록 |

> ⚠️ `!close`, `!stop`은 실제 바이낸스 주문을 발생시킵니다.

---

## 6. 바이낸스 API 설정

선물 거래에 필요한 권한:

- **Futures Trading (선물 거래)** ✓
- **Read (읽기)** ✓
- IP 화이트리스트 권장

> ⚠️ **Hedge Mode** (dualSidePosition)를 바이낸스 선물 계정 설정에서  
> 수동으로 활성화해야 합니다 (S_d8 스트래들 작동 요건).
