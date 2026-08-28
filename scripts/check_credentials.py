"""유출된 자격증명이 실제로 폐기됐는지 검증한다 — 2026-08-27.

`check_secrets.py`가 "코드에 시크릿이 있는가"를 본다면, 이쪽은 **"히스토리에 이미
새어나간 그 값들이 아직 살아 있는가"**를 본다. 둘은 다른 질문이고, 공개 전환의
마지막 관문은 후자다.

검사 대상은 `.env`(현재 값)가 아니라 **유출 커밋에서 직접 뽑은 값**이다. 회전 후에는
`.env`에 새 값이 들어가므로, 현재 값을 보면 "옛 키가 죽었는지"를 영영 확인할 수 없다.

값은 어디에도 출력하지 않는다 — 살아있음/죽음만 찍는다.

    uv run python scripts/check_credentials.py

전부 폐기됐으면 exit 0. 하나라도 살아 있으면 exit 1.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.error
import urllib.request

#: `.env` 전체가 담긴 커밋 중 가장 이른 것. 값은 세 커밋 모두 동일하다.
LEAK_COMMIT = "1f7fc27"

USER_AGENT = "DiscordBot (https://github.com/quantbox, 1.0)"
TIMEOUT_S = 15


def _leaked_env() -> dict[str, str]:
    """유출 커밋의 `.env`에서 자격증명을 뽑는다. 값은 반환만 하고 출력하지 않는다."""
    blob = subprocess.run(["git", "show", f"{LEAK_COMMIT}:.env"], capture_output=True, text=True, check=False)
    if blob.returncode:
        return {}
    out: dict[str, str] = {}
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "DISCORD_WEBHOOK_URL", "DISCORD_BOT_TOKEN"):
        m = re.search(rf"^{key}=(.+)$", blob.stdout, re.M)
        if m:
            out[key] = m.group(1).strip().strip('"')
    return out


def _http_alive(url: str, headers: dict[str, str] | None = None) -> bool | None:
    """200이면 살아 있음, 401/403/404면 폐기. None은 판정 불가(네트워크 등)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        return None
    except Exception:
        return None


def check_discord_webhook(url: str) -> bool | None:
    return _http_alive(url)


def check_discord_bot(token: str) -> bool | None:
    return _http_alive("https://discord.com/api/v10/users/@me", {"Authorization": f"Bot {token}"})


def check_binance(key: str, secret: str) -> bool | None:
    """키가 아직 유효한지. 서명 요청이 -2015/-2014면 폐기된 것이다."""
    try:
        from binance.client import Client

        Client(key, secret).get_account_api_permissions()
        return True
    except Exception as e:
        msg = str(e)
        # -2008 Invalid Api-Key ID = 키 자체가 존재하지 않는다(삭제됨).
        # -2015 rejected IP/permissions, -2014 bad api-key fmt, -1022 bad signature.
        # 2026-08-28: -2008이 목록에 없어 삭제된 키를 "판정 불가"로 보고했다.
        # 검증기가 모르는 코드를 통과시키지 않는 건 옳지만, 아는 코드는 알아야 한다.
        if any(code in msg for code in ("-2008", "-2015", "-2014", "-1022", "Invalid API-key")):
            return False
        return None


def main() -> int:
    leaked = _leaked_env()
    if not leaked:
        print(f"🔴 {LEAK_COMMIT}에서 .env를 읽지 못했다 — 검증 불가", file=sys.stderr)
        return 1

    results: list[tuple[str, bool | None]] = []
    if "DISCORD_BOT_TOKEN" in leaked:
        results.append(("Discord 봇 토큰", check_discord_bot(leaked["DISCORD_BOT_TOKEN"])))
    if "DISCORD_WEBHOOK_URL" in leaked:
        results.append(("Discord 웹훅", check_discord_webhook(leaked["DISCORD_WEBHOOK_URL"])))
    if "BINANCE_API_KEY" in leaked and "BINANCE_API_SECRET" in leaked:
        results.append(("바이낸스 API 키", check_binance(leaked["BINANCE_API_KEY"], leaked["BINANCE_API_SECRET"])))

    print(f"유출 커밋 {LEAK_COMMIT}의 자격증명 상태:\n")
    alive = unknown = 0
    for name, state in results:
        if state is True:
            print(f"  🔴 {name:18s} 아직 살아 있다 — 폐기 필요")
            alive += 1
        elif state is False:
            print(f"  ✅ {name:18s} 폐기 확인")
        else:
            print(f"  ⚠️  {name:18s} 판정 불가 (네트워크/응답 이상) — 수동 확인 필요")
            unknown += 1

    print()
    if alive or unknown:
        print(f"공개 전환 불가: 살아있음 {alive}건, 판정불가 {unknown}건", file=sys.stderr)
        return 1
    print("✅ 유출된 자격증명 전부 폐기됨 — 공개 전환의 자격증명 조건 충족")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
