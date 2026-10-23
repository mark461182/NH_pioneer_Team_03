"""1주차: 접속 토큰 발급 → 국내 현재가 1회 조회 → 로그 저장.

실행 전 준비
  1) pip install requests pyyaml
  2) 홈 폴더에 아래 경로로 config.yaml 을 만들고 키를 입력
     윈도우: %USERPROFILE%\\NHFutures\\config\\config.yaml

실행
  python first_call.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path.home() / "NHFutures" / "config" / "config.yaml"
RESULT_DIR = Path(__file__).parent / "results"

DEFAULT_BASE_URL = {
    "paper": "https://apidemo.futures.co.kr",   # 모의
    "real": "https://api.futures.co.kr",        # 실전
}

TOKEN_PATH = "/auth-service/v1/token"
PRICE_PATH = "/trade/v1/domestic/price"

SYMBOL = "A016C000"      # 코스피200 F 202612 (발대식 자료 예시)
TIMEOUT = 10


def mask(value: str) -> str:
    """키·토큰을 그대로 찍지 않고 앞뒤 일부만 보여 줍니다."""
    if not value:
        return "(비어 있음)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]} ({len(value)}자)"


def load_settings() -> tuple[str, str, str]:
    """설정 파일에서 환경, 서버 주소, 국내 키를 읽습니다."""
    if not CONFIG_PATH.exists():
        raise SystemExit(f"설정 파일이 없습니다: {CONFIG_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

    env_name = str(config.get("default_environment") or "paper").strip()
    if env_name not in DEFAULT_BASE_URL:
        raise SystemExit(f"default_environment 는 paper 또는 real 이어야 합니다: {env_name}")

    env = config.get(env_name) or {}
    base_url = str(env.get("base_url") or "").strip() or DEFAULT_BASE_URL[env_name]

    market = env.get("domestic") or {}
    app_key = str(market.get("app_key") or "").strip()
    app_secret = str(market.get("app_secret") or "").strip()
    if not app_key or not app_secret:
        raise SystemExit(f"config.yaml 의 {env_name} > domestic 에 키가 비어 있습니다.")

    return base_url, app_key, app_secret


def issue_token(base_url: str, app_key: str, app_secret: str) -> str:
    """접속 토큰을 발급받습니다. form 형식으로 보냅니다."""
    res = requests.post(
        base_url + TOKEN_PATH,
        data={
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret,
        },
        timeout=TIMEOUT,
    )
    print(f"[토큰] 상태코드 {res.status_code}")
    if res.status_code >= 400:
        print("[토큰] 서버 응답:")
        print(res.text.strip())
        raise SystemExit("토큰 발급이 거부되었습니다.")

    body = res.json()
    token = str(body.get("access_token") or "").strip()
    if not token:
        raise SystemExit(f"응답에 access_token 이 없습니다: {body}")

    print(f"[토큰] 값 {mask(token)}")
    print(f"[토큰] 타입 {body.get('token_type', 'Bearer')}")
    print(f"[토큰] 만료 {body.get('expires_at') or body.get('expires_in')}")
    return token


def get_price(base_url: str, token: str, symbol: str) -> dict:
    """국내 현재가를 1회 조회합니다. 읽기만 하므로 주문은 나가지 않습니다."""
    res = requests.post(
        base_url + PRICE_PATH,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"sym": symbol},
        timeout=TIMEOUT,
    )
    print(f"[현재가] 상태코드 {res.status_code}")
    if res.status_code >= 400:
        print("[현재가] 서버 응답:")
        print(res.text.strip())
        raise SystemExit("현재가 조회가 거부되었습니다.")
    return res.json()


def main() -> None:
    base_url, app_key, app_secret = load_settings()
    print(f"[설정] 서버 {base_url}")
    print(f"[설정] APP Key {mask(app_key)}")
    print(f"[설정] Secret  {mask(app_secret)}\n")

    token = issue_token(base_url, app_key, app_secret)
    print()

    body = get_price(base_url, token, SYMBOL)
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1500])

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULT_DIR / f"first_call_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "요청시각": datetime.now().isoformat(timespec="seconds"),
                "서버": base_url,
                "경로": PRICE_PATH,
                "종목": SYMBOL,
                "응답": body,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[기록] {out}")


if __name__ == "__main__":
    main()
