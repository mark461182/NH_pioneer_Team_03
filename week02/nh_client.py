"""NH선물 REST 최소 클라이언트 — 토큰 발급과 국내 시세 조회.

키는 팀 공용 설정 파일(~/NHFutures/config/config.yaml, 서식은 week01/config.example.yaml)에서 읽는다.
시세 조회만 하며 주문 함수는 두지 않는다.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

BASE_URL = {"paper": "https://apidemo.futures.co.kr", "real": "https://api.futures.co.kr"}
CONFIG_PATH = Path.home() / "NHFutures" / "config" / "config.yaml"


def load_keys(env_name: str) -> tuple[str, str, str]:
    """(서버 주소, 국내 APP Key, 국내 Secret Key)"""
    if not CONFIG_PATH.exists():
        raise SystemExit(f"설정 파일이 없습니다: {CONFIG_PATH}")
    env = (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}).get(env_name) or {}
    market = env.get("domestic") or {}
    base = str(env.get("base_url") or "").strip() or BASE_URL[env_name]
    return base, str(market.get("app_key") or "").strip(), str(market.get("app_secret") or "").strip()


class NHClient:
    def __init__(self, env_name: str = "real"):
        self.base, self.appkey, self.appsecret = load_keys(env_name)
        self.token = self._issue_token()

    def _issue_token(self) -> str:
        # 토큰 발급만 form 형식을 받는다
        data = urllib.parse.urlencode(
            {"grant_type": "client_credentials", "appkey": self.appkey, "appsecret": self.appsecret}
        ).encode()
        req = urllib.request.Request(self.base + "/auth-service/v1/token", data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["access_token"]

    def post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code == 429 and attempt < 2:  # 호출 제한 — 잠깐 쉬고 다시
                    time.sleep(1.0 + attempt)
                    continue
                raise RuntimeError(f"{path} HTTP {e.code}: {body[:300]}") from None
        raise RuntimeError(f"{path} 재시도 초과")

    def chart_minute(self, sym: str, start_date: str, end_date: str, minutes: str = "1",
                     next_key: str = "99999999", rows: str = "300") -> dict:
        return self.post("/trade/v1/domestic/chart-minute", {
            "sym": sym, "inq_strt_dt": start_date, "inq_ed_dt": end_date,
            "strt_tm": "000000", "ed_tm": "235959", "dcnt": minutes,
            "nght_tp": "1", "cidx_yn": "0", "fut_cidx_exp_mod": "0",
            "nxt_key": next_key, "req_qty": rows,
        })
