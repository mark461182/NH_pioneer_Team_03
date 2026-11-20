"""NH 실시간(웹소켓) 호가·체결 녹화기 — 조회만, 주문 없음.

REST 로 몇 초마다 물어보는 `analysis/book_rec.py` 와 달리, 서버가 **바뀔 때마다 밀어 주는**
체결(KA)·호가(KB) 메시지를 받은 순서 그대로, 받은 시각(밀리초)과 함께 한 줄씩 적습니다.

    python ws_rec.py --app-dir <nh-futures-open-api 폴더> --until 15:50
    python ws_rec.py --app-dir <...> --syms A086C000,A016C000,A056C000 --env real --seconds 60

- 접속 토큰과 웹소켓 접속키는 앱의 인증 모듈로 받습니다(앱이 파일에 둔 토큰을 같이 씀). 키·토큰·접속 주소는 어디에도 적지 않습니다.
- 서버가 보내는 `pingpong` 은 받은 그대로 되돌립니다(안 되돌리면 세션이 끊김). 웹소켓 규격의 핑은 보내지 않습니다.
- 끊기면 1·2·5·10·30·60초 간격으로 다시 붙고, 구독을 다시 겁니다. 끊긴 동안의 값은 메워지지 않으므로 끊김·재접속도 한 줄로 적습니다.
- 메시지 내용은 해석하지 않고 그대로 둡니다. 한 줄 = {"rt": 받은 시각, "m": 서버 메시지}
- 구독은 종목 × (체결, 호가). NH 의 전체 구독 상한은 20개입니다.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote as urlquote

KST = dt.timezone(dt.timedelta(hours=9))
STREAM_PATH = "/trade/ws-stream"
TRADE, BOOK = "KA", "KB"                       # 국내_시세체결, 국내_시세호가 (주간)
BACKOFF = (1, 2, 5, 10, 30, 60)
IDLE_TIMEOUT = 90.0                            # 이만큼 아무것도 안 오면 죽은 연결로 보고 다시 붙음
DEFAULT_SYMS = "A086C000,A016C000,A056C000"    # KRX300 · 코스피200 · 미니코스피 2612


def now() -> dt.datetime:
    return dt.datetime.now(KST)


def stamp() -> str:
    return now().isoformat(timespec="milliseconds")


def ws_base(base_url: str) -> str:
    return "wss://" + base_url[len("https://"):] if base_url.startswith("https://") else base_url


def is_pingpong(m: dict) -> bool:
    return str(m.get("code") or "").lower() == "pingpong"


class Recorder:
    def __init__(self, session, env: str, feeds: list[tuple[str, str]], out, end: dt.datetime):
        self.s, self.env, self.feeds, self.out, self.end = session, env, feeds, out, end
        self.count: Counter = Counter()
        self.pings = 0
        self.connects = 0

    def write(self, obj: dict) -> None:
        self.out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def event(self, kind: str, **kw) -> None:
        self.write({"rt": stamp(), "ev": kind, **kw})
        self.out.flush()
        print(stamp()[11:23], kind, kw, flush=True)

    async def once(self) -> None:
        from websockets.asyncio.client import connect as ws_connect

        key = await asyncio.to_thread(self.s.tokens.websocket_key, "domestic", self.env)       # 한 번 쓰고 버리는 접속키
        token = (await asyncio.to_thread(self.s.tokens.token, "domestic", self.env)).access_token
        url = f"{ws_base(self.s.base_url(self.env))}{STREAM_PATH}?access_key={urlquote(key, safe='')}"
        async with ws_connect(url, open_timeout=10, close_timeout=1, ping_interval=None, ping_timeout=None) as sock:
            await sock.send(json.dumps({"header": {"api_ver": "1", "action": "0"}, "body": {"action_code": "session_init", "session_token": token}}))
            t_end = time.time() + 10
            while True:
                m = json.loads(await asyncio.wait_for(sock.recv(), max(0.1, t_end - time.time())))
                if is_pingpong(m):
                    await sock.send(json.dumps(m, ensure_ascii=False))
                    continue
                if m.get("code") == "INIT" and m.get("msg") == "INIT_SUCCESS":
                    break
                raise ConnectionError(f"세션 초기화 거절: code={m.get('code')} msg={m.get('msg')}")
            self.connects += 1
            self.event("connected", n=self.connects, env=self.env)
            for code, sym in self.feeds:
                await sock.send(json.dumps({"header": {"api_ver": "1", "action": "A"}, "body": {"action_code": code, "action_sym": sym}}))
            last_flush = time.time()
            while now() < self.end:
                try:
                    raw = await asyncio.wait_for(sock.recv(), min(IDLE_TIMEOUT, max(0.5, (self.end - now()).total_seconds())))
                except asyncio.TimeoutError:
                    if now() >= self.end:
                        return
                    raise TimeoutError(f"{IDLE_TIMEOUT:.0f}초 동안 서버 신호 없음")
                rt = stamp()
                try:
                    m = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(m, dict):
                    continue
                if is_pingpong(m):
                    self.pings += 1
                    await sock.send(json.dumps(m, ensure_ascii=False))
                    continue
                if "code" in m:                                  # 구독 답(RT_INIT)·거절(RT_FAIL) 등 — 내용 중 구분에 필요한 것만
                    d = m.get("data") if isinstance(m.get("data"), dict) else {}
                    self.event("control", code=m.get("code"), msg=m.get("msg") or m.get("message"), action=d.get("action"),
                               action_code=d.get("action_code"), action_sym=d.get("action_sym"))
                    continue
                h = m.get("header") if isinstance(m.get("header"), dict) else {}
                self.count[(str(h.get("action_code")), str(h.get("action_sym")))] += 1
                self.write({"rt": rt, "m": m})
                if time.time() - last_flush > 1.0:
                    self.out.flush()
                    last_flush = time.time()

    async def run(self) -> None:
        fails = 0
        while now() < self.end:
            t0 = time.time()
            try:
                await self.once()
                self.event("closed", why="끝 시각" if now() >= self.end else "서버가 닫음")
            except Exception as e:                               # 끊김·거절·시간 초과 — 적고 다시 붙음
                self.event("disconnected", error=f"{type(e).__name__}: {str(e)[:200]}")
            if now() >= self.end:
                break
            fails = 0 if time.time() - t0 > 60 else fails + 1
            await asyncio.sleep(BACKOFF[min(fails, len(BACKOFF)) - 1] if fails else 0.5)
        self.out.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--env", default="real", choices=("real", "paper"), help="시세를 받을 환경 (조회만)")
    ap.add_argument("--syms", default=DEFAULT_SYMS)
    ap.add_argument("--until", help="끝 시각 HH:MM (KST)")
    ap.add_argument("--seconds", type=float, help="이만큼만 받고 끝 (시험용)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.app_dir).resolve()))
    from backend.nh_api.session import create_session

    if a.seconds:
        end = now() + dt.timedelta(seconds=a.seconds)
    elif a.until:
        h, m = map(int, a.until.split(":"))
        end = now().replace(hour=h, minute=m, second=0, microsecond=0)
    else:
        end = now().replace(hour=15, minute=50, second=0, microsecond=0)
    syms = [s.strip() for s in a.syms.split(",") if s.strip()]
    feeds = [(c, s) for s in syms for c in (TRADE, BOOK)]
    if len(feeds) > 20:
        sys.exit("구독은 20개까지입니다 (종목 10개)")
    day = now().strftime("%Y%m%d")
    out_dir = Path(a.out) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ws_{day}.jsonl"
    session = create_session()
    with open(path, "a", encoding="utf-8") as f:
        rec = Recorder(session, a.env, feeds, f, end)
        rec.event("start", env=a.env, feeds=[f"{c}:{s}" for c, s in feeds], until=end.strftime("%H:%M:%S"))
        asyncio.run(rec.run())
        rec.event("end", connects=rec.connects, pingpong=rec.pings, counts={f"{c}:{s}": n for (c, s), n in sorted(rec.count.items())})
    session.close()
    print("저장:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
