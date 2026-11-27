"""6주차 ④ 실시간 모니터링 — NH 웹소켓(국내_시세체결 KA · 국내_주문/체결 O1_K)으로 지켜보다가 이상이면 멈춤.

20초 REST 루프(4주차 실행기) 사이에 일어나는 일을 잡는 자리입니다. 두 부분으로 나눴습니다.

  Watch      판단만 하는 부분 (네트워크 없음 → test6.py 가 가짜 이벤트로 시험)
  NHStream   NH 앱(nh-futures-open-api)의 backend.realtime.link.Connection 으로 웹소켓을 붙여 Watch 에 넘김
             시세는 실전 키(real), 주문/체결 알림은 모의 키(paper) — 4주차와 같은 이유(모의 키에 국내 시세 권한 없음)

Watch 가 보는 것                                   하는 일
  연결 상태가 LIVE 아님 ≥ 20초                      신규 중지 (청산·REST 루프는 계속) → LIVE 로 돌아오면 재개 + 「REST 로 다시 채우기」
  장중 두 다리 중 하나라도 체결 틱 없음 ≥ 120초     신규 중지 (시세 멈춤) → 틱 오면 재개
  내 주문이 거부(9)·자동취소(8)                     신규 중지 — **사람이 확인할 때까지** (자동으로 안 풀림)
  내 주문이 접수 뒤 10초 안 전량 체결 안 됨          경보 / 30초 → 신규 중지 (사람 확인)
  보유 중 틱마다 z·평가손익 다시 계산               가격 손절선·금액 손절선을 넘으면 그 자리에서 청산 요청 (봉 끝을 안 기다림)

모든 일은 log 함수(실행기의 Rec.log)로 log.jsonl 에 남깁니다 → README 6절 「작동 기록」.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class WatchCfg:
    down_after: float = 20.0         # 연결이 LIVE 아닌 채 이만큼 → 신규 중지
    stale_after: float = 120.0       # 장중 틱 없음 → 신규 중지
    unfilled_warn: float = 10.0      # 내 주문 미체결 경보
    unfilled_block: float = 30.0     # 미체결 신규 중지
    session: tuple = ("09:00", "15:20")


STICKY = ("주문거부", "미체결", "사람확인")      # 사람이 풀어야 하는 것 (자동 해제 없음)


@dataclass
class Tracked:
    no: str
    sym: str
    side: str
    qty: int
    t_sent: float
    filled: int = 0
    warned: bool = False
    done: bool = False


@dataclass
class Watch:
    cfg: WatchCfg
    syms: tuple                                   # (삼성 종목코드, 하이닉스 종목코드)
    log: Callable = print
    block: Callable = lambda why: None             # 엔진 신규 중지
    unblock: Callable = lambda why: None
    urgent_exit: Callable = lambda why: None       # 틱 손절 — 실행기가 바로 청산
    clock: Callable = time.time
    hhmm: Callable = lambda: time.strftime("%H:%M")
    state: str = "stopped"
    state_since: float = 0.0
    last_tick: dict = field(default_factory=dict)
    price: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)
    issues: dict = field(default_factory=dict)    # 이름 → 내용
    need_refill: bool = False
    guard: dict | None = None                     # 보유 중 틱 손절 기준 (실행기가 봉마다 갱신)
    _fired: bool = False
    fired_at: float = 0.0                         # 틱 손절이 잡힌 시각 (실행기가 청산 주문까지의 지연을 잼)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ── 이벤트 (NHStream 이 부름, 다른 스레드) ──
    def on_state(self, state: str, detail: str = ""):
        with self._lock:
            if state != self.state:
                self.log("실시간연결", 상태=state, 내용=detail or None)
                if state == "live" and self.state in ("reconnecting", "connecting") and self.state_since:
                    self.need_refill = True
                self.state, self.state_since = state, self.clock()

    def on_tick(self, sym: str, px: float):
        with self._lock:
            self.price[sym], self.last_tick[sym] = px, self.clock()
            g, fire = self.guard, None
            if g is not None and not self._fired and all(s in self.price for s in self.syms):
                fire = self._tick_stop(g)
            if fire:
                self._fired = True
                self.fired_at = self.clock()
        if fire:
            self.log("틱손절", 사유=fire[0], **fire[1])
            self.urgent_exit(fire[0])

    def _tick_stop(self, g: dict):
        """g = {dir, mu, se, entry, stop_gap, tranches:[(qa,qb,pa,pb)], loss_lim, mult, mu0?}.  z = (ln A − ln B − μ)/σ_eq.
        mu0 (진입 시점 평균)가 있으면 고정 z 로도 봄 — 이동 z 와 둘 중 먼저 닿는 쪽."""
        pa, pb = self.price[self.syms[0]], self.price[self.syms[1]]
        sp = math.log(pa) - math.log(pb)
        z = (sp - g["mu"]) / g["se"]
        d = g["dir"]
        if g.get("price_stop", True) and -z * d >= g["entry"] + g["stop_gap"]:
            return "틱 가격손절", {"z": round(z, 2), "손절선": round(g["entry"] + g["stop_gap"], 2), "A": pa, "B": pb}
        mu0 = g.get("mu0")
        if g.get("price_stop", True) and mu0 is not None and math.isfinite(mu0):
            z0 = (sp - mu0) / g["se"]
            if -z0 * d >= g["entry"] + g["stop_gap"]:
                return "틱 고정z손절", {"z0": round(z0, 2), "손절선": round(g["entry"] + g["stop_gap"], 2), "A": pa, "B": pb}
        if g.get("loss_lim"):
            pnl = sum(d * (qa * (pa - a0) - qb * (pb - b0)) * g["mult"] for qa, qb, a0, b0 in g["tranches"])
            if pnl <= -g["loss_lim"]:
                return "틱 금액손절", {"평가손익원": round(pnl), "한도원": -g["loss_lim"], "A": pa, "B": pb}
        return None

    def set_guard(self, g: dict | None):
        with self._lock:
            self.guard = g
            if g is None:
                self._fired = False

    def on_order(self, ev: dict):
        """order_event() 모양: id, status(accepted/confirmed/partial/filled/auto-cancelled/rejected), filled, remaining, errorCode."""
        no = str(ev.get("id") or "")
        with self._lock:
            tr = self.orders.get(no)
        st = ev.get("status")
        if tr is None:
            return
        self.log("주문알림", 주문번호=no, 상태=st, 체결=ev.get("filled"), 잔량=ev.get("remaining"),
                 체결가=ev.get("fillPrice"), 오류=ev.get("errorCode"), 접수후초=round(self.clock() - tr.t_sent, 1))
        if st in ("rejected", "auto-cancelled"):
            tr.done = True
            self._raise("주문거부", f"{tr.sym} {tr.side} {tr.qty} — {st} {ev.get('errorCode') or ''}".strip())
        elif st in ("filled", "partial"):
            tr.filled = max(tr.filled, int(ev.get("filled") or 0))
            if st == "filled" or (ev.get("remaining") == 0):
                tr.done = True

    def track(self, no: str, sym: str, side: str, qty: int):
        """실행기가 주문을 낸 직후 부름."""
        if no:
            with self._lock:
                self.orders[no] = Tracked(no, sym, side, qty, self.clock())

    def cancelled(self, no: str):
        """실행기가 스스로 취소한 주문 (6초 안 덜 체결 → 취소 후 남은 수량 재주문). 미체결 감시에서 뺌 —
        남은 수량은 새 주문번호로 다시 track 됨."""
        tr = self.orders.get(no)
        if tr is not None:
            tr.done = True

    def down_for(self) -> float:
        """시세 웹소켓이 LIVE 가 아닌 채 지난 초 (LIVE 면 0)."""
        if self.state == "live" or not self.state_since:
            return 0.0
        return self.clock() - self.state_since

    def settle(self, no: str, filled: int):
        """REST 주문내역으로 체결을 확인했을 때 (알림이 늦거나 빠졌을 때)."""
        tr = self.orders.get(no)
        if tr is not None:
            tr.filled = max(tr.filled, filled)
            tr.done = tr.done or tr.filled >= tr.qty

    # ── 주기 점검 (실행기 루프가 부름) ──
    def check(self):
        now = self.clock()
        in_session = self.cfg.session[0] <= self.hhmm() < self.cfg.session[1]
        # 연결
        if self.state != "live" and self.state_since and now - self.state_since >= self.cfg.down_after:
            self._raise("연결끊김", f"웹소켓 {self.state} {now - self.state_since:.0f}초")
        elif self.state == "live":
            self._clear("연결끊김")
        # 시세 멈춤
        if in_session and self.state == "live":
            stale = [s for s in self.syms if now - self.last_tick.get(s, self.state_since) >= self.cfg.stale_after]
            if stale:
                self._raise("시세정지", f"{stale} 체결 틱 {self.cfg.stale_after:.0f}초 넘게 없음")
            else:
                self._clear("시세정지")
        # 미체결
        for tr in list(self.orders.values()):
            if tr.done:
                continue
            age = now - tr.t_sent
            if age >= self.cfg.unfilled_block:
                tr.done = True
                self._raise("미체결", f"{tr.sym} {tr.side} {tr.qty} 중 {tr.filled} 체결, {age:.0f}초")
            elif age >= self.cfg.unfilled_warn and not tr.warned:
                tr.warned = True
                self.log("미체결경보", 주문번호=tr.no, 종목=tr.sym, 수량=tr.qty, 체결=tr.filled, 접수후초=round(age, 1))

    def _raise(self, name: str, what: str):
        if name not in self.issues:
            self.issues[name] = what
            self.log("이상감지", 항목=name, 내용=what, 자동해제=name not in STICKY)
            self.block(f"{name}: {what}")

    def _clear(self, name: str):
        if name in self.issues and name not in STICKY:
            what = self.issues.pop(name)
            self.log("이상해소", 항목=name, 내용=what)
            if not self.issues:
                self.unblock(name)

    def ack(self):
        """사람이 HTS 에서 확인한 뒤 (실행기 재시작 또는 --ack) 끈끈한 것까지 지움."""
        if self.issues:
            self.log("이상확인", 항목=list(self.issues))
            self.issues.clear()
            self.unblock("사람 확인")


# ── NH 웹소켓 ─────────────────────────────────────────────────
class NHStream:
    """NH 앱의 Connection 두 개(시세=real, 주문=paper)를 별도 스레드의 asyncio 루프에서 돌림."""

    def __init__(self, watch: Watch, syms: tuple, quote_env: str = "real", order_env: str = "paper"):
        self.w, self.syms, self.qenv, self.oenv = watch, syms, quote_env, order_env
        self.loop = None
        self.th = threading.Thread(target=self._run, daemon=True, name="nh-stream")

    def start(self):
        self.th.start()

    def _run(self):
        import asyncio
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:                       # 스트림이 죽어도 실행기는 계속 (Watch 가 「연결끊김」 으로 막음)
            self.w.log("실시간오류", 오류=f"{type(e).__name__}: {str(e)[:200]}")
            self.w.on_state("stopped", "스트림 스레드 종료")

    async def _main(self):
        import asyncio
        from backend.nh_api import runtime
        from backend.realtime.feeds import DOMESTIC_DAY, ORDERS, QUOTE, Feed, order_event, quote_patch, trade_of
        from backend.realtime.ledger import stream_key
        from backend.realtime.link import Connection
        cl = runtime.client()
        session = getattr(cl, "session", None) or getattr(cl, "_session")

        def on_data(key, body):
            _, api_id, sym = key
            if api_id == "KA":
                tr = trade_of(body, context=DOMESTIC_DAY)
                px = tr["price"] if tr else quote_patch(body, context=DOMESTIC_DAY).get("last")
                if tr and px:
                    self.w.on_tick(sym, float(px))
            elif api_id == "O1_K":
                self.w.on_order(order_event(body, context=DOMESTIC_DAY))

        def state_of(tag):
            def f(market, state, detail, sym=""):
                if tag == "quote":                    # 시세 연결이 끊기면 판단 근거가 없음 → 이것만 Watch 상태로
                    self.w.on_state(state, detail)
                else:
                    self.w.log("주문알림연결", 상태=state, 내용=detail or None)
            return f

        def make(env, tag):
            return Connection(market="domestic", environment=env, base_url=session.base_url(env),
                              issue_key=lambda: session.tokens.websocket_key("domestic", env),
                              access_token=lambda: session.tokens.token("domestic", env).access_token,
                              on_data=on_data, on_state=state_of(tag))
        cq, co = make(self.qenv, "quote"), make(self.oenv, "order")
        for s in self.syms:
            f = Feed(DOMESTIC_DAY, QUOTE, s)
            await cq.add(stream_key(f), f)
        f = Feed(DOMESTIC_DAY, ORDERS)
        await co.add(stream_key(f), f)
        cq.start(); co.start()
        while True:
            await asyncio.sleep(3600)
