"""5주차 — KRX300 선물 장 시작 지정가 전략의 실행기.

08:45~09:00 에는 KRX300 선물 호가가 100틱 안팎으로 벌어져 있습니다. 그 안쪽,
「코스피200 선물로 환산한 적정가 ∓ d틱」에 지정가를 걸어 두고, 체결되면 미니코스피로
바로 헤지한 뒤 09:15 에 두 다리를 정리합니다.

    적정가 F = (전일 15:00~15:19 의 KRX300/코스피200 선물 가격 비율 중앙값) × 지금 코스피200 선물 호가 중간값

이 파일에는 주문을 내는 코드가 없습니다. 주문은 `broker` 가 냅니다.
    broker.send(sym, side, qty, price=None) -> 주문번호      price 없으면 시장가
    broker.filled(주문번호) -> (체결 수량, 평균 체결가)
    broker.cancel(주문번호, sym, side)
4주차 `nh_pair.NHPaperBroker`(모의계좌)와 같은 모양이고, 아래 `ReplayBroker` 는 녹화한 호가로
체결을 흉내 내는 시험용입니다. 모의계좌 실행은 `run_open_limit.py` 가 합니다.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time as _time
from dataclasses import dataclass, field

KRX300 = "A086C000"      # KRX300 선물 2612
K200 = "A016C000"        # 코스피200 선물 2612 — 적정가 계산용
MINI = "A056C000"        # 미니코스피 2612 — 헤지용
TICK = 0.2               # KRX300 선물 호가 단위
MULT = {KRX300: 50_000, MINI: 50_000}


@dataclass(frozen=True)
class Params:
    ratio: float                          # 전일 비율 (KRX300 / 코스피200)
    d_ticks: int = 30                     # 적정가에서 몇 틱 떨어진 곳에 걸지
    qty: int = 1                          # KRX300 계약 수
    hedge_qty: int = 4                    # 미니코스피 계약 수 (KRX300 1계약 1.7억 ≈ 미니 4계약 1.6억)
    start: dt.time = dt.time(8, 45, 0)    # 이때부터 지정가를 걸고
    stop: dt.time = dt.time(8, 59, 30)    # 이때 미체결을 전부 취소 (09:00 현물 개장 전에 빠짐)
    exit: dt.time = dt.time(9, 15, 0)     # 이때 두 다리를 정리 (KRX300 은 상대 호가를 넘기는 지정가, 미니는 시장가)
    requote_ticks: int = 2                # 목표가가 이만큼 넘게 바뀌면 다시 걺
    max_k200_spread: float = 0.5          # 코스피200 호가가 이보다 벌어져 있으면 적정가를 믿지 않고 쉼 (포인트)
    fill_wait: float = 6.0                # 헤지·청산 주문 체결을 기다리는 시간(초)
    fill_poll: float = 1.0
    cross_ticks: int = 10                 # KRX300 청산 지정가를 상대 최우선 호가보다 몇 틱 더 넘겨 낼지 (KRX300 은 시장가 주문이 거부됨)
    max_exit_tries: int = 40              # 청산을 이만큼 시도해도 안 되면 포지션이 남았다고 적고 끝냄


def bid_price(fair: float, d: int) -> float:
    """적정가에서 d틱 아래, 호가 단위에 맞춰 내림."""
    return round(math.floor((fair - d * TICK) / TICK + 1e-9) * TICK, 1)


def ask_price(fair: float, d: int) -> float:
    """적정가에서 d틱 위, 호가 단위에 맞춰 올림."""
    return round(math.ceil((fair + d * TICK) / TICK - 1e-9) * TICK, 1)


def opposite(side: str) -> str:
    return "sell" if side == "buy" else "buy"


@dataclass
class Quote:
    """한 시점의 호가. 없는 쪽은 None."""
    t: dt.datetime
    k_bid: float | None
    k_ask: float | None
    x_bid: float | None = None
    x_ask: float | None = None


@dataclass
class Resting:
    no: str
    side: str
    price: float
    stuck: int = 0          # 취소가 거부된 횟수


@dataclass
class OpenLimit:
    """상태: 대기 → 호가중 → (철수확인) → 보유 → (청산중) → 끝.  하루에 한 번만 포지션을 잡습니다.

    - 내가 낸 지정가는 취소한 뒤에도 끝까지 체결 여부를 다시 봅니다 — 취소 요청과 체결이 엇갈릴 수 있어서입니다.
    - 취소가 거부된 주문은 살아 있는 것으로 보고, 취소될 때까지 그쪽에 새 주문을 내지 않고 매번 다시 취소합니다.
    - 청산은 지금까지의 체결을 모두 더한 순포지션을 0 으로 만드는 것이고, 안 되면 다음 호가에서 다시 시도합니다.
      KRX300 이 정리되기 전에는 헤지(미니)를 풀지 않습니다.
    """
    p: Params
    broker: object
    log: object = None
    sleep: object = None                               # 시장가 체결 대기에 쓰는 sleep (재생·시험에서는 기다리지 않음)
    state: str = "대기"
    resting: dict = field(default_factory=dict)       # side -> Resting (지금 걸려 있다고 아는 주문)
    limits: list = field(default_factory=list)        # 낸 지정가 전부 (주문번호, side, 가격)
    counted: set = field(default_factory=set)         # 체결을 이미 반영한 주문번호
    fills: list = field(default_factory=list)         # 체결 전부 (sym, side, qty, price, 구분)
    entry: dict | None = None
    hedge: dict | None = None
    exits: list = field(default_factory=list)
    unverified: list = field(default_factory=list)    # 취소 요청은 받아들여졌지만 잔량 0 을 아직 못 본 주문
    takes: list = field(default_factory=list)         # 헤지·청산 주문 [주문번호, sym, side, 반영한 체결 수량, 구분]
    exit_tries: int = 0
    note: str = ""
    _t: str = ""

    def _log(self, kind: str, **kw) -> None:
        if self.log:
            self.log(kind, t=self._t, **kw)

    def _note(self, s: str) -> None:
        self.note = (self.note + " / " if self.note else "") + s

    # ── 주문 묶음 ──────────────────────────────────────────────
    def _cancel(self, side: str, why: str) -> bool:
        """취소 요청이 받아들여지면 True. 거부되면 주문을 그대로 들고 있다가 다음 번에 다시 시도."""
        r = self.resting.get(side)
        if r is None:
            return True
        try:
            self.broker.cancel(r.no, KRX300, side)
        except Exception as e:      # 접수 확인 전이거나 이미 체결된 주문 — 살아 있을 수 있으므로 버리지 않음
            r.stuck += 1
            self._log("취소거부", side=side, 가격=r.price, 주문번호=r.no, 횟수=r.stuck, 오류=str(e)[:120])
            return False
        self.resting.pop(side)
        self.unverified.append(r)
        self._log("취소", side=side, 가격=r.price, 주문번호=r.no, 이유=why)
        return True

    def _cancel_all(self, why: str) -> bool:
        return all([self._cancel(side, why) for side in list(self.resting)])

    def _verify_cancels(self) -> None:
        """취소했다고 본 주문에 잔량이 남아 있으면 다시 취소. 브로커가 잔량을 못 알려 주면 확인하지 않음."""
        open_qty = getattr(self.broker, "open_qty", None)
        if open_qty is None:
            self.unverified.clear()
            return
        keep = []
        for r in self.unverified:
            try:
                rem = open_qty(r.no)
            except Exception as e:
                self._log("조회실패", 주문번호=r.no, 오류=str(e)[:150])
                keep.append(r)
                continue
            if rem:
                try:
                    self.broker.cancel(r.no, KRX300, r.side)
                    self._log("취소재시도", side=r.side, 가격=r.price, 주문번호=r.no, 잔량=rem)
                except Exception as e:
                    self._log("취소거부", side=r.side, 가격=r.price, 주문번호=r.no, 오류=str(e)[:120])
                keep.append(r)
        self.unverified = keep

    def _place(self, side: str, price: float, fair: float) -> None:
        try:
            no = self.broker.send(KRX300, side, self.p.qty, price=price)
        except Exception as e:
            self._log("주문거부", side=side, 가격=price, 오류=str(e)[:200])
            return
        self.resting[side] = Resting(no, side, price)
        self.limits.append((no, side, price))
        self._log("지정가", side=side, 가격=price, 적정가=round(fair, 2), 주문번호=no)

    def _take(self, sym: str, side: str, qty: int, kind: str, q: Quote | None = None) -> dict:
        """바로 체결시키는 주문을 내고 잠깐 기다림. 거부·미체결은 숨기지 않고 돌려줌.

        미니는 시장가. KRX300 은 시장가가 거부되므로 상대 최우선 호가를 cross_ticks 만큼 넘긴 지정가로 내고,
        기다려도 남은 수량은 취소합니다(다음 시도에서 새 호가로 다시 냄).
        """
        out = {"sym": sym, "side": side, "qty": 0, "price": 0.0, "주문번호": None, "주문수량": qty}
        price = None
        if sym == KRX300:
            ref = None if q is None else (q.x_ask if side == "buy" else q.x_bid)
            if ref is None:
                self._log(kind + "보류", sym=sym, side=side, 수량=qty, 이유="KRX300 상대 호가를 모름")
                return out
            price = round(ref + (1 if side == "buy" else -1) * self.p.cross_ticks * TICK, 1)
            out["주문가격"] = price
        try:
            no = self.broker.send(sym, side, qty, price=price) if price is not None else self.broker.send(sym, side, qty)
        except Exception as e:
            self._log(kind + "거부", sym=sym, side=side, 수량=qty, 가격=price, 오류=str(e)[:200])
            out["거부"] = True
            return out
        out["주문번호"] = no
        waited = 0.0
        while True:
            got, px = self._filled(no)
            if got >= qty or waited >= self.p.fill_wait:
                break
            (self.sleep or _time.sleep)(self.p.fill_poll)
            waited += self.p.fill_poll
        if got < qty and price is not None:             # 남은 지정가를 걸어 두지 않음
            try:
                self.broker.cancel(no, sym, side)
            except Exception as e:
                self._log("취소거부", sym=sym, side=side, 주문번호=no, 오류=str(e)[:120])
            got, px = self._filled(no)
        out["qty"], out["price"] = got, px
        if got:
            self.fills.append((sym, side, got, px, kind))
        self.takes.append([no, sym, side, got, kind])
        self._log(kind, sym=sym, side=side, 주문수량=qty, 체결수량=got, 체결가=px, 주문가격=price, 주문번호=no)
        return out

    def _filled(self, no: str) -> tuple:
        try:
            return self.broker.filled(no)
        except Exception as e:
            self._log("조회실패", 주문번호=no, 오류=str(e)[:150])
            return 0, 0.0

    def _late_takes(self) -> None:
        """헤지·청산 주문에 나중에 더 붙은 체결을 반영 (취소와 엇갈린 경우)."""
        for t in self.takes:
            no, sym, side, counted, kind = t
            got, px = self._filled(no)
            if got > counted:
                self.fills.append((sym, side, got - counted, px, kind + "(늦은 체결)"))
                t[3] = got
                self._log("늦은체결", sym=sym, side=side, 수량=got - counted, 체결가=px, 주문번호=no)

    def net(self) -> dict:
        """지금까지의 체결을 모두 더한 순포지션 (매수 +)."""
        out = {}
        for sym, side, qty, _, _ in self.fills:
            out[sym] = out.get(sym, 0) + (qty if side == "buy" else -qty)
        return out

    def _new_fills(self) -> list:
        """낸 지정가 전부(취소한 것 포함)에서 아직 반영 안 한 체결을 찾음."""
        new = []
        for no, side, price in self.limits:
            if no in self.counted:
                continue
            try:
                got, px = self.broker.filled(no)
            except Exception as e:
                self._log("조회실패", 주문번호=no, 오류=str(e)[:150])
                continue
            if got > 0:
                self.counted.add(no)
                new.append((side, got, px or price, no))
                if self.resting.get(side) and self.resting[side].no == no:
                    self.resting.pop(side)
        return new

    def _on_fills(self, q: Quote, new: list) -> None:
        for side, got, px, no in new:
            self.fills.append((KRX300, side, got, px, "지정가체결"))
            self._log("체결", side=side, 수량=got, 체결가=px, 주문번호=no, 코스피200=_mid(q.k_bid, q.k_ask))
        self._cancel_all("체결 — 남은 지정가 정리")
        late = self._new_fills()                       # 취소와 엇갈려 체결된 것
        for side, got, px, no in late:
            self.fills.append((KRX300, side, got, px, "지정가체결"))
            self._log("체결", side=side, 수량=got, 체결가=px, 주문번호=no, 비고="취소와 엇갈림")
        net = sum(g * (1 if s == "buy" else -1) for s, g, _, _ in new + late)
        if net == 0:                                   # 매수·매도가 둘 다 체결 — 이미 평평하므로 헤지하지 않음
            self.state = "끝"
            self._note("매수·매도 지정가가 모두 체결 — 헤지 없이 종료")
            self._log("종료", 이유="양쪽 체결")
            return
        side = "buy" if net > 0 else "sell"
        got = abs(net)
        px = next(p_ for s, _, p_, _ in new + late if s == side)
        self.entry = {"side": side, "qty": got, "price": px, "t": q.t.isoformat()}
        hq = max(1, round(self.p.hedge_qty * got / self.p.qty))
        h = self._take(MINI, opposite(side), hq, "헤지")
        self.hedge = {"side": h["side"], "qty": h["qty"], "price": h["price"], "주문번호": h["주문번호"], "주문수량": hq}
        if h["주문번호"] is None:
            self._note("헤지 주문 거부 — KRX300 한 다리만 보유 중")
        elif h["qty"] < hq:
            self._note(f"헤지 미완료 — 미니 {hq - h['qty']}계약 남음")
        self.state = "보유"

    def _absorb_late(self) -> None:
        """보유·청산 중에 뒤늦게 드러난 체결(취소가 안 먹은 지정가, 엇갈린 청산 주문)을 반영."""
        for side, got, px, no in self._new_fills():
            self.fills.append((KRX300, side, got, px, "지정가체결(늦은 체결)"))
            self._log("늦은체결", sym=KRX300, side=side, 수량=got, 체결가=px, 주문번호=no)
            self._note("지정가 늦은 체결 발견")
        self._late_takes()
        self._verify_cancels()
        if self.resting:
            self._cancel_all("보유 중 — 남은 지정가 정리")

    def _exit(self, q: Quote) -> None:
        """순포지션을 0 으로. KRX300 먼저, KRX300 이 0 이 된 뒤에만 미니를 정리. 안 되면 「청산중」으로 남아 다시 시도."""
        self._absorb_late()
        n = self.net().get(KRX300, 0)
        if n:
            e = self._take(KRX300, "sell" if n > 0 else "buy", abs(n), "청산", q)
            if e["주문번호"] is not None or e.get("거부"):
                self.exits.append(e)
        if not self.net().get(KRX300, 0):
            m = self.net().get(MINI, 0)
            if m:
                self.exits.append(self._take(MINI, "sell" if m > 0 else "buy", abs(m), "청산", q))
        self.exit_tries += 1
        left = {s_: v for s_, v in self.net().items() if v}
        if not left:
            self.state = "끝"
        elif self.exit_tries >= self.p.max_exit_tries:
            self.state = "끝"
            self._note(f"청산 미완료 — 남은 포지션 {left}")
            self._log("종료", 이유="청산 시도 한도", 남은포지션=left)
        else:
            self.state = "청산중"

    # ── 한 번의 판단 ────────────────────────────────────────────
    def step(self, q: Quote) -> None:
        now = q.t.time()
        self._t = q.t.strftime("%H:%M:%S")
        if self.state == "끝":
            return
        if self.state == "청산중":
            self._exit(q)
            return
        if self.state == "보유":
            if now >= self.p.exit:
                self._exit(q)
            else:
                self._absorb_late()
            return
        if self.state in ("호가중", "철수확인"):                  # 낸 주문이 체결됐나부터
            new = self._new_fills()
            if new:
                self._on_fills(q, new)
                if self.state == "보유" and now >= self.p.exit:
                    self._exit(q)
                return
        if self.state in ("호가중", "철수확인"):
            self._verify_cancels()
        if self.state == "철수확인":                              # 취소를 낸 뒤 한 번 더 봤는데 체결 없음
            if not self._cancel_all("주문 시간 끝") or self.unverified:
                return                                            # 아직 살아 있는 주문이 있으면 끝내지 않고 계속 취소
            self.state = "끝"
            self._note("체결 없음")
            self._log("종료", 이유="체결 없이 주문 시간 끝")
            return
        if now >= self.p.stop:                                    # 시간 끝 — 남은 주문을 빼고, 다음 번에 체결 여부를 한 번 더 확인
            if self.state == "대기":
                self.state = "끝"
                self._note("주문 시간에 시작하지 못함")
                self._log("종료", 이유="주문 없이 시간 끝")
                return
            self._cancel_all("주문 시간 끝")
            self.state = "철수확인"
            return
        if now < self.p.start:
            return
        if q.k_bid is None or q.k_ask is None or q.k_ask - q.k_bid > self.p.max_k200_spread or q.k_ask <= q.k_bid:
            self._cancel_all("코스피200 호가 이상")                # 적정가를 못 믿으면 주문을 걸어 두지 않음
            if self.state == "대기":
                return
            self.state = "호가중"
            return
        fair = self.p.ratio * (q.k_bid + q.k_ask) / 2
        want = {"buy": bid_price(fair, self.p.d_ticks), "sell": ask_price(fair, self.p.d_ticks)}
        if q.x_ask is not None and want["buy"] >= q.x_ask:        # 상대 호가를 넘어서 바로 체결되는 가격이면 그쪽은 쉼
            want.pop("buy")
        if q.x_bid is not None and want["sell"] <= q.x_bid:
            want.pop("sell")
        for side in ("buy", "sell"):
            r = self.resting.get(side)
            px = want.get(side)
            if px is None:
                if r:
                    self._cancel(side, "상대 호가와 겹침")
                continue
            if r and abs(r.price - px) <= self.p.requote_ticks * TICK + 1e-9:
                continue
            if r and not self._cancel(side, "적정가 이동"):
                continue                                          # 옛 주문이 안 빠졌으면 그쪽에 새 주문을 얹지 않음
            self._place(side, px, fair)
        self.state = "호가중"

    # ── 결과 ───────────────────────────────────────────────────
    def pnl(self) -> dict:
        """체결가만으로 계산한 손익(원). 수수료 제외. 포지션이 남아 있으면 손익을 확정하지 않음."""
        if not self.fills:
            return {"거래": 0, "손익": 0.0, "메모": self.note}
        cash, by_sym = 0.0, {}
        for sym, side, qty, px, _ in self.fills:
            sgn = 1 if side == "buy" else -1
            cash -= sgn * qty * px * MULT[sym]
            by_sym[sym] = by_sym.get(sym, 0.0) - sgn * qty * px * MULT[sym]
        left = {s_: n for s_, n in self.net().items() if n}
        if left:
            return {"거래": 1, "손익": None, "남은 포지션": left, "메모": self.note}
        return {"거래": 1, "손익": cash, "KRX300 다리": by_sym.get(KRX300, 0.0), "미니 다리": by_sym.get(MINI, 0.0),
                "KRX300 틱 환산": round(cash / (TICK * MULT[KRX300]), 1), "남은 포지션": {}, "메모": self.note}


def _mid(b, a):
    return None if b is None or a is None else (b + a) / 2


# ── 녹화 자료로 체결을 흉내 내는 브로커 ───────────────────────────────
class ReplayBroker:
    """`book_rec.py` 가 5초마다 적은 호가·현재가로 체결을 판정합니다 (시험·재생·주문 없는 예행연습용).

    - 지정가: 주문을 낸 **다음 기록부터**, 그 사이 KRX300 거래가 있었고(누적 거래량 증가) 마지막 체결가가
      내 가격 이하(매수)·이상(매도)이면 내 가격에 전량 체결.
    - 시장가: 그 기록의 상대 최우선 호가에 전량 체결. 호가가 없으면 거부.
    - 상대 최우선 호가를 넘기는 지정가: 그 호가에 바로 전량 체결.
    대기열 순서, 기록 사이의 순서, 내 주문이 남에게 준 영향은 반영하지 못합니다.
    """

    def __init__(self):
        self.snap: dict | None = None
        self.prev_vol: float | None = None
        self.traded = False
        self.orders: dict[str, tuple[int, float]] = {}
        self.resting: dict[str, tuple[str, str, int, float]] = {}
        self.sent: list[tuple] = []
        self._n = 0

    def advance(self, snap: dict) -> None:
        """다음 기록으로 넘어가며 걸려 있던 지정가의 체결을 판정."""
        x = snap.get(KRX300, {})
        vol, last = x.get("vol"), x.get("last")
        self.traded = self.prev_vol is not None and vol is not None and vol > self.prev_vol
        if self.traded and last is not None:
            for no, (sym, side, qty, px) in list(self.resting.items()):
                if sym == KRX300 and ((side == "buy" and last <= px + 1e-9) or (side == "sell" and last >= px - 1e-9)):
                    self.orders[no] = (qty, px)
                    self.resting.pop(no)
        if vol is not None:
            self.prev_vol = vol
        self.snap = snap

    def best(self, sym: str) -> tuple[float | None, float | None]:
        o = (self.snap or {}).get(sym, {})
        b = max((p for p, _ in o.get("b", [])), default=None)
        a = min((p for p, _ in o.get("a", [])), default=None)
        return b, a

    def send(self, sym: str, side: str, qty: int, price: float | None = None) -> str:
        self._n += 1
        no = f"RP{self._n:05d}"
        self.sent.append((no, sym, side, qty, price))
        bid, ask = self.best(sym)
        px = ask if side == "buy" else bid
        if price is not None:
            if px is not None and ((side == "buy" and price >= px) or (side == "sell" and price <= px)):
                self.orders[no] = (qty, px)
                return no
            self.resting[no] = (sym, side, qty, float(price))
            self.orders[no] = (0, 0.0)
            return no
        if px is None:
            raise RuntimeError(f"{sym} 상대 호가 없음 — 시장가 체결 불가")
        self.orders[no] = (qty, px)
        return no

    def filled(self, no: str) -> tuple[int, float]:
        return self.orders.get(no, (0, 0.0))

    def cancel(self, no: str, sym: str, side: str) -> None:
        self.resting.pop(no, None)

    def open_qty(self, no: str) -> int:
        return self.resting[no][2] if no in self.resting else 0


def quote_from_snap(snap: dict) -> Quote:
    t = dt.datetime.fromisoformat(snap["t"]).replace(tzinfo=None)

    def best(sym):
        o = snap.get(sym, {})
        return max((p for p, _ in o.get("b", [])), default=None), min((p for p, _ in o.get("a", [])), default=None)

    kb, ka = best(K200)
    xb, xa = best(KRX300)
    return Quote(t, kb, ka, xb, xa)


def replay(path: str, params: Params, log=None) -> tuple[OpenLimit, ReplayBroker]:
    """녹화 파일을 처음부터 흘려 실행기를 돌립니다."""
    broker = ReplayBroker()
    eng = OpenLimit(params, broker, log=log, sleep=lambda s: None)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                snap = json.loads(ln)
            except Exception:
                continue
            broker.advance(snap)
            eng.step(quote_from_snap(snap))
            if eng.state == "끝":
                break
    return eng, broker


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="녹화한 호가로 장 시작 지정가 전략을 재생합니다 (주문 없음).")
    ap.add_argument("book", help="book_rec.py 가 만든 jsonl")
    ap.add_argument("--ratio", type=float, required=True, help="전일 15:00~15:19 KRX300/코스피200 비율 중앙값")
    ap.add_argument("--d", type=int, default=30, help="적정가에서 몇 틱")
    ap.add_argument("--requote", type=int, default=2, help="목표가가 몇 틱 넘게 바뀌면 다시 걸지")
    ap.add_argument("--out", help="사건 기록을 남길 jsonl")
    a = ap.parse_args()
    rows = []

    def _log(kind, **kw):
        rows.append({"사건": kind, **kw})
        print(kind, kw)

    eng, br = replay(a.book, Params(ratio=a.ratio, d_ticks=a.d, requote_ticks=a.requote), log=_log)
    res = eng.pnl()
    print("결과", json.dumps(res, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"사건": "결과", **res}, ensure_ascii=False) + "\n")
