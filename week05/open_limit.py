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
    exit: dt.time = dt.time(9, 15, 0)     # 이때 두 다리를 시장가로 정리
    requote_ticks: int = 2                # 목표가가 이만큼 넘게 바뀌면 다시 걺
    max_k200_spread: float = 0.5          # 코스피200 호가가 이보다 벌어져 있으면 적정가를 믿지 않고 쉼 (포인트)
    fill_wait: float = 6.0                # 시장가 주문 체결을 기다리는 시간(초)
    fill_poll: float = 1.0


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


@dataclass
class OpenLimit:
    """상태: 대기 → 호가중 → (철수확인) → 보유 → 끝.  하루에 한 번만 포지션을 잡습니다.

    내가 낸 지정가는 취소한 뒤에도 끝까지 체결 여부를 다시 봅니다 — 취소 요청과 체결이 엇갈릴 수 있어서입니다.
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
    note: str = ""
    _t: str = ""

    def _log(self, kind: str, **kw) -> None:
        if self.log:
            self.log(kind, t=self._t, **kw)

    def _note(self, s: str) -> None:
        self.note = (self.note + " / " if self.note else "") + s

    # ── 주문 묶음 ──────────────────────────────────────────────
    def _cancel(self, side: str, why: str) -> None:
        r = self.resting.pop(side, None)
        if r is None:
            return
        try:
            self.broker.cancel(r.no, KRX300, side)
            self._log("취소", side=side, 가격=r.price, 주문번호=r.no, 이유=why)
        except Exception as e:      # 이미 체결됐거나 없는 주문이면 오류가 정상 — 체결 여부는 _new_fills 가 계속 확인
            self._log("취소응답", side=side, 주문번호=r.no, 오류=str(e)[:120])

    def _cancel_all(self, why: str) -> None:
        for side in list(self.resting):
            self._cancel(side, why)

    def _place(self, side: str, price: float, fair: float) -> None:
        try:
            no = self.broker.send(KRX300, side, self.p.qty, price=price)
        except Exception as e:
            self._log("주문거부", side=side, 가격=price, 오류=str(e)[:200])
            return
        self.resting[side] = Resting(no, side, price)
        self.limits.append((no, side, price))
        self._log("지정가", side=side, 가격=price, 적정가=round(fair, 2), 주문번호=no)

    def _market(self, sym: str, side: str, qty: int, kind: str) -> dict:
        """시장가를 내고 체결을 잠깐 기다림. 거부·미체결은 숨기지 않고 돌려줌."""
        out = {"sym": sym, "side": side, "qty": 0, "price": 0.0, "주문번호": None, "주문수량": qty}
        try:
            no = self.broker.send(sym, side, qty)
        except Exception as e:
            self._log(kind + "거부", sym=sym, side=side, 수량=qty, 오류=str(e)[:200])
            return out
        out["주문번호"] = no
        waited = 0.0
        while True:
            try:
                got, px = self.broker.filled(no)
            except Exception as e:
                got, px = 0, 0.0
                self._log("조회실패", 주문번호=no, 오류=str(e)[:150])
            if got >= qty or waited >= self.p.fill_wait:
                break
            (self.sleep or _time.sleep)(self.p.fill_poll)
            waited += self.p.fill_poll
        out["qty"], out["price"] = got, px
        if got:
            self.fills.append((sym, side, got, px, kind))
        self._log(kind, sym=sym, side=side, 주문수량=qty, 체결수량=got, 체결가=px, 주문번호=no)
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
        h = self._market(MINI, opposite(side), hq, "헤지")
        self.hedge = {"side": h["side"], "qty": h["qty"], "price": h["price"], "주문번호": h["주문번호"], "주문수량": hq}
        if h["주문번호"] is None:
            self._note("헤지 주문 거부 — KRX300 한 다리만 보유 중")
        elif h["qty"] < hq:
            self._note(f"헤지 미완료 — 미니 {hq - h['qty']}계약 남음")
        self.state = "보유"

    def _exit(self, q: Quote) -> None:
        legs = [(KRX300, opposite(self.entry["side"]), self.entry["qty"])]
        if self.hedge and self.hedge["qty"] > 0:
            legs.append((MINI, opposite(self.hedge["side"]), self.hedge["qty"]))
        for sym, side, qty in legs:
            e = self._market(sym, side, qty, "청산")
            self.exits.append(e)
            if e["주문번호"] is None:
                self._note(f"청산 주문 거부 — {sym}")
            elif e["qty"] < qty:
                self._note(f"청산 미완료 — {sym} {qty - e['qty']}계약 남음")
        self.state = "끝"

    # ── 한 번의 판단 ────────────────────────────────────────────
    def step(self, q: Quote) -> None:
        now = q.t.time()
        self._t = q.t.strftime("%H:%M:%S")
        if self.state == "끝":
            return
        if self.state == "보유":
            if now >= self.p.exit:
                self._exit(q)
            return
        if self.state in ("호가중", "철수확인"):                  # 낸 주문이 체결됐나부터
            new = self._new_fills()
            if new:
                self._on_fills(q, new)
                if self.state == "보유" and now >= self.p.exit:
                    self._exit(q)
                return
        if self.state == "철수확인":                              # 취소를 낸 뒤 한 번 더 봤는데 체결 없음
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
            if r:
                self._cancel(side, "적정가 이동")
            self._place(side, px, fair)
        self.state = "호가중"

    # ── 결과 ───────────────────────────────────────────────────
    def pnl(self) -> dict:
        """체결가만으로 계산한 손익(원). 수수료 제외. 포지션이 남아 있으면 손익을 확정하지 않음."""
        if not self.fills:
            return {"거래": 0, "손익": 0.0, "메모": self.note}
        cash, net, by_sym = 0.0, {}, {}
        for sym, side, qty, px, _ in self.fills:
            sgn = 1 if side == "buy" else -1
            cash -= sgn * qty * px * MULT[sym]
            by_sym[sym] = by_sym.get(sym, 0.0) - sgn * qty * px * MULT[sym]
            net[sym] = net.get(sym, 0) + sgn * qty
        left = {s_: n for s_, n in net.items() if n}
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
        if price is not None:
            self.resting[no] = (sym, side, qty, float(price))
            self.orders[no] = (0, 0.0)
            return no
        bid, ask = self.best(sym)
        px = ask if side == "buy" else bid
        if px is None:
            raise RuntimeError(f"{sym} 상대 호가 없음 — 시장가 체결 불가")
        self.orders[no] = (qty, px)
        return no

    def filled(self, no: str) -> tuple[int, float]:
        return self.orders.get(no, (0, 0.0))

    def cancel(self, no: str, sym: str, side: str) -> None:
        self.resting.pop(no, None)


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
