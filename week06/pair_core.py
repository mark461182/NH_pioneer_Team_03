"""삼성전자–SK하이닉스 주식선물 페어 — 네트워크 없이 도는 핵심 로직.

실행기(pair_spread.py)와 백테스트(backtest_pair.py)가 이 파일 하나를 같이 씁니다.
같은 규칙이 두 곳에서 따로 구현되면 「백테스트는 A, 실행은 B」가 되기 쉬워서입니다.

    스프레드  s_t = ln(A_t) − β·ln(B_t)          A = 삼성전자 선물, B = SK하이닉스 선물
    β        전 거래일까지 N일 1분 수익률 회귀 기울기 (그날 장중엔 고정 — 미래 정보 없음)
    z        (s_t − 이동평균) / 이동표준편차, 창 W분 (날을 넘겨 이어 계산)
    상관      1분 수익률 이동상관, 창 W분 (밤사이 갭 수익률은 뺌)

    z > +ENTRY → 스프레드 비쌈 → A 매도 · B 매수   (「숏 스프레드」, dir = −1)
    z < −ENTRY → 스프레드 쌈   → A 매수 · B 매도   (「롱 스프레드」, dir = +1)
    |z| < EXIT → 수렴 청산
    |z| > STOP · 손실 한도 · 보유시간 · 장 마감 · 상관 붕괴 → 위험 청산
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace as dataclasses_replace
from datetime import time as dtime

import numpy as np
import pandas as pd


# ── 상품 ───────────────────────────────────────────────────────
MULT = 10  # 주식선물 거래승수(10주). 실행기는 종목 마스터의 ctrt_size 로 덮어씁니다.


def tick_size(p: float) -> float:
    """주식선물 호가단위 = 기초주식(유가증권) 호가단위 (2023-01 개편 뒤 같음).

    README 9-1절: 실행 전 실제 호가로 한 번 더 확인할 항목.
    """
    for bound, tick in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
                        (200_000, 100), (500_000, 500)):
        if p < bound:
            return tick
    return 1_000


# ── 파라미터 ──────────────────────────────────────────────────
@dataclass(frozen=True)
class Params:
    beta_days: int = 5          # β 추정에 쓰는 지난 거래일 수
    window: int = 120           # z·상관 이동창 (분)
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 3.5           # |z| 가 이만큼 벌어지면 손절
    min_corr: float = 0.5       # 진입 허용 최소 상관
    break_corr: float = 0.2     # 보유 중 이 밑으로 떨어지면 청산
    max_hold: int = 90          # 최대 보유 (분)
    cooldown: int = 10          # 청산 뒤 재진입 금지 (분)
    loss_limit: float = 300_000 # 한 거래 평가손실 한도 (원)
    day_loss_limit: float = 600_000  # 하루 실현손실 한도 (원) — 넘으면 그날 신규 금지
    no_entry_before: dtime = dtime(9, 15)
    no_entry_after: dtime = dtime(14, 50)
    flat_at: dtime = dtime(15, 15)   # 강제 청산 (주식선물 단일가 15:35 전에 끝냄)
    unit_b: int = 1             # B(하이닉스) 계약 수. A 는 β 로 맞춤
    max_a: int = 30             # A 계약 수 상한


# ── 통계 ──────────────────────────────────────────────────────
def intraday_returns(px: pd.Series) -> pd.Series:
    """1분 로그수익률. 날이 바뀌는 첫 봉(밤사이 갭)은 NaN."""
    r = np.log(px).diff()
    day = px.index.normalize()
    r[day != pd.Series(day, index=px.index).shift(1).values] = np.nan
    return r


def estimate_beta(ra: pd.Series, rb: pd.Series) -> float:
    m = pd.concat([ra, rb], axis=1).dropna()
    if len(m) < 100 or m.iloc[:, 1].var() == 0:
        return float("nan")
    return float(m.cov().iloc[0, 1] / m.iloc[:, 1].var())


def daily_betas(a: pd.Series, b: pd.Series, days: int) -> pd.Series:
    """날짜 → 그날 쓸 β (전날까지 `days` 거래일로 계산). 첫 `days` 일은 NaN."""
    ra, rb = intraday_returns(a), intraday_returns(b)
    dates = sorted(set(a.index.normalize()))
    out = {}
    for i, d in enumerate(dates):
        if i < days:
            out[d] = float("nan")
            continue
        lo, hi = dates[i - days], d
        sel = (ra.index >= lo) & (ra.index < hi)
        out[d] = estimate_beta(ra[sel], rb[sel])
    return pd.Series(out)


def features(a: pd.Series, b: pd.Series, p: Params, betas: pd.Series | None = None) -> pd.DataFrame:
    """1분 종가 두 줄 → β·스프레드·z·상관 표. 인덱스는 시각(같은 분끼리 맞춘 것)."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if betas is None:
        betas = daily_betas(df["a"], df["b"], p.beta_days)
    df["beta"] = df.index.normalize().map(betas).astype(float)
    df["spread"] = np.log(df["a"]) - df["beta"] * np.log(df["b"])
    # β 가 날마다 바뀌면 스프레드 수준도 바뀌므로, z 는 「그날 β 로 다시 계산한 과거 스프레드」 기준
    roll_mean = np.empty(len(df)); roll_std = np.empty(len(df))
    la, lb, beta = np.log(df["a"].values), np.log(df["b"].values), df["beta"].values
    w = p.window
    for i in range(len(df)):
        if i < w or not math.isfinite(beta[i]):
            roll_mean[i] = roll_std[i] = np.nan
            continue
        hist = la[i - w:i] - beta[i] * lb[i - w:i]   # 직전 W분 (현재 봉 제외)
        roll_mean[i], roll_std[i] = hist.mean(), hist.std(ddof=1)
    df["z"] = (df["spread"] - roll_mean) / roll_std
    df["sd"] = roll_std                                 # 스프레드 σ — 비용 하한 계산에 씀
    ra, rb = intraday_returns(df["a"]), intraday_returns(df["b"])
    df["corr"] = ra.rolling(w, min_periods=w // 2).corr(rb)
    return df


def hedge_qty(beta: float, pa: float, pb: float, p: Params) -> int:
    """B 를 unit_b 계약 둘 때 금액 기준으로 맞는 A 계약 수.

    스프레드 수익률 = rA − β·rB  →  A 금액 : B 금액 = 1 : β
    B 금액 = unit_b·pb·MULT  →  A 금액 = B 금액 / β  →  A 계약 = unit_b·pb / (β·pa)
    """
    if not (beta > 0 and pa > 0 and pb > 0):
        return 0
    return int(max(1, min(p.max_a, round(p.unit_b * pb / (beta * pa)))))


# ── 신호 상태기계 ─────────────────────────────────────────────
@dataclass
class Position:
    dir: int                 # +1 롱 스프레드(A 매수·B 매도) / −1 숏 스프레드
    qa: int                  # A 계약 수 (양수)
    qb: int                  # B 계약 수 (양수)
    pa: float                # A 평균 진입가
    pb: float                # B 평균 진입가
    t_in: pd.Timestamp
    z_in: float
    beta: float

    def pnl(self, pa: float, pb: float, mult: int = MULT) -> float:
        """평가손익(원, 비용 전)."""
        return self.dir * (self.qa * (pa - self.pa) - self.qb * (pb - self.pb)) * mult


@dataclass
class Decision:
    action: str              # "enter" | "exit" | "hold" | "none"
    dir: int = 0
    reason: str = ""


@dataclass
class Signal:
    """봉 하나마다 decide() 를 부르면 무엇을 할지 알려 줍니다. 주문은 안 냅니다."""
    p: Params
    pos: Position | None = None
    last_exit: pd.Timestamp | None = None
    day: object = None
    day_pnl: float = 0.0
    halted: bool = False     # 하루 손실 한도 도달 → 그날 신규 금지 (다음 날 풀림)
    blocked: bool = False    # 레그 실패·잔고 불일치 → 사람이 확인할 때까지 신규 금지 (날이 바뀌어도 안 풀림)

    def new_day(self, d) -> None:
        if d != self.day:
            self.day, self.day_pnl, self.halted = d, 0.0, False

    def decide(self, t: pd.Timestamp, z: float, corr: float, pa: float, pb: float,
               mult: int = MULT, blocked_dirs: frozenset = frozenset()) -> Decision:
        """blocked_dirs: 추세 필터가 막은 진입 방향 (+1 롱 스프레드 / −1 숏 스프레드). 청산에는 영향 없음."""
        self.new_day(t.date())
        p, tt = self.p, t.time()
        if self.pos is not None:
            pos = self.pos
            held = (t - pos.t_in).total_seconds() / 60
            if tt >= p.flat_at:
                return Decision("exit", reason="장마감 청산")
            if pos.pnl(pa, pb, mult) <= -p.loss_limit:
                return Decision("exit", reason="손실한도")
            if math.isfinite(z) and abs(z) >= p.stop and z * pos.dir < 0:
                return Decision("exit", reason="z 손절")
            if math.isfinite(corr) and corr < p.break_corr:
                return Decision("exit", reason="상관 붕괴")
            if held >= p.max_hold:
                return Decision("exit", reason="보유시간")
            if math.isfinite(z) and z * pos.dir >= -p.exit:
                return Decision("exit", reason="수렴")
            return Decision("hold")

        if self.halted or self.blocked or not math.isfinite(z) or not math.isfinite(corr):
            return Decision("none")
        if not (p.no_entry_before <= tt < p.no_entry_after):
            return Decision("none")
        if self.last_exit is not None and (t - self.last_exit).total_seconds() < p.cooldown * 60 \
                and self.last_exit.date() == t.date():
            return Decision("none")
        if corr < p.min_corr:
            return Decision("none", reason="상관 낮음")
        if abs(z) >= p.stop:          # 이미 손절선 밖이면 들어가지 않음
            return Decision("none", reason="z 과대")
        if z >= p.entry:
            if -1 in blocked_dirs:
                return Decision("none", reason="추세 반대 (숏 스프레드 막힘)")
            return Decision("enter", dir=-1, reason=f"z {z:+.2f} ≥ {p.entry}")
        if z <= -p.entry:
            if +1 in blocked_dirs:
                return Decision("none", reason="추세 반대 (롱 스프레드 막힘)")
            return Decision("enter", dir=+1, reason=f"z {z:+.2f} ≤ −{p.entry}")
        return Decision("none")

    def opened(self, pos: Position) -> None:
        self.pos = pos

    def closed(self, t: pd.Timestamp, realized: float) -> None:
        self.pos, self.last_exit = None, t
        self.day_pnl += realized
        if self.day_pnl <= -self.p.day_loss_limit:
            self.halted = True


# ── 두 다리 주문 ──────────────────────────────────────────────
@dataclass
class Leg:
    sym: str
    side: str        # "buy" | "sell"
    qty: int


@dataclass
class LegFill:
    leg: Leg
    filled: int = 0
    avg: float = 0.0
    order_nos: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.filled >= self.leg.qty


@dataclass
class PairResult:
    ok: bool                 # 두 다리 모두 전량 체결
    fills: list[LegFill]
    unwound: list[LegFill]   # 한 다리만 체결돼 되돌린 주문
    note: str = ""
    missed: bool = False     # 지정가 먼저 걸기에서 아무것도 안 체결됨 (노출 없음 — 레그 실패 아님)


def opposite(side: str) -> str:
    return "sell" if side == "buy" else "buy"


class PairExecutor:
    """두 다리를 「같이」 체결시키는 규칙.

    1) 두 다리를 바로 이어서 시장가로 냄 (B 먼저 — 덜 유동적인 쪽을 먼저)
    2) `wait` 초 동안 `poll` 초마다 체결 확인
    3) 둘 다 전량 → 끝
    4) 모자란 다리 → 남은 주문 취소 → 남은 수량만 한 번 더 시장가 (`retries` 번)
    5) 그래도 모자라면 → **이미 체결된 부분을 반대 주문으로 되돌림** (한 다리만 남는 노출 금지)
       되돌린 뒤 두 다리 순수량이 「같은 비율」이 아니면 ok=False 로 알리고 실행기는 신규를 멈춤
    """

    def __init__(self, broker, *, wait: float = 6.0, poll: float = 1.0, retries: int = 1,
                 sleep=None, log=None):
        import time as _t
        self.b, self.wait, self.poll, self.retries = broker, wait, poll, retries
        self.sleep = sleep or _t.sleep
        self.log = log or (lambda *a, **k: None)

    def _send(self, lf: LegFill, qty: int, why: str) -> str | None:
        try:
            no = self.b.send(lf.leg.sym, lf.leg.side, qty)
            lf.order_nos.append(no)
            self.log("주문", sym=lf.leg.sym, side=lf.leg.side, qty=qty, 주문번호=no, 이유=why)
            return no
        except Exception as e:  # 거부·연결 끊김 — 멈추지 않고 기록
            lf.errors.append(str(e)[:200])
            self.log("주문거부", sym=lf.leg.sym, side=lf.leg.side, qty=qty, 오류=str(e)[:200], 이유=why)
            return None

    def _refresh(self, lf: LegFill) -> None:
        tot, amt = 0, 0.0
        for no in lf.order_nos:
            try:
                q, px = self.b.filled(no)
            except Exception as e:
                lf.errors.append("조회 " + str(e)[:150])
                self.log("조회실패", 주문번호=no, 오류=str(e)[:150])
                continue
            tot += q; amt += q * px
        lf.filled, lf.avg = tot, (amt / tot if tot else 0.0)

    def _wait_fills(self, fills: list[LegFill]) -> None:
        t = 0.0
        while t < self.wait:
            self.sleep(self.poll); t += self.poll
            for lf in fills:
                self._refresh(lf)
            if all(lf.done for lf in fills):
                return

    def _cancel_rest(self, lf: LegFill) -> None:
        for no in lf.order_nos:
            try:
                self.b.cancel(no, lf.leg.sym, lf.leg.side)
            except Exception as e:  # 이미 체결·취소된 주문이면 오류가 정상
                self.log("취소응답", 주문번호=no, 오류=str(e)[:120])

    def execute(self, legs: list[Leg], why: str, unwind: bool = True) -> PairResult:
        """unwind=True(진입): 한쪽만 체결되면 되돌림. False(청산): 되돌리지 않고 남은 수량을 알려 줌
        — 청산에서 되돌리면 포지션을 다시 여는 셈이라, 실행기가 남은 다리만 다음 루프에 다시 냄."""
        fills = [LegFill(l) for l in legs]
        for lf in fills:
            self._send(lf, lf.leg.qty, why)
        self._wait_fills(fills)

        for attempt in range(self.retries):
            short = [lf for lf in fills if not lf.done]
            if not short:
                break
            for lf in short:
                self._cancel_rest(lf)
            self.sleep(self.poll)
            for lf in short:
                self._refresh(lf)
                rest = lf.leg.qty - lf.filled
                if rest > 0:
                    self._send(lf, rest, why + f" · 재주문{attempt + 1}")
            self._wait_fills(short)

        if all(lf.done for lf in fills):
            return PairResult(True, fills, [])

        # 한쪽(또는 둘 다)이 덜 채워짐 → 체결된 만큼 되돌려 노출을 없앰
        for lf in fills:
            self._cancel_rest(lf)
            self._refresh(lf)
        if not unwind:
            note = "청산 미완료 — 남은 수량 " + ", ".join(f"{lf.leg.sym} {lf.leg.qty - lf.filled}" for lf in fills if not lf.done)
            self.log("청산미완료", 결과=note)
            return PairResult(False, fills, [], note)
        unwound = []
        for lf in fills:
            if lf.filled > 0:
                u = LegFill(Leg(lf.leg.sym, opposite(lf.leg.side), lf.filled))
                self._send(u, lf.filled, why + " · 레그실패 되돌림")
                self._wait_fills([u])
                unwound.append(u)
        flat = all(u.done for u in unwound)
        note = "레그실패 — 체결분 되돌림 완료" if flat else "레그실패 — 되돌림 미완료, HTS 확인 필요"
        self.log("레그실패", 결과=note,
                 체결=[(lf.leg.sym, lf.leg.side, lf.filled, lf.leg.qty) for lf in fills])
        return PairResult(False, fills, unwound, note)


    # ── 비싼 다리(하이닉스)를 지정가로 먼저 (README 3-8) ──
    def execute_passive_first(self, passive: Leg, limit_px: float, hedge: Leg, why: str,
                              wait: float = 30.0, unwind: bool = True) -> PairResult:
        """하이닉스를 **유리한 쪽 최우선 호가에 지정가**로 걸고(매수=매수1호가, 매도=매도1호가), 체결된 만큼
        삼성을 시장가로 붙임. 호가 차이 1틱(하이닉스 0.18%)을 내는 대신 받는 쪽이 됨.

        - `wait` 초 안에 하나도 안 체결 → 취소, missed=True (노출 없음, 신호를 놓친 것뿐)
        - 일부 체결 → 남은 건 취소, 체결 비율만큼 삼성 수량을 줄여 붙임
        - 삼성이 안 붙으면 → unwind=True(진입)면 하이닉스 체결분을 시장가로 되돌림 / False(청산)면 남김
        """
        pf = LegFill(passive)
        try:
            no = self.b.send(passive.sym, passive.side, passive.qty, price=limit_px)
            pf.order_nos.append(no)
            self.log("지정가", sym=passive.sym, side=passive.side, qty=passive.qty, 가격=limit_px, 주문번호=no, 이유=why)
        except Exception as e:
            pf.errors.append(str(e)[:200])
            self.log("주문거부", sym=passive.sym, side=passive.side, 가격=limit_px, 오류=str(e)[:200], 이유=why)
            return PairResult(False, [pf, LegFill(hedge)], [], "지정가 거부 — 거래 없음", missed=True)
        t = 0.0
        while t < wait:
            self.sleep(self.poll); t += self.poll
            self._refresh(pf)
            if pf.done:
                break
        if not pf.done:
            self._cancel_rest(pf)
            self.sleep(self.poll)
            self._refresh(pf)                     # 취소 직전 체결분까지
        if pf.filled == 0:
            self.log("지정가미체결", sym=passive.sym, 기다린초=t, 이유=why)
            return PairResult(False, [pf, LegFill(hedge)], [], "지정가 미체결 — 거래 없음", missed=True)
        hq = round(hedge.qty * pf.filled / passive.qty)
        if hq == 0:                                # 삼성 9계약 중 4계약만 체결 → 하이닉스 0.44계약은 못 붙임 → 되돌림
            u = LegFill(Leg(passive.sym, opposite(passive.side), pf.filled))
            self._send(u, pf.filled, why + " · 일부 체결이 적어 되돌림")
            self._wait_fills([u])
            self.log("지정가일부되돌림", 체결=pf.filled, 주문=passive.qty, 되돌림완료=u.done)
            return PairResult(False, [pf, LegFill(hedge)], [u], "지정가 일부 체결이 1단위 안 됨 — 되돌림", missed=u.done)
        hres = self.execute([Leg(hedge.sym, hedge.side, hq)], why + " · 지정가 체결 뒤 헤지", unwind=False)
        hf = hres.fills[0]
        if hf.done:
            return PairResult(True, [pf, hf], [])
        if not unwind:
            return PairResult(False, [pf, hf], [], f"청산 미완료 — {hedge.sym} 남은 {hq - hf.filled}")
        unwound = []
        for lf in (pf, hf):
            if lf.filled > 0:
                u = LegFill(Leg(lf.leg.sym, opposite(lf.leg.side), lf.filled))
                self._send(u, lf.filled, why + " · 헤지 실패 되돌림")
                self._wait_fills([u])
                unwound.append(u)
        flat = all(u.done for u in unwound)
        note = "레그실패(헤지) — 되돌림 완료" if flat else "레그실패(헤지) — 되돌림 미완료, HTS 확인 필요"
        self.log("레그실패", 결과=note)
        return PairResult(False, [pf, hf], unwound, note)


class PaperFillBroker:
    """시세를 받아 그 자리에서 체결시키는 종이 브로커 (시뮬레이션·시험용).

    매수 = 매도호가, 매도 = 매수호가. `reject` 에 종목을 넣으면 그 종목 주문을 거부,
    `partial` 에 {종목: 비율} 을 넣으면 그만큼만 체결 — 레그 실패 시험용.
    """

    def __init__(self, quote, reject: set | None = None, partial: dict | None = None):
        self.quote, self.reject, self.partial = quote, set(reject or ()), dict(partial or {})
        self.orders: dict[str, tuple[int, float]] = {}
        self.net: dict[str, int] = {}
        self.log: list[tuple] = []
        self._n = 0

    def send(self, sym: str, side: str, qty: int, price: float | None = None) -> str:
        """price 없음 = 시장가(바로 체결). price 있음 = 지정가 — 상대 호가가 그 가격에 오면 체결(조회할 때 판정)."""
        if sym in self.reject:
            raise RuntimeError(f"시험 거부 {sym}")
        self._n += 1
        no = f"SIM{self._n:06d}"
        if price is not None:
            self.resting = getattr(self, "resting", {})
            self.resting[no] = (sym, side, qty, float(price))
            self.orders[no] = (0, 0.0)
            self.log.append((no, sym, side, qty, 0))
            return no
        bid, ask = self.quote(sym)
        q = int(qty * self.partial.get(sym, 1.0))
        self.orders[no] = (q, ask if side == "buy" else bid)
        self.net[sym] = self.net.get(sym, 0) + (q if side == "buy" else -q)
        self.log.append((no, sym, side, qty, q))
        return no

    def net_positions(self) -> dict[str, int]:
        return dict(self.net)

    def filled(self, no: str) -> tuple[int, float]:
        r = getattr(self, "resting", {}).get(no)
        if r is not None:                       # 걸린 지정가: 상대 호가가 닿았나 (시뮬레이션 — 대기열 순서는 무시)
            sym, side, qty, px = r
            bid, ask = self.quote(sym)
            if (side == "buy" and ask <= px) or (side == "sell" and bid >= px):
                self.resting.pop(no)
                self.orders[no] = (qty, px)
                self.net[sym] = self.net.get(sym, 0) + (qty if side == "buy" else -qty)
        return self.orders.get(no, (0, 0.0))

    def cancel(self, no: str, sym: str, side: str) -> None:
        getattr(self, "resting", {}).pop(no, None)


# ── 비용 ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Costs:
    """왕복 비용 가정. 수수료율은 NH 주식선물 실제 요율을 **모름** → README 에 가정으로 적음."""
    slip_ticks: float = 1.0      # 다리마다 한 번 체결할 때 불리하게 미끄러지는 틱 수 (시장가)
    fee_rate: float = 0.00003    # 체결금액 대비 수수료(편도). 0.003% 가정
    slip_ticks_b: float | None = None   # 하이닉스 다리만 따로 (지정가 먼저 걸기 = 0). None 이면 slip_ticks
    passive: str | None = None          # 지정가로 먼저 걸 다리: "a" 삼성 · "b" 하이닉스 · "auto" 단위당 틱 비용이 큰 쪽 (README 3-8)


def leg_cost(qty: int, px: float, c: Costs, mult: int = MULT, slip: float | None = None) -> float:
    """한 다리 편도 비용(원) = 슬리피지 + 수수료."""
    slip = c.slip_ticks if slip is None else slip
    return qty * mult * (slip * tick_size(px) + c.fee_rate * px)


def passive_leg_of(qa: int, pa: float, qb: int, pb: float, c: Costs) -> str | None:
    """지정가로 먼저 걸 다리. auto = 「계약 수 × 1틱」 이 큰 쪽 (= 시장가로 낼 때 슬리피지가 큰 쪽).

    11/08 실제 데이터: 삼성 약 21만 원(1틱 500원 = 0.23%) × 9계약 vs 하이닉스 약 130만 원(1틱 1,000원 = 0.08%) × 1계약
    → 1단위 편도 슬리피지 삼성 45,000원 vs 하이닉스 10,000원 → auto 는 삼성.
    """
    p = c.passive or ("b" if c.slip_ticks_b == 0 else None)
    if p == "auto":
        return "a" if qa * tick_size(pa) >= qb * tick_size(pb) else "b"
    return p


def leg_slips(qa: int, pa: float, qb: int, pb: float, c: Costs) -> tuple[float, float]:
    """(삼성 슬리피지 틱, 하이닉스 슬리피지 틱) — 지정가로 거는 다리는 0."""
    leg = passive_leg_of(qa, pa, qb, pb, c)
    sa = 0.0 if leg == "a" else c.slip_ticks
    sb = 0.0 if leg == "b" else (c.slip_ticks if c.slip_ticks_b is None else c.slip_ticks_b)
    return sa, sb


def round_trip_cost(qa: int, qb: int, pa: float, pb: float, c: Costs, mult: int = MULT) -> float:
    sa, sb = leg_slips(qa, pa, qb, pb, c)
    return 2 * (leg_cost(qa, pa, c, mult, sa) + leg_cost(qb, pb, c, mult, sb))


def cost_floor_z(sigma: float, qa: int, pa: float, qb: int, pb: float, c: Costs, mult: int = MULT) -> float:
    """진입선 − 청산선이 적어도 이만큼은 돼야 비용을 넘는다 (z 단위).

    스프레드 s = lnA − β·lnB 가 Δs 움직이면 손익 ≈ A 금액 × Δs  (A 금액 = qa·pa·mult)
    → 1σ 움직임의 값어치 = qa·pa·mult·σ.  왕복 비용 ÷ 그 값 = 비용을 메우는 데 필요한 z 폭
    """
    one_sigma = qa * pa * mult * sigma
    return float("inf") if one_sigma <= 0 else round_trip_cost(qa, qb, pa, pb, c, mult) / one_sigma


# ── OU 과정 (진입·청산선을 추정값에서 도출, README 3-7) ──────────────
#
#   스프레드 X 를 dX = κ(μ − X)dt + σ dW 로 봄.  1분 간격 이산화: X_{t+1} = a + b·X_t + ε
#   → κ = −ln b (1/분),  μ = a/(1−b),  σ_eq(정상분포 표준편차) = sd(ε)/√(1−b²),  반감기 = ln2/κ
#   표준화 Y = (X − μ)/σ_eq 는 시간을 1/κ 분 단위로 재면  dY = −Y dτ + √2 dW  (계수 없는 꼴)
#
#   매매 한 주기 = ① 청산 수준 m 에서 다시 ±a 중 하나에 닿을 때까지 기다림 + ② a 에서 m 까지 돌아옴
#   한 주기 이익 = (a − m − c) σ_eq   (c = 왕복 비용을 σ_eq 단위로)
#   → 분당 기대이익 ∝ κ (a − m − c) / (T₁(m→±a) + T₂(a→m))   를 최대로 하는 (a, m)   [Bertram 2010 의 틀]
#
#   평균 첫 도달 시간 (dY = −Y dτ + √2 dW, 생성자 L = −y d/dy + d²/dy², L T = −1 의 해):
#     T₂(a→m), a > m  = ∫_m^a e^{u²/2} ∫_u^∞ e^{−v²/2} dv du
#     T₁(y→±a), |y|<a = ∫_|y|^a e^{u²/2} ∫_0^u e^{−v²/2} dv du
_SQ = math.sqrt(math.pi / 2)


def _outer(lo: float, hi: float, inner, n: int = 2001) -> float:
    if hi <= lo:
        return 0.0
    u = np.linspace(lo, hi, n)
    f = np.exp(u * u / 2) * inner(u)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(u)))


_erf = np.vectorize(math.erf)
_erfc = np.vectorize(math.erfc)


def mfpt_down(a: float, m: float) -> float:
    """표준화 OU 에서 a 에서 출발해 m(<a) 에 처음 닿기까지 기대 시간 (1/κ 단위)."""
    return _outer(m, a, lambda u: _SQ * _erfc(u / math.sqrt(2)))


def mfpt_out(y: float, a: float) -> float:
    """표준화 OU 에서 y(|y|<a) 에서 출발해 +a 또는 −a 에 처음 닿기까지 기대 시간 (1/κ 단위)."""
    return _outer(abs(y), a, lambda u: _SQ * _erf(u / math.sqrt(2)))


@dataclass
class OUFit:
    ok: bool
    kappa: float = float("nan")     # 1/분
    mu: float = float("nan")
    sigma_eq: float = float("nan")
    half_life: float = float("nan") # 분
    n: int = 0
    why: str = ""


def fit_ou(x: np.ndarray, min_n: int = 200) -> OUFit:
    """1분 간격 스프레드 → AR(1) 회귀 → OU 값. 회귀가 안 보이면(b ≥ 1 등) ok=False."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return OUFit(False, n=len(x), why=f"표본 {len(x)} < {min_n}")
    x0, x1 = x[:-1], x[1:]
    vx = x0.var()
    if vx <= 0:
        return OUFit(False, n=len(x), why="변동 없음")
    b = float(((x0 - x0.mean()) * (x1 - x1.mean())).mean() / vx)
    a = float(x1.mean() - b * x0.mean())
    if not (0 < b < 1):
        return OUFit(False, n=len(x), why=f"b={b:.4f} — 평균회귀 안 보임")
    eps = x1 - (a + b * x0)
    k = -math.log(b)
    return OUFit(True, k, a / (1 - b), float(eps.std(ddof=2) / math.sqrt(1 - b * b)), math.log(2) / k, len(x))


@dataclass
class OUPlan:
    ok: bool
    entry: float = float("nan")     # a (σ_eq 단위)
    exit: float = float("nan")      # m (같은 쪽 거리, 음수면 평균 너머까지 보유)
    rate: float = float("nan")      # 분당 기대이익 (σ_eq 단위)
    cycle_min: float = float("nan") # 한 주기 기대 시간 (분)
    cost_sig: float = float("nan")  # c — 왕복 비용 (σ_eq 단위)
    why: str = ""


def ou_plan(fit: OUFit, cost_sig: float, entries=None, exits=None) -> OUPlan:
    """분당 기대이익이 가장 큰 (진입 a, 청산 m). 이익이 0 이하면 ok=False (그날은 거래 안 함)."""
    if not fit.ok:
        return OUPlan(False, cost_sig=cost_sig, why=fit.why)
    entries = np.arange(0.5, 5.01, 0.1) if entries is None else entries
    # 청산은 평균(0) 이나 그 전까지만. 이론상 최적은 평균 너머 반대편까지 들고 가는 것(Bertram: m = −a)
    # 이지만, 장중 평균 μ 를 이동평균으로 근사하므로 평균 너머는 추정오차에 그대로 노출됨 → 보수적으로 막음
    exits = np.arange(0.0, 4.91, 0.1) if exits is None else exits
    best = None
    t1_cache: dict = {}
    for a in entries:
        for m in exits:
            if m >= a - 0.05 or abs(m) >= a:
                continue
            gain = a - m - cost_sig
            if gain <= 0:
                continue
            key = (round(abs(m), 3), round(a, 3))
            if key not in t1_cache:
                t1_cache[key] = mfpt_out(m, a)
            T = t1_cache[key] + mfpt_down(a, m)
            r = gain / T * fit.kappa                     # σ_eq 단위 이익 / 분
            if best is None or r > best[0]:
                best = (r, a, m, T / fit.kappa)
    if best is None:
        return OUPlan(False, cost_sig=cost_sig, why=f"비용 {cost_sig:.2f}σ 를 넘는 진입선 없음")
    r, a, m, cyc = best
    return OUPlan(True, round(float(a), 2), round(float(m), 2), r, cyc, cost_sig)


def fit_ou_days(days: list[np.ndarray], min_n: int = 200, min_len: int = 30) -> OUFit:
    """여러 날의 스프레드를 **날마다 평균을 빼고** 이어 붙여 AR(1). 날을 넘는 쌍(밤사이 갭)은 안 씀.

    날마다 스프레드 수준이 밤사이 바뀌므로, 그냥 이어 붙이면 회귀가 느려 보임(b → 1).
    """
    x0s, x1s = [], []
    for x in days:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < min_len:          # 하루 봉 수가 너무 적은 날은 뺌 (1분봉 30개, 30분봉은 5개)
            continue
        x = x - x.mean()
        x0s.append(x[:-1]); x1s.append(x[1:])
    if not x0s:
        return OUFit(False, why="표본 없음")
    x0, x1 = np.concatenate(x0s), np.concatenate(x1s)
    if len(x0) < min_n:
        return OUFit(False, n=len(x0), why=f"표본 {len(x0)} < {min_n}")
    b = float((x0 * x1).sum() / (x0 * x0).sum())
    if not (0 < b < 1):
        return OUFit(False, n=len(x0), why=f"b={b:.4f} — 평균회귀 안 보임")
    eps = x1 - b * x0
    k = -math.log(b)
    return OUFit(True, k, 0.0, float(eps.std(ddof=1) / math.sqrt(1 - b * b)), math.log(2) / k, len(x0))


@dataclass
class DayPlan:
    """그날 쓸 숫자 — 전날까지 데이터로만 정함 (장중 고정)."""
    date: object
    ok: bool
    beta: float = float("nan")
    fit: OUFit | None = None
    plan: OUPlan | None = None
    qa: int = 0
    window: int = 120
    params: Params | None = None
    why: str = ""
    day_trend: int = 0            # 전날까지 3일 스프레드 날평균이 한 방향으로 움직였으면 ±1
    drift_sig: float = float("nan")  # 그 하루 평균 이동 (σ_eq 단위)
    tail: dict = field(default_factory=dict)   # 리스크 레이어용: 지난 며칠 H분(최대 보유) 스프레드 변화의 VaR·ES

    def row(self) -> dict:
        f, pl = self.fit or OUFit(False), self.plan or OUPlan(False)
        return {"date": str(self.date), "ok": self.ok, "beta": round(self.beta, 4), "kappa": round(f.kappa, 5),
                "half_life_min": round(f.half_life, 1), "sigma_eq": f.sigma_eq, "cost_sigma": round(pl.cost_sig, 3),
                "entry": pl.entry, "exit": pl.exit, "cycle_min": round(pl.cycle_min, 1), "qa": self.qa,
                "window": self.window, "day_trend": self.day_trend, "drift_sigma_per_day": round(self.drift_sig, 2),
                "tail_h": self.tail.get("h"), "var99_h": self.tail.get("var"), "es975_h": self.tail.get("es"),
                "why": self.why}


# ── 스프레드 정의 세 가지 (README 3-2·3-11) ─────────────────────
SPREAD_MODES = ("beta", "ratio", "kalman")


def kalman_beta(a: pd.Series, b: pd.Series, beta_drift_day: float = 0.05, lam: float = 0.99,
                init_days: int = 1) -> pd.DataFrame:
    """칼만 필터로 1분 수익률 헤지비율 β_t 를 장중에도 갱신하고, 그 β 로 잔차를 쌓은 스프레드를 만듦.

        관측  rA_t = β_t · rB_t + ε_t          ε 분산 R (지수가중으로 계속 추정, λ = 0.99)
        상태  β_t = β_{t−1} + η_t              η 분산 q = (하루 β 변화 폭 / √390)²   ← 「β 가 하루에 0.05 쯤 움직일 수 있다」
        스프레드 s_t = Σ (rA − β_{t|t−1} · rB)   ← 매 분 **직전까지의** β 로 계산(미래 정보 없음). 장 시작 첫 분(밤사이 갭)은 뺌

    수준 회귀(ln A − β ln B)와 달리 β 가 바뀌어도 스프레드가 튀지 않음 — 수익률 잔차를 쌓기 때문.
    처음 `init_days` 일은 그 기간 회귀로 β 출발값만 정함(그 날들은 plan_days 준비 기간이라 거래 없음).
    """
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    ra, rb = intraday_returns(df["a"]).values, intraday_returns(df["b"]).values
    day = df.index.normalize()
    first = day < sorted(set(day))[min(init_days, len(set(day)) - 1)] if len(set(day)) > 1 else np.ones(len(df), bool)
    m = first & np.isfinite(ra) & np.isfinite(rb)
    beta = float(np.sum(ra[m] * rb[m]) / np.sum(rb[m] ** 2)) if m.sum() > 30 else 1.0
    R = float(np.nanvar(ra[m] - beta * rb[m])) if m.sum() > 30 else 1e-6
    P, q = 0.05, (beta_drift_day / math.sqrt(390)) ** 2
    bp = np.full(len(df), np.nan); sv = np.zeros(len(df)); acc = 0.0
    for t in range(len(df)):
        bp[t] = beta
        if not (math.isfinite(ra[t]) and math.isfinite(rb[t])):
            sv[t] = acc
            continue
        Pp = P + q
        e = ra[t] - beta * rb[t]
        S = rb[t] * rb[t] * Pp + R
        K = Pp * rb[t] / S
        beta, P = beta + K * e, (1 - K * rb[t]) * Pp
        R = max(lam * R + (1 - lam) * (e * e - rb[t] * rb[t] * Pp), 1e-12)
        acc += e
        sv[t] = acc
    return pd.DataFrame({"beta": bp, "s": sv}, index=df.index)


class SpreadView:
    """정의에 따라 「날 d 의 β 로 본 스프레드」 를 돌려줌.

    beta   : s = ln A − β_d · ln B   (β_d = 전날까지 5일 수익률 회귀, 하루 고정)      ← 지금까지의 기본
    ratio  : s = ln A − ln B          (β = 1, 추정 없음 — 기준선)
    kalman : s = 칼만 잔차 누적        (β_t 매 분 갱신)
    """

    def __init__(self, df: pd.DataFrame, mode: str, betas: pd.Series | None, base: Params):
        assert mode in SPREAD_MODES, mode
        self.mode, self.df = mode, df
        self.la, self.lb = np.log(df["a"].values), np.log(df["b"].values)
        self.day = df.index.normalize()
        if mode == "kalman":
            k = kalman_beta(df["a"], df["b"])
            self.kb, self.ks = k["beta"].values, k["s"].values
        elif mode == "beta":
            self.betas = daily_betas(df["a"], df["b"], base.beta_days) if betas is None else betas

    def day_beta(self, d, i_first_today: int | None = None) -> float:
        """그날 계약 수에 쓸 β (칼만은 그날 첫 분 직전 값 = 전날 마지막 갱신값)."""
        if self.mode == "ratio":
            return 1.0
        if self.mode == "beta":
            return float(self.betas.get(d, float("nan")))
        return float(self.kb[i_first_today]) if i_first_today is not None else float("nan")

    def x(self, bt: float, sl: slice) -> np.ndarray:
        """구간의 스프레드 (beta·ratio 는 그날 β 로 다시 계산 — 창이 전날로 넘어가도 같은 β)."""
        if self.mode == "kalman":
            return self.ks[sl]
        return self.la[sl] - bt * self.lb[sl]

    def hedge_beta(self, i: int, bt: float) -> float:
        """i 분에 진입할 때 계약 수에 쓸 β."""
        return float(self.kb[i]) if self.mode == "kalman" else bt


def _day_ranges(day) -> dict:
    out, start = {}, 0
    for i in range(1, len(day) + 1):
        if i == len(day) or day[i] != day[i - 1]:
            out[day[start]] = (start, i)
            start = i
    return out


def plan_days(a: pd.Series, b: pd.Series, base: Params, c: "Costs", fit_days: int = 2,
              min_hl: float = 3.0, max_hl: float = 120.0, max_cycle: float = 780.0,
              mult: int = MULT, betas: pd.Series | None = None, day_trend_sig: float = 1.0,
              tail_days: int = 5, spread: str = "beta", view: "SpreadView | None" = None) -> dict:
    """날짜 → DayPlan.  OU 는 전날까지 `fit_days` 일 스프레드 (정의는 `spread`, README 3-11).

    거래 안 하는 날: OU 회귀가 안 보임 · 반감기 < 3분(20초 REST 로 못 따라감) 또는 > 120분(장중에 안 돌아옴)
                    · 비용 넘는 진입선 없음 · 기대 주기 > 780분(이틀에 한 번도 안 옴)
    """
    from risk_layer import spread_tail
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    V = view or SpreadView(df, spread, betas, base)
    rng = _day_ranges(V.day)
    dates = list(rng.keys())
    need = max(fit_days, base.beta_days if V.mode == "beta" else 1)
    out = {}
    for i, d in enumerate(dates):
        bt = V.day_beta(d, rng[d][0])
        if i < max(fit_days, 1) or not math.isfinite(bt) or (V.mode == "beta" and i < need):
            out[d] = DayPlan(d, False, bt, why="준비 기간 (β·OU 표본 부족)")
            continue
        xs = [V.x(bt, slice(*rng[pd_])) for pd_ in dates[i - fit_days:i]]
        fit = fit_ou_days(xs)
        last = df.iloc[rng[dates[i - 1]][1] - 1]
        qa = hedge_qty(bt, float(last["a"]), float(last["b"]), base)
        if not fit.ok:
            out[d] = DayPlan(d, False, bt, fit, qa=qa, why=fit.why)
            continue
        if qa <= 0 or fit.sigma_eq <= 0:
            out[d] = DayPlan(d, False, bt, fit, qa=qa, why="계약 수 0 (β ≤ 0) 또는 σ 0")
            continue
        cost_sig = round_trip_cost(qa, base.unit_b, float(last["a"]), float(last["b"]), c, mult) / (
            qa * float(last["a"]) * mult * fit.sigma_eq)
        if not (min_hl <= fit.half_life <= max_hl):
            out[d] = DayPlan(d, False, bt, fit, OUPlan(False, cost_sig=cost_sig), qa,
                             why=f"반감기 {fit.half_life:.1f}분 (허용 {min_hl:.0f}~{max_hl:.0f})")
            continue
        pl = ou_plan(fit, cost_sig)
        if not pl.ok or pl.cycle_min > max_cycle:
            out[d] = DayPlan(d, False, bt, fit, pl, qa, why=pl.why or f"기대 주기 {pl.cycle_min:.0f}분 > {max_cycle:.0f}")
            continue
        w = int(min(240, max(30, round(5 * fit.half_life))))
        prm = dataclasses_replace(base, entry=pl.entry, exit=pl.exit, stop=round(pl.entry + 1.5, 2), window=w,
                                  max_hold=int(min(120, max(10, round(3 * fit.half_life)))))
        trend, drift = 0, float("nan")
        if i >= 3:                                    # 아침 추세 점검: 전날까지 3일 날평균
            means = [float(np.mean(V.x(bt, slice(*rng[pd_])))) for pd_ in dates[i - 3:i]]
            d1, d2 = means[1] - means[0], means[2] - means[1]
            drift = (means[2] - means[0]) / 2 / fit.sigma_eq
            if d1 * d2 > 0 and abs(drift) >= day_trend_sig:
                trend = 1 if d2 > 0 else -1
        tail = spread_tail([V.x(bt, slice(*rng[pd_])) for pd_ in dates[max(0, i - tail_days):i]], prm.max_hold)
        out[d] = DayPlan(d, True, bt, fit, pl, qa, w, prm, day_trend=trend, drift_sig=drift, tail=tail)
    return out


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """직전 w 개 평균 (현재 제외). 앞쪽은 NaN."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    if len(x) > w:
        out[w:] = (c[w:-1] - c[:-w - 1]) / w
    return out


def _roll_slope(x: np.ndarray, w: int) -> np.ndarray:
    """직전 w 개에 맞춘 직선의 기울기 (현재 제외)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= w:
        return out
    j = np.arange(n, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)]); cj = np.concatenate([[0.0], np.cumsum(j * x)])
    i = np.arange(w, n)
    sx = c[i] - c[i - w]
    sjx = cj[i] - cj[i - w] - (i - w) * sx            # Σ k·x_{i−w+k}, k = 0..w−1
    kbar = (w - 1) / 2
    out[w:] = (sjx - kbar * sx) / (w * (w * w - 1) / 12)
    return out


def features_ou(a: pd.Series, b: pd.Series, plans: dict, trend_filter: bool = False,
                slope_win: int = 60, spread: str = "beta", view: "SpreadView | None" = None,
                base: Params = Params()) -> pd.DataFrame:
    """OU 모드의 z: (s − 직전 W분 평균) / σ_eq.  W·σ_eq·β 는 그날 DayPlan 것. 거래 안 하는 날은 z = NaN.

    trend_filter=True 면 「추세 반대 진입」 을 막는 칸 block_long / block_short 를 채움 (README 3-9):
      아침 — 전날까지 3일 스프레드 날평균이 두 번 연속 같은 방향, 하루 1σ_eq 이상 → 그날 추세 ±1
      장중 — 직전 `slope_win` 분 스프레드 기울기 × 예상 보유시간 이 왕복 비용(σ_eq 단위, 최소 0.5) 보다 크면 추세 ±1
      추세가 위(+1)면 숏 스프레드(스프레드 하락에 거는 진입)를 막음, 아래(−1)면 롱 스프레드를 막음.
    beta 칸 = 그 분에 진입하면 계약 수에 쓸 β (칼만은 매 분 다름).
    """
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    V = view or SpreadView(df, spread, None, base)
    n = len(df)
    z = np.full(n, np.nan); beta = np.full(n, np.nan); sd = np.full(n, np.nan); spr = np.full(n, np.nan)
    blk_l = np.zeros(n, bool); blk_s = np.zeros(n, bool); slope_sig = np.full(n, np.nan)
    for d, (i0, i1) in _day_ranges(V.day).items():
        pl = plans.get(d)
        if pl is None or not math.isfinite(pl.beta):
            continue
        bt = pl.beta
        beta[i0:i1] = V.kb[i0:i1] if V.mode == "kalman" else bt
        w = pl.window if pl.ok else 120
        lo = max(0, i0 - max(w, slope_win))
        x = V.x(bt, slice(lo, i1))                     # 창이 전날로 넘어가도 그날 β 로
        spr[i0:i1] = x[i0 - lo:]
        if not pl.ok:
            continue
        se = pl.fit.sigma_eq
        sd[i0:i1] = se
        z[i0:i1] = ((x - _roll_mean(x, w)) / se)[i0 - lo:]
        if trend_filter:
            hold = mfpt_down(pl.plan.entry, pl.plan.exit) / pl.fit.kappa       # 예상 보유 (분)
            gate = max(pl.plan.cost_sig, 0.5)
            ss = (_roll_slope(x, slope_win) * hold / se)[i0 - lo:]
            slope_sig[i0:i1] = ss
            up = np.nan_to_num(ss) > gate; dn = np.nan_to_num(ss) < -gate
            blk_s[i0:i1] = up | (pl.day_trend == 1)    # 위로 가는 추세 → 하락에 거는 숏 스프레드 막음
            blk_l[i0:i1] = dn | (pl.day_trend == -1)
    out = df.copy()
    out["beta"], out["z"], out["sd"], out["spread"] = beta, z, sd, spr
    ra, rb = intraday_returns(df["a"]), intraday_returns(df["b"])
    out["corr"] = ra.rolling(120, min_periods=60).corr(rb)
    out["block_long"], out["block_short"], out["slope_sig"] = blk_l, blk_s, slope_sig
    return out


def plan_today(hist_a: pd.Series, hist_b: pd.Series, today, base: Params, c: Costs, mult: int = MULT,
               spread: str = "beta") -> DayPlan:
    """실행기용: 지난 거래일 1분봉만으로 오늘 DayPlan. (오늘 09:00 자리에 전날 종가 한 줄을 붙여 plan_days 를 그대로 씀)"""
    t0 = pd.Timestamp(today) + pd.Timedelta(hours=9)
    a = pd.concat([hist_a, pd.Series([hist_a.iloc[-1]], index=[t0])])
    b = pd.concat([hist_b, pd.Series([hist_b.iloc[-1]], index=[t0])])
    return plan_days(a, b, base, c, mult=mult, spread=spread)[pd.Timestamp(today).normalize()]
