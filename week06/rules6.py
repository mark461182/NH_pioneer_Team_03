"""6주차 위험관리 엔진 — 백테스트(backtest6.py)와 실행기(runner6.py)가 **같은 파일 하나**를 씀.

4주차 pair_core.Signal 은 「한 번에 들어가 한 번에 나오는」 포지션 하나만 알았습니다.
이 엔진은 포지션을 **트랜치(분할 묶음) 여러 개**로 들고, 봉 하나마다 무엇을 할지(주문 목록)만 돌려줍니다.
주문은 내지 않습니다. 체결은 부르는 쪽이 내고 opened()/closed() 로 알려 줍니다.

규칙 (값과 이유는 README 3절)
  ① 손절선      가격: z 가 진입선 + 1.5σ_eq 밖 · 금액: 트랜치 평가손실 ≥ 단위당 30만 원 · 시간: 반감기 × 3
               분할 청산 뒤 남은 트랜치는 「이익 보호선」: z 가 다시 진입선까지 벌어지면 청산
  ② 분할 매수   1차 1단위 at z = a · 2차 1단위 at z = a + 0.5 (같은 방향, 손절선 안, 보유 절반 시간 안)
     분할 매도   1차 at z = (a + m)/2 (중간) · 나머지 at z = m (원래 수렴선)
  ③ 한도        최대 2단위 · 하루 실현손실 −60만 원 → 그날 신규 금지 · 하루 손절 2회 → 그날 신규 금지
               누적 낙폭 −150만 → 2차 분할 금지(최대 1단위) · −250만 → 그날 중단, 다음 날부터 1단위
               −100만 위로 회복하면 원래대로
  ④ 실시간 이상  monitor6.py 가 block()/unblock() 으로 신규만 막음 (청산은 막지 않음)

1단위 = 하이닉스 1계약 + 금액을 맞춘 삼성 계약 (pair_core.hedge_qty).

───────── 11/22 개선 (improved() = 설정 E, README 4-6) — 기본값 Rules6() 는 여전히 설정 D ─────────
  ⑤ 고정 z 손절   z 를 「직전 W분 평균」 이 아니라 **진입 시점 평균 μ₀ 에 고정**해서도 잼: z₀ = (s − μ₀)/σ_eq
                  이동 z 와 고정 z 중 먼저 a+1.5 에 닿는 쪽에서 손절. 2차 진입선(a+0.5)도 고정 z 로
  ⑥ 비율 한도     원 단위 고정 → 진입 명목(두 다리 합) 대비: 금액 손절 1.3% · 하루 2.6% · 낙폭 6.6% / 11% / 회복 4.4%
  ⑦ 재검증 재개   낙폭 중단 뒤 자동 재개 대신 — shadow: 주문 없이 1단위 가상 매매를 이어가다 최근 20건 순손익 합 > 0 이면
                  1단위로 재개(고점을 그 시점으로 다시 잡음) / manual: 사람이 풀 때까지(실행기 --ack)
  ⑧ 그날 쉬기     밤사이 스프레드 갭 |s_시가 − s_전일종가| > 2σ_eq · 주식선물 만기일(둘째 목요일)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import time as dtime

import pandas as pd

import pair_core as pc


# ── 규칙 값 ───────────────────────────────────────────────────
@dataclass(frozen=True)
class Rules6:
    # ① 손절
    price_stop: bool = True
    stop_gap: float = 1.5            # 가격 손절: |z| ≥ 진입선 + stop_gap (σ_eq 단위)
    loss_stop: float | None = 300_000  # 금액 손절: 트랜치 평가손실 ≥ 단위 수 × 이 값 (원)
    time_stop: bool = True           # 보유 ≥ plan.max_hold (= 반감기 × 3)
    corr_break: bool = True          # 1분 상관 < 0.2 → 청산 (4주차와 같음)
    protect: bool = True             # 분할 청산 뒤 남은 것: z 가 진입선까지 되벌어지면 청산
    # ② 분할
    split_entry: bool = True
    add_gap: float = 0.5             # 2차 진입: |z| ≥ 진입선 + add_gap
    add_within: float = 0.5          # 2차는 최대 보유시간의 이 비율 안에서만
    split_exit: bool = True          # 1차 청산을 진입선과 수렴선의 중간에서
    # ③ 한도
    units_max: int = 2
    day_loss: float | None = 600_000
    day_stops: int | None = 2        # 하루 손절 청산 횟수 한도 (가격·금액·시간·상관)
    dd_half: float | None = -1_500_000
    dd_stop: float | None = -2_500_000
    dd_recover: float = -1_000_000
    # ── 11/22 개선 (전부 꺼 둔 것이 설정 D) ──
    anchor_stop: bool = False        # ⑤ 진입 시점 평균에 고정한 z 로도 가격 손절·2차 진입
    loss_stop_pct: float | None = None   # ⑥ 금액 손절 = 보유 트랜치 진입 명목 × 이 값 (주면 loss_stop 대신)
    day_loss_pct: float | None = None    #    하루 손실 한도 = 1단위 명목 × 이 값
    dd_half_pct: float | None = None     #    낙폭 (양수로 적음: 0.066 = 고점 대비 −6.6%)
    dd_stop_pct: float | None = None
    dd_recover_pct: float | None = None
    dd_resume: str = "auto"          # ⑦ auto(다음 날 1단위) | shadow(가상 매매로 재검증) | manual(사람이 풀 때까지)
    shadow_n: int = 20
    gap_skip: float | None = None    # ⑧ 밤사이 갭이 이만큼(σ_eq 배) 넘으면 그날 쉼
    skip_expiry: bool = False        #    주식선물 만기일(둘째 목요일) 쉼
    # 공통 (4주차와 같음)
    no_entry_before: dtime = dtime(9, 15)
    no_entry_after: dtime = dtime(14, 50)
    flat_at: dtime = dtime(15, 15)
    cooldown: int = 10
    min_corr: float = 0.5


def no_risk() -> Rules6:
    """비교 기준 A — 위험관리 없음: 2단위 한 번에, 수렴·장마감으로만 청산, 한도 없음."""
    return Rules6(price_stop=False, loss_stop=None, time_stop=False, corr_break=False, protect=False,
                  split_entry=False, split_exit=False, day_loss=None, day_stops=None, dd_half=None, dd_stop=None)


def week4() -> Rules6:
    """비교 기준 B — 4주차 안전장치 그대로 (2단위 한 번에): 가격·금액·시간·상관 손절 + 하루 손실 한도."""
    return Rules6(protect=False, split_entry=False, split_exit=False, day_stops=None, dd_half=None, dd_stop=None)


def improved(**kw) -> Rules6:
    """설정 E — 11/22 개선 전부. 비율은 2026-11-06 의 1단위 명목(약 2,270만 원)에서 D 의 원 단위 값을 환산."""
    base = dict(anchor_stop=True, loss_stop_pct=0.013, day_loss_pct=0.026, dd_half_pct=0.066, dd_stop_pct=0.11,
                dd_recover_pct=0.044, dd_resume="shadow", gap_skip=2.0, skip_expiry=True)
    return Rules6(**{**base, **kw})


def is_expiry(d) -> bool:
    """주식선물 최종거래일 = 결제월 둘째 목요일 (휴일로 당겨지는 달은 못 잡음)."""
    return d.weekday() == 3 and 8 <= d.day <= 14


def day_skip(rules: Rules6, d, gap_sig: float) -> str:
    """그날 신규를 쉴 이유 (없으면 빈 글). gap_sig = (오늘 첫 봉 스프레드 − 전일 마지막 봉) / σ_eq."""
    if rules.skip_expiry and is_expiry(d):
        return "만기일 (둘째 목요일)"
    if rules.gap_skip is not None and math.isfinite(gap_sig) and abs(gap_sig) > rules.gap_skip:
        return f"갭필터 (밤사이 스프레드 갭 {gap_sig:+.1f}σ)"
    return ""


STOP_REASONS = ("가격손절", "고정z손절", "금액손절", "시간손절", "상관붕괴", "이익보호")


# ── 상태 ──────────────────────────────────────────────────────
@dataclass
class Tranche:
    tid: int
    units: int
    qa: int
    qb: int
    pa: float
    pb: float
    t_in: pd.Timestamp
    z_in: float
    tag: str                          # "1차" | "2차" | "한번에"
    shadow: bool = False              # 가상 매매(낙폭 중단 중 재검증용 — 주문·손익 없음)

    def notional(self, mult: int) -> float:
        return (self.qa * self.pa + self.qb * self.pb) * mult

    def pnl(self, d: int, pa: float, pb: float, mult: int) -> float:
        return d * (self.qa * (pa - self.pa) - self.qb * (pb - self.pb)) * mult


@dataclass
class Order:
    kind: str                         # "open" | "close"
    units: int = 0                    # open: 단위 수
    tids: tuple = ()                  # close: 닫을 트랜치
    dir: int = 0
    why: str = ""
    tag: str = ""


@dataclass
class Engine6:
    rules: Rules6
    p: pc.Params = field(default_factory=pc.Params)   # 그날 OU 숫자(진입·청산·창·보유) — set_day 로 바꿈
    mult: int = pc.MULT
    dir: int = 0
    tranches: list = field(default_factory=list)
    partial_done: bool = False        # 분할 1차 청산을 했나 (이익 보호선 켜짐)
    adds_done: int = 0
    next_tid: int = 1
    last_exit: pd.Timestamp | None = None
    day: object = None
    day_pnl: float = 0.0
    day_stop_n: int = 0
    halted: str = ""                  # 그날 신규 금지 사유 (날 바뀌면 풀림)
    blocked: str = ""                 # 실시간 이상·레그 실패 — 사람이/모니터가 풀 때까지
    equity: float = 0.0
    peak: float = 0.0
    dd_mode: str = "normal"           # normal | half | stop
    dd_stop_day: object = None
    mu0: float = float("nan")         # 진입 시점 평균 (고정 z 기준)
    ref_unit: float = 0.0             # 가장 최근 진입의 1단위 명목 (비율 한도 기준)
    shadow: bool = False              # 낙폭 중단 중 가상 매매 상태
    shadow_pending: bool = False
    shadow_nets: list = field(default_factory=list)
    _shadow_ep: float = 0.0
    _mu0_next: float = float("nan")
    events: list = field(default_factory=list)   # 규칙이 작동한 기록 (README 6절 로그)
    _ignored: set = field(default_factory=set)

    # ── 기록 ──
    def note(self, t, rule: str, **kw):
        self.events.append({"t": t, "규칙": rule, **kw})

    # ── 하루 ──
    def set_day(self, d, params: pc.Params | None, skip: str = ""):
        if d == self.day:
            return
        self.day, self.day_pnl, self.day_stop_n, self.halted = d, 0.0, 0, ""
        if self.dd_mode == "stop" and self.dd_stop_day != d and self.rules.dd_resume == "auto":
            self.dd_mode = "half"
            self.note(pd.Timestamp(d), "낙폭한도", 내용="중단 다음 날 → 최대 1단위로 재개")
        if params is not None:
            self.p = params
        if skip:
            self.halted = skip
            self.note(pd.Timestamp(d), skip.split(" ")[0], 내용=f"그날 신규 진입 쉼 — {skip}")

    def resume(self, t, why: str = "사람 확인"):
        """낙폭 중단을 풂 (manual: 실행기 --ack / shadow: 재검증 통과). 고점을 지금으로 다시 잡고 1단위부터."""
        if self.dd_mode == "stop":
            self.dd_mode, self.shadow, self.shadow_pending, self.peak = "half", False, False, self.equity
            self.shadow_nets, self._shadow_ep = [], 0.0
            self.note(t, "낙폭한도", 내용=f"재개 ({why}) — 최대 1단위, 고점 기준을 지금으로")

    # ── 한도 값 (원) — 비율이 주어지면 명목 대비, 아니면 고정 금액 ──
    def _lim(self, pct, won, sign: float = 1.0):
        if pct is not None:
            return sign * pct * self.ref_unit if self.ref_unit > 0 else None
        return won

    def loss_limit(self):
        R = self.rules
        if R.loss_stop_pct is not None:
            return R.loss_stop_pct * sum(x.notional(self.mult) for x in self.tranches)
        return None if R.loss_stop is None else R.loss_stop * self.units

    # ── 보유 정보 ──
    @property
    def units(self) -> int:
        return sum(t.units for t in self.tranches)

    def open_pnl(self, pa: float, pb: float) -> float:
        return sum(t.pnl(self.dir, pa, pb, self.mult) for t in self.tranches)

    def cap_units(self) -> int:
        """지금 들고 있을 수 있는 최대 단위 (③ 낙폭 상태 반영)."""
        if self.dd_mode == "stop":
            return 1 if self.shadow else 0      # 가상 매매는 1단위로만
        if self.dd_mode == "half":
            return 1
        return self.rules.units_max

    # ── 봉 하나 ──
    def on_bar(self, t: pd.Timestamp, z: float, corr: float, pa: float, pb: float,
               blocked_dirs: frozenset = frozenset(), s: float = float("nan"), se: float = float("nan")) -> list[Order]:
        """s = 그 분 스프레드, se = σ_eq (고정 z 에만 씀 — 안 주면 이동 z 만으로 판단)."""
        R, p, tt = self.rules, self.p, t.time()
        if self.tranches:
            return self._manage(t, z, corr, pa, pb, s, se)
        # ── 신규 진입 ──
        if not (math.isfinite(z) and math.isfinite(corr)):
            return []
        if not (R.no_entry_before <= tt < R.no_entry_after):
            return []
        if self.last_exit is not None and self.last_exit.date() == t.date() and \
                (t - self.last_exit).total_seconds() < R.cooldown * 60:
            return []
        if corr < R.min_corr:
            return []
        stop_z = p.entry + R.stop_gap
        if R.price_stop and abs(z) >= stop_z:       # 이미 손절선 밖이면 안 들어감
            return []
        side = -1 if z >= p.entry else (+1 if z <= -p.entry else 0)
        if side == 0 or side in blocked_dirs:
            return []
        why = self.entry_block()
        if why:
            key = (t.date(), why.split(" ")[0])
            if key not in self._ignored:                 # 같은 날 같은 이유는 한 번만 적음
                self._ignored.add(key)
                self.note(t, why.split(" ")[0], 내용=f"진입 신호 z {z:+.2f} 무시 — {why}")
            return []
        cap = self.cap_units()
        self._mu0_next = s - z * se if (math.isfinite(s) and math.isfinite(se)) else float("nan")
        why_in = f"z {z:+.2f} 진입선 {p.entry}"
        if R.split_entry:
            return [Order("open", units=1, dir=side, why=why_in, tag="1차")]
        if R.split_exit and cap >= 2:        # 분할 청산만 쓸 때: 같은 봉에 1단위씩 두 묶음으로 들어감
            return [Order("open", units=1, dir=side, why=why_in, tag="한번에"),
                    Order("open", units=cap - 1, dir=side, why=why_in, tag="한번에")]
        return [Order("open", units=cap, dir=side, why=why_in, tag="한번에")]

    def entry_block(self) -> str:
        if self.blocked:
            return f"실시간이상 {self.blocked}"
        if self.halted:
            return self.halted
        if self.cap_units() == 0:
            return "낙폭한도 중단"
        return ""

    def _manage(self, t, z, corr, pa, pb, s=float("nan"), se=float("nan")) -> list[Order]:
        R, p, d = self.rules, self.p, self.dir
        z0 = (s - self.mu0) / se if (R.anchor_stop and math.isfinite(s) and math.isfinite(self.mu0) and se > 0) else float("nan")
        out0 = -z0 * d                                            # 고정 z 로 본 「벌어진 정도」 (없으면 NaN)
        all_ids = tuple(x.tid for x in self.tranches)
        held = (t - min(x.t_in for x in self.tranches)).total_seconds() / 60
        zd = z * d if math.isfinite(z) else float("nan")      # 음수 = 아직 벌어진 쪽 (롱 스프레드면 z<0)
        if t.time() >= R.flat_at:
            return [Order("close", tids=all_ids, why="장마감")]
        if R.price_stop and math.isfinite(z) and -zd >= p.entry + R.stop_gap:
            return [Order("close", tids=all_ids, why="가격손절")]
        if R.price_stop and math.isfinite(out0) and out0 >= p.entry + R.stop_gap:
            return [Order("close", tids=all_ids, why="고정z손절")]
        lim = self.loss_limit()
        if lim is not None and lim > 0:
            pnl = self.open_pnl(pa, pb)
            if pnl <= -lim:
                return [Order("close", tids=all_ids, why="금액손절")]
        if R.corr_break and math.isfinite(corr) and corr < p.break_corr:
            return [Order("close", tids=all_ids, why="상관붕괴")]
        if R.time_stop and held >= p.max_hold:
            return [Order("close", tids=all_ids, why="시간손절")]
        if not math.isfinite(z):
            return []
        # 수렴
        if zd >= -p.exit:
            return [Order("close", tids=all_ids, why="수렴")]
        # 분할 1차 청산 (중간선) — 두 트랜치 이상일 때만 나눠 나옴
        mid = (p.entry + p.exit) / 2
        if R.split_exit and not self.partial_done and len(self.tranches) >= 2 and zd >= -mid:
            first = min(self.tranches, key=lambda x: x.tid)
            return [Order("close", tids=(first.tid,), why="분할청산1")]
        # 이익 보호선: 분할 청산 뒤 다시 진입선까지 벌어지면 남은 것 청산
        if R.protect and self.partial_done and -zd >= p.entry:
            return [Order("close", tids=all_ids, why="이익보호")]
        # 분할 2차 진입
        wide = out0 if math.isfinite(out0) else -zd               # 2차 진입선도 고정 z 로 (있으면)
        if R.split_entry and self.adds_done == 0 and not self.partial_done and \
                wide >= p.entry + R.add_gap and held < R.add_within * p.max_hold:
            room = self.cap_units() - self.units
            why = self.entry_block()
            if room <= 0 or why:
                if not getattr(self, "_add_noted", False):
                    self.note(t, "분할제한", 내용=f"2차 진입 자리 z {z:+.2f} — " + (why or f"낙폭 상태 {self.dd_mode} (최대 {self.cap_units()}단위)"))
                    self._add_noted = True
                return []
            return [Order("open", units=1, dir=d, why=f"z {z:+.2f} ≥ 진입선+{R.add_gap}", tag="2차")]
        return []

    # ── 체결 통보 ──
    def opened(self, t, units: int, qa: int, qb: int, pa: float, pb: float, z: float, d: int, tag: str):
        if not self.tranches:
            self.dir, self.partial_done, self.adds_done, self._add_noted = d, False, 0, False
            self.mu0 = self._mu0_next
        if tag == "2차":
            self.adds_done += 1
        self.tranches.append(Tranche(self.next_tid, units, qa, qb, pa, pb, t, z, tag, self.shadow))
        self.next_tid += 1
        if not self.shadow and units > 0:
            self.ref_unit = self.tranches[-1].notional(self.mult) / units
        self.note(t, ("가상" if self.shadow else "") + ("분할매수" if tag == "2차" else "진입"), 트랜치=tag, 단위=units, 삼성=qa, 하이닉스=qb, z=round(z, 2),
                  보유단위=self.units)

    def closed(self, t, tids: tuple, net: float, why: str, z: float = float("nan")):
        """net = 이번에 닫은 트랜치들의 순손익(비용 뒤)."""
        left = [x for x in self.tranches if x.tid not in tids]
        gone = [x for x in self.tranches if x.tid in tids]
        self.tranches = left
        if why == "분할청산1":
            self.partial_done = True
        if gone and all(x.shadow for x in gone):                  # 가상 매매: 손익·한도에 안 넣고 재검증에만
            self._shadow_ep += net
            self.note(t, "가상청산", 사유=why, 가상순손익원=round(net), 남은단위=self.units)
            if not left:
                self.dir, self.last_exit = 0, t
                self.shadow_nets.append(self._shadow_ep)
                self._shadow_ep = 0.0
                n = self.rules.shadow_n
                if len(self.shadow_nets) >= n and sum(self.shadow_nets[-n:]) > 0:
                    self.resume(t, f"가상 매매 최근 {n}건 합 {sum(self.shadow_nets[-n:]):+,.0f}원")
            return
        self.day_pnl += net
        self.equity += net
        self.peak = max(self.peak, self.equity)
        rule = "손절" if why in STOP_REASONS else ("분할청산" if why == "분할청산1" else "청산")
        self.note(t, rule, 사유=why, 단위=sum(x.units for x in gone), 순손익원=round(net), z=None if not math.isfinite(z) else round(z, 2),
                  남은단위=self.units, 일손익원=round(self.day_pnl))
        if why in STOP_REASONS:
            self.day_stop_n += 1
        if not left:
            self.dir, self.last_exit = 0, t
        self._limits(t)
        if not left and self.shadow_pending:
            self.shadow, self.shadow_pending = True, False

    def _limits(self, t):
        R = self.rules
        day_lim = self._lim(R.day_loss_pct, R.day_loss)
        if not self.halted and day_lim is not None and self.day_pnl <= -day_lim:
            self.halted = f"하루손실한도 (일손익 {self.day_pnl:,.0f}원 ≤ −{day_lim:,.0f})"
            self.note(t, "하루손실한도", 내용="그날 신규 진입 중지", 일손익원=round(self.day_pnl))
        if not self.halted and R.day_stops is not None and self.day_stop_n >= R.day_stops:
            self.halted = f"하루손절횟수 ({self.day_stop_n}회)"
            self.note(t, "하루손절횟수", 내용=f"손절 {self.day_stop_n}회 — 그날 신규 진입 중지")
        dd = self.equity - self.peak
        d_half, d_stop = self._lim(R.dd_half_pct, R.dd_half, -1.0), self._lim(R.dd_stop_pct, R.dd_stop, -1.0)
        d_rec = self._lim(R.dd_recover_pct, R.dd_recover, -1.0)
        if d_stop is not None and dd <= d_stop and self.dd_mode != "stop":
            self.dd_mode, self.dd_stop_day = "stop", self.day
            how = {"auto": "그날 중단", "shadow": "중단 — 가상 매매로 재검증", "manual": "중단 — 사람이 풀 때까지"}[R.dd_resume]
            self.note(t, "낙폭한도", 내용=f"고점 대비 {dd:,.0f}원 — {how}", 고점대비원=round(dd))
            if R.dd_resume == "shadow":
                if self.tranches:
                    self.shadow_pending = True
                else:
                    self.shadow = True
        elif d_half is not None and dd <= d_half and self.dd_mode == "normal":
            self.dd_mode = "half"
            self.note(t, "낙폭한도", 내용=f"고점 대비 {dd:,.0f}원 — 최대 1단위(2차 분할 금지)", 고점대비원=round(dd))
        elif d_half is not None and d_rec is not None and dd > d_rec and self.dd_mode == "half":
            self.dd_mode = "normal"
            self.note(t, "낙폭한도", 내용=f"고점 대비 {dd:,.0f}원으로 회복 — 원래대로", 고점대비원=round(dd))

    # ── 실시간 이상 (monitor6) ──
    def block(self, t, why: str):
        if not self.blocked:
            self.blocked = why
            self.note(t, "실시간이상", 내용=f"신규 중지 — {why}")

    def unblock(self, t, why: str = ""):
        if self.blocked:
            self.note(t, "실시간회복", 내용=f"신규 재개 — {why or self.blocked} 해소")
            self.blocked = ""

    # ── 상태 저장 (실행기 재시작용) ──
    def to_state(self) -> dict:
        return {"dir": self.dir, "tranches": [{**x.__dict__, "t_in": str(x.t_in)} for x in self.tranches],
                "partial_done": self.partial_done, "adds_done": self.adds_done, "next_tid": self.next_tid,
                "last_exit": None if self.last_exit is None else str(self.last_exit), "day": str(self.day),
                "day_pnl": self.day_pnl, "day_stop_n": self.day_stop_n, "halted": self.halted, "blocked": self.blocked,
                "equity": self.equity, "peak": self.peak, "dd_mode": self.dd_mode, "dd_stop_day": str(self.dd_stop_day),
                "mu0": self.mu0, "ref_unit": self.ref_unit, "shadow": self.shadow, "shadow_nets": self.shadow_nets}

    CARRY = ("equity", "peak", "dd_mode", "dd_stop_day", "ref_unit", "shadow", "shadow_nets", "blocked")

    def carry_state(self) -> dict:
        """날을 넘겨 이어갈 것만 (누적손익·낙폭 상태·사람 확인 중지)."""
        s = self.to_state()
        return {k: s[k] for k in self.CARRY}

    def load_state(self, s: dict):
        import datetime as dt
        self.dir = s.get("dir", 0)
        self.tranches = [Tranche(**{**x, "t_in": pd.Timestamp(x["t_in"])}) for x in s.get("tranches", [])]
        for k in ("partial_done", "adds_done", "next_tid", "day_pnl", "day_stop_n", "halted", "blocked",
                  "equity", "peak", "dd_mode", "mu0", "ref_unit", "shadow", "shadow_nets"):
            if k in s:
                setattr(self, k, s[k])
        self.last_exit = pd.Timestamp(s["last_exit"]) if s.get("last_exit") else None
        dd = s.get("day")
        if "day" in s:
            self.day = dt.date.fromisoformat(dd) if dd not in (None, "None") else None
        sd = s.get("dd_stop_day")
        self.dd_stop_day = dt.date.fromisoformat(sd) if sd not in (None, "None") else None


def with_values(r: Rules6, **kw) -> Rules6:
    return replace(r, **kw)
