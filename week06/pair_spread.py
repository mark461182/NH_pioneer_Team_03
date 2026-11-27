"""삼성전자–SK하이닉스 주식선물 페어 — 장중 실행기 (4주차, Team-03)

  python pair_spread.py --app-dir <nh-futures-open-api 폴더>               # 신호만 기록 (주문 없음)
  python pair_spread.py --app-dir <...> --paper-order                     # NH 국내 모의계좌 주문
  python pair_spread.py --app-dir <...> --sim                             # 시뮬레이션: 실전 시세 호가로 종이 체결
  python pair_spread.py --app-dir <...> --paper-order --entry 2.5 --window 240   # 백테스트로 고른 숫자
  python pair_spread.py --app-dir <...> --paper-order --ou --passive auto           # OU 숫자 + 하이닉스 지정가 (README 3-7·3-8)
  python pair_spread.py --app-dir <...> --paper-order --ou --passive auto --trend-filter   # + 스프레드 추세 필터 (3-9)
  python pair_spread.py --app-dir <...> --paper-order --ou --passive auto --risk           # + 리스크 레이어 (3-10)

흐름 (20초마다)
  실전 키 1분봉 두 종목 → (지난 N일 + 오늘) → β·스프레드·z·상관 (pair_core.features, 백테스트와 같은 함수)
  → pair_core.Signal.decide → 진입/청산이면 pair_core.PairExecutor 로 **두 다리 함께** 주문
  → 체결 확인 · 레그 실패 처리 · 기록 → 5분마다 계좌 잔고와 대조

두 다리 규칙 (pair_core.PairExecutor)
  진입: 하이닉스 → 삼성 순서로 시장가, 6초 안 체결 확인 → 모자라면 취소 후 남은 수량 1회 재주문
        → 그래도 한 다리만 체결이면 체결분을 반대 주문으로 되돌리고 「레그실패」, 그날 신규 중지
  청산: 같은 방식이지만 **되돌리지 않음** — 남은 다리만 다음 루프마다 다시 냄 (경보 기록)

기록 results/<날짜>/ : log.jsonl(모든 일) · signals.csv(분마다 z·β·상관·결정) · trades.csv(거래·다리별 체결·슬리피지)
                     · state.json(재시작 복구)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

import pair_core as pc
from risk_layer import RiskConfig, RiskLayer

KST = dt.timezone(dt.timedelta(hours=9))
POLL = 20
RECONCILE_EVERY = 300
HEARTBEAT_EVERY = 300
ROOT = Path(__file__).resolve().parent / "results"
SECRET = ("acnt", "acct", "account", "ac_no", "ac_nm", "pwd", "token", "key", "user_id", "ip_addr", "mac")


def now_kst() -> dt.datetime:
    return dt.datetime.now(KST)


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not any(w in k.lower() for w in SECRET)}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"(app_?(?:key|secret)\W{0,6})[A-Za-z0-9+/=_-]{6,}", r"\1***", x, flags=re.I)
    return x


class Rec:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def log(self, kind, **kw):
        kw = clean(kw)
        rec = {"t": now_kst().isoformat(timespec="seconds"), "kind": kind, **kw}
        with (self.out / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[{rec['t'][11:19]}] {kind:8s} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    def append_csv(self, name: str, row: dict):
        path = self.out / name
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


class Runner:
    """시세·주문·시계는 밖에서 넣음 → test_pair.py 가 가짜로 바꿔 시험."""

    def __init__(self, a, R: Rec, A: dict, B: dict, bars_today, quote, broker, hist: pd.DataFrame):
        self.a, self.R, self.A, self.B = a, R, A, B
        self.bars_today, self.quote, self.b = bars_today, quote, broker
        self.p = replace(pc.Params(), entry=a.entry, exit=a.exit, stop=a.stop, window=a.window,
                         loss_limit=a.loss_limit, day_loss_limit=a.day_loss, unit_b=a.unit_b)
        self.sig = pc.Signal(self.p)
        self.ex = pc.PairExecutor(broker, log=R.log) if broker is not None else None
        self.hist = hist                      # 지난 거래일 1분봉 (a, b 종가) — β·z 계산용
        self.mult = A["mult"]
        self.exit_left: list[dict] = []      # 청산이 덜 된 다리 [{sym, side, qty}]
        self.entry_info: dict = {}
        self.trades: list[dict] = []
        self.last_bar = None
        self.halt_why = ""
        self.t_rec = self.t_beat = 0.0
        self.state_path = R.out / "state.json"
        self.plan = None                      # OU 모드: 오늘 DayPlan
        self.no_trade = ""                    # OU 모드: 오늘 쉬는 이유
        self.risk = None                      # 리스크 레이어 (README 3-10) — 날을 넘는 상태는 results/risk_state.json
        if getattr(a, "risk", False):
            self.risk = RiskLayer(RiskConfig(budget=a.risk_budget, k_max=a.k_max), R.out.parent / "risk_state.json")
            self.p = replace(self.p, loss_limit=a.risk_budget, day_loss_limit=2 * a.risk_budget)
            self.sig.p = self.p
        if getattr(a, "ou", False):
            self._plan_ou()
        self._load()

    def _plan_ou(self):
        """README 3-7: 오늘 숫자를 전날까지 데이터의 OU 추정으로 정함 (장중 고정)."""
        c = self.costs()
        base = replace(self.p, entry=2.0, exit=0.5)
        if len(self.hist) < 300:
            self.no_trade = "과거 1분봉 부족"
        else:
            self.plan = pc.plan_today(self.hist["a"], self.hist["b"], now_kst().date(), base, c, self.mult,
                                      spread=getattr(self.a, "spread", "beta"))
            if self.plan.ok:
                self.p = replace(self.plan.params, loss_limit=self.p.loss_limit, day_loss_limit=self.p.day_loss_limit,
                                 unit_b=self.p.unit_b)
                self.sig.p = self.p
                if self.risk is not None:
                    self.risk.note_sigma(str(now_kst().date()), self.plan.fit.sigma_eq)
                    self.risk.save()
            else:
                self.no_trade = self.plan.why
        self.R.log("OU계획", **(self.plan.row() if self.plan else {}), 쉬는이유=self.no_trade or None,
                   진입=self.p.entry, 청산=self.p.exit, 손절=self.p.stop, 창=self.p.window, 최대보유=self.p.max_hold)

    # ── 상태 ──
    def _save(self):
        s = {"pos": None if self.sig.pos is None else {**asdict(self.sig.pos), "t_in": str(self.sig.pos.t_in)},
             "exit_left": self.exit_left, "entry_info": self.entry_info, "day": str(self.sig.day),
             "day_pnl": self.sig.day_pnl, "halted": self.sig.halted, "blocked": self.sig.blocked, "halt_why": self.halt_why,
             "last_exit": str(self.sig.last_exit) if self.sig.last_exit is not None else None, "trades": len(self.trades)}
        self.state_path.write_text(json.dumps(s, ensure_ascii=False, default=str, indent=1), encoding="utf-8")

    def _load(self):
        if not self.state_path.exists():
            return
        s = json.loads(self.state_path.read_text(encoding="utf-8"))
        if s.get("pos"):
            d = dict(s["pos"]); d["t_in"] = pd.Timestamp(d["t_in"])
            self.sig.pos = pc.Position(**{k: d[k] for k in ("dir", "qa", "qb", "pa", "pb", "t_in", "z_in", "beta")})
        self.exit_left, self.entry_info = s.get("exit_left", []), s.get("entry_info", {})
        self.sig.day = dt.date.fromisoformat(s["day"]) if s.get("day") not in (None, "None") else None
        self.sig.day_pnl, self.sig.halted, self.halt_why = s.get("day_pnl", 0.0), s.get("halted", False), s.get("halt_why", "")
        self.sig.blocked = s.get("blocked", False)
        if s.get("last_exit"):
            self.sig.last_exit = pd.Timestamp(s["last_exit"])
        tp = self.R.out / "trades.csv"
        if tp.exists():
            self.trades = pd.read_csv(tp).to_dict("records")
        self.R.log("상태복구", 포지션=s.get("pos"), 남은청산=self.exit_left, 신규중지=self.sig.halted or self.sig.blocked,
                   이유=self.halt_why)

    def halt(self, why: str):
        if not self.sig.blocked:
            self.sig.blocked, self.halt_why = True, why
            self.R.log("신규중지", 이유=why)

    # ── 계좌 대조 ──
    def mine(self) -> dict[str, int]:
        out = {self.A["sym"]: 0, self.B["sym"]: 0}
        pos = self.sig.pos
        if self.exit_left:              # 청산 중: 덜 된 다리만 계좌에 남음 (매수로 청산할 다리 = 지금 매도 포지션)
            for l in self.exit_left:
                out[l["sym"]] += -l["qty"] if l["side"] == "buy" else l["qty"]
        elif pos is not None:
            out[self.A["sym"]] += pos.dir * pos.qa
            out[self.B["sym"]] += -pos.dir * pos.qb
        return out

    def reconcile(self, force: bool = False):
        if self.b is None or not hasattr(self.b, "net_positions"):
            return
        if not force and time.time() - self.t_rec < RECONCILE_EVERY:
            return
        self.t_rec = time.time()
        try:
            acct = self.b.net_positions()
        except Exception as e:
            self.R.log("잔고조회실패", 오류=str(e)[:200])
            return
        mine = self.mine()
        acct = {s: acct.get(s, 0) for s in mine}
        if acct != mine:
            self.R.log("잔고불일치", 계좌=acct, 실행기=mine)
            self.halt("계좌 잔고와 실행기 포지션이 다름 — HTS 확인 후 재시작")
        else:
            self.R.log("잔고대조", 계좌=acct)

    # ── 시세 → 특징 ──
    def features(self) -> pd.DataFrame | None:
        fa, fb = self.bars_today(self.A["sym"]), self.bars_today(self.B["sym"])
        if fa is None or fb is None or not len(fa) or not len(fb):
            return None
        today = pd.DataFrame({"a": fa["close"], "b": fb["close"]}).dropna()
        cur = pd.Timestamp(now_kst().replace(tzinfo=None)).floor("min")
        today = today[today.index < cur]                 # 아직 안 끝난 분은 뺌
        full = pd.concat([self.hist, today])
        full = full[~full.index.duplicated(keep="last")].sort_index()
        if self.plan is not None:                         # OU 모드: z = (s − 직전 W분 평균) / σ_eq
            return pc.features_ou(full["a"], full["b"], {pd.Timestamp(now_kst().date()): self.plan},
                                  trend_filter=getattr(self.a, "trend_filter", False),
                                  spread=getattr(self.a, "spread", "beta"), base=self.p)
        betas = pc.daily_betas(full["a"], full["b"], self.p.beta_days)
        return pc.features(full["a"], full["b"], self.p, betas)

    # ── 주문 ──
    def passive_mode(self) -> str | None:
        """--passive none/a/b/auto (--passive-b 는 b 와 같음)."""
        m = getattr(self.a, "passive", "none")
        if m == "none" and getattr(self.a, "passive_b", False):
            m = "b"
        return None if m == "none" else m

    def costs(self) -> pc.Costs:
        return pc.Costs(fee_rate=self.a.fee_rate, passive=self.passive_mode())

    def passive_split(self, legs: list[pc.Leg], qa_ref: dict, qb_ref: dict):
        """(지정가 다리, 지정가, 나머지 다리). 지정가 = 내 쪽 최우선 호가 (매수 = 매수1호가, 매도 = 매도1호가)."""
        lb, la_ = legs                              # legs_enter 순서: [하이닉스, 삼성]
        L = pc.passive_leg_of(la_.qty, qa_ref["last"], lb.qty, qb_ref["last"], self.costs())
        leg, other, q = (la_, lb, qa_ref) if L == "a" else (lb, la_, qb_ref)
        return leg, (q["bid"] if leg.side == "buy" else q["ask"]), other

    def legs_enter(self, dir_: int, qa: int, qb: int | None = None) -> list[pc.Leg]:
        a_side = "buy" if dir_ > 0 else "sell"
        return [pc.Leg(self.B["sym"], pc.opposite(a_side), qb or self.p.unit_b),   # 하이닉스 먼저
                pc.Leg(self.A["sym"], a_side, qa)]

    def size_with_risk(self, d: pc.Decision, beta: float, qa_ref: dict, qb_ref: dict) -> tuple[int, int]:
        """리스크 레이어: (삼성 계약, 하이닉스 계약). 0 이면 이번 신호는 안 함."""
        q1 = pc.hedge_qty(beta, qa_ref["last"], qb_ref["last"], replace(self.p, unit_b=1))
        cu = pc.round_trip_cost(q1, 1, qa_ref["last"], qb_ref["last"],
                                self.costs(), self.mult)
        margin = None
        if hasattr(self.b, "max_qty"):                     # ④ 증거금: 주문가능수량 API
            try:
                a_side = "buy" if d.dir > 0 else "sell"
                ma = self.b.max_qty(self.A["sym"], a_side, qa_ref["last"])
                mb = self.b.max_qty(self.B["sym"], pc.opposite(a_side), qb_ref["last"])
                margin = int(min(mb, ma // max(q1, 1)))
            except Exception as e:
                self.R.log("주문가능수량실패", 오류=str(e)[:150])
        k, why = self.risk.size(day=str(now_kst().date()), qa=q1, pa=qa_ref["last"], pb=qb_ref["last"], mult=self.mult,
                                sigma_eq=self.plan.fit.sigma_eq, entry=self.p.entry, stop=self.p.stop,
                                tail=self.plan.tail, cost_unit=cu, margin_units=margin)
        self.R.log("리스크레이어", **why)
        if k == 0:
            return 0, 0
        return pc.hedge_qty(beta, qa_ref["last"], qb_ref["last"], replace(self.p, unit_b=k)), k

    def enter(self, d: pc.Decision, row, t):
        qa_ref, qb_ref = self.quote(self.A["sym"]), self.quote(self.B["sym"])
        beta = float(row["beta"])
        if self.risk is not None and self.plan is not None:
            qa, qb = self.size_with_risk(d, beta, qa_ref, qb_ref)
            if qb == 0:
                self.R.append_csv("trades.csv", {"구분": "리스크레이어0단위", "t": str(t), "z": round(float(row["z"]), 2)})
                return
        else:
            qa, qb = pc.hedge_qty(beta, qa_ref["last"], qb_ref["last"], self.p), self.p.unit_b
        if qa <= 0:
            self.R.log("진입보류", 이유="계약 수 계산 불가", beta=beta)
            return
        legs = self.legs_enter(d.dir, qa, qb)
        imb = qa * qa_ref["last"] * beta - qb * qb_ref["last"]
        self.R.log("진입신호", 이유=d.reason, dir=d.dir, z=round(float(row["z"]), 2), beta=round(beta, 3),
                   상관=round(float(row["corr"]), 2), 삼성=qa, 하이닉스=qb,
                   금액불균형원=round(imb * self.mult), 호가={"A": qa_ref, "B": qb_ref})
        if self.ex is None:
            self.R.log("가상", 내용="신호만 기록 (--paper-order / --sim 아님)")
            return
        t0 = time.time()
        if self.passive_mode():         # README 3-8: 비싼 다리는 최우선 호가 지정가로 먼저, 체결되면 다른 다리 시장가
            pleg, lim, other = self.passive_split(legs, qa_ref, qb_ref)
            self.entry_info["pleg"] = pleg.sym
            res = self.ex.execute_passive_first(pleg, lim, other, f"진입 {d.reason}", wait=self.a.passive_wait, unwind=True)
            if res.missed:
                self.R.append_csv("trades.csv", {"구분": "신호놓침", "t": str(t), "z": round(float(row["z"]), 2),
                                                 "메모": res.note})
                return
        else:
            res = self.ex.execute(legs, f"진입 {d.reason}", unwind=True)
        lag = round(time.time() - t0, 1)
        by = {lf.leg.sym: lf for lf in res.fills}
        fa, fb = by.get(self.A["sym"], pc.LegFill(legs[1])), by.get(self.B["sym"], pc.LegFill(legs[0]))
        if not res.ok:
            self.halt(f"레그실패 — {res.note}")
            self.R.append_csv("trades.csv", {"구분": "레그실패", "t": str(t), "z": round(float(row["z"]), 2),
                                             "A체결": f"{fa.filled}/{fa.leg.qty}", "B체결": f"{fb.filled}/{fb.leg.qty}",
                                             "되돌림": [(u.leg.sym, u.leg.side, u.filled) for u in res.unwound],
                                             "메모": res.note})
            return
        pos = pc.Position(d.dir, fa.filled, fb.filled, fa.avg, fb.avg, t, float(row["z"]), beta)
        self.sig.opened(pos)
        ref_a = qa_ref["ask"] if fa.leg.side == "buy" else qa_ref["bid"]
        ref_b = qb_ref["ask"] if fb.leg.side == "buy" else qb_ref["bid"]
        sgn = lambda side: 1 if side == "buy" else -1
        self.entry_info = {"t_in": str(t), "z_in": round(float(row["z"]), 2), "beta": round(beta, 4),
                           "A_ref": ref_a, "B_ref": ref_b, "A_fill": fa.avg, "B_fill": fb.avg,
                           "A_slip원": round(sgn(fa.leg.side) * (fa.avg - ref_a) * fa.filled * self.mult),
                           "B_slip원": round(sgn(fb.leg.side) * (fb.avg - ref_b) * fb.filled * self.mult),
                           "진입소요초": lag, "주문번호": [fb.order_nos, fa.order_nos],
                           "지정가다리": self.entry_info.get("pleg")}
        self.R.log("진입", **self.entry_info, 삼성=pos.qa, 하이닉스=pos.qb, dir=pos.dir)

    def exit(self, why: str, t):
        pos = self.sig.pos
        if self.exit_left:
            legs = [pc.Leg(l["sym"], l["side"], l["qty"]) for l in self.exit_left]
        else:
            legs = [pc.Leg(self.B["sym"], "buy" if pos.dir > 0 else "sell", pos.qb),
                    pc.Leg(self.A["sym"], "sell" if pos.dir > 0 else "buy", pos.qa)]
        if self.ex is None:
            q = self.quote(self.A["sym"]), self.quote(self.B["sym"])
            self._book_close(why, t, q[0]["last"], q[1]["last"], virtual=True)
            return
        res = None
        if self.passive_mode() and why == "수렴" and not self.exit_left:   # 수렴 청산만 지정가 먼저 (위험 청산은 바로 시장가)
            pleg, lim, other = self.passive_split(legs, self.quote(self.A["sym"]), self.quote(self.B["sym"]))
            res = self.ex.execute_passive_first(pleg, lim, other, f"청산 {why}", wait=self.a.passive_wait, unwind=False)
            if res.missed:
                self.R.log("지정가청산못함", 내용="시장가로 바로 청산")
                res = None
        if res is None:
            res = self.ex.execute(legs, f"청산 {why}", unwind=False)
        left = [{"sym": lf.leg.sym, "side": lf.leg.side, "qty": lf.leg.qty - lf.filled} for lf in res.fills if not lf.done]
        # 이번에 체결된 가격을 다리별로 누적 (재시도에 걸쳐 평균)
        fills = self.entry_info.setdefault("exit_fills", {})
        for lf in res.fills:
            if lf.filled:
                q0, a0 = fills.get(lf.leg.sym, (0, 0.0))
                fills[lf.leg.sym] = (q0 + lf.filled, (a0 * q0 + lf.avg * lf.filled) / (q0 + lf.filled))
        self.exit_left = left
        if left:
            self.R.log("경보", 내용=f"청산 덜 됨 {left} — 다음 루프에 다시 냄. 계속되면 HTS 에서 직접 청산", 사유=why)
            self.entry_info["exit_why"] = self.entry_info.get("exit_why") or why
            return
        xa = fills.get(self.A["sym"], (0, float("nan")))[1]
        xb = fills.get(self.B["sym"], (0, float("nan")))[1]
        self._book_close(self.entry_info.get("exit_why") or why, t, xa, xb)

    def _book_close(self, why, t, xa, xb, virtual=False):
        pos = self.sig.pos
        gross = pos.pnl(xa, xb, self.mult)
        fee = 2 * self.a.fee_rate * self.mult * (pos.qa * (pos.pa + xa) / 2 + pos.qb * (pos.pb + xb) / 2)
        net = gross - fee
        row = {"구분": "가상" if virtual else "거래", "t_in": str(pos.t_in), "t_out": str(t), "dir": pos.dir,
               "삼성": pos.qa, "하이닉스": pos.qb, "beta": round(pos.beta, 4), "z_in": round(pos.z_in, 2),
               "A_in": pos.pa, "B_in": pos.pb, "A_out": xa, "B_out": xb, "청산사유": why,
               "손익_비용전원": round(gross), "수수료가정원": round(fee), "순손익원": round(net),
               **{k: v for k, v in self.entry_info.items() if k.endswith("slip원") or k == "진입소요초"}}
        self.R.append_csv("trades.csv", row)
        self.trades.append(row)
        self.R.log("청산", **row)
        self.sig.closed(t, net)
        if self.risk is not None and not virtual:
            self.risk.on_close(net, str(pd.Timestamp(t).date()))
            self.R.log("리스크상태", 누적손익원=round(self.risk.st.equity), 고점대비원=round(self.risk.st.equity - self.risk.st.peak),
                       상태=self.risk.st.mode)
        self.exit_left, self.entry_info = [], {}
        if self.sig.halted:
            self.R.log("신규중지", 이유=f"하루 손실 한도 {self.p.day_loss_limit:,.0f}원 (일손익 {self.sig.day_pnl:,.0f}원) — 오늘만")

    # ── 한 바퀴 ──
    def step(self):
        f = self.features()
        if f is None or not len(f):
            self.R.log("시세없음", 내용="1분봉 빈 값 — 다음 루프")
            return
        row, t = f.iloc[-1], f.index[-1]
        new_bar = t != self.last_bar
        if self.exit_left:                                   # 덜 된 청산이 먼저
            self.exit(self.entry_info.get("exit_why") or "청산 재시도", t)
        if new_bar:
            self.last_bar = t
            blk = frozenset(([+1] if bool(row.get("block_long", False)) else [])
                            + ([-1] if bool(row.get("block_short", False)) else []))
            d = self.sig.decide(t, float(row["z"]), float(row["corr"]), float(row["a"]), float(row["b"]), self.mult, blk)
            if d.reason.startswith("추세 반대"):
                self.R.log("추세필터", 내용=d.reason, z=round(float(row["z"]), 2),
                           아침추세=self.plan.day_trend if self.plan else None,
                           장중기울기σ=None if pd.isna(row.get("slope_sig")) else round(float(row["slope_sig"]), 2))
            self.R.append_csv("signals.csv", {"t": str(t), "A": row["a"], "B": row["b"], "beta": round(float(row["beta"]), 4),
                                              "z": round(float(row["z"]), 3), "corr": round(float(row["corr"]), 3),
                                              "결정": d.action, "이유": d.reason,
                                              "포지션": 0 if self.sig.pos is None else self.sig.pos.dir})
            if d.action == "enter" and self.no_trade:
                self.R.log("진입안함", 이유=f"OU 계획상 오늘 쉼 — {self.no_trade}", z=round(float(row["z"]), 2))
            elif d.action == "enter":
                self.enter(d, row, t)
            elif d.action == "exit" and not self.exit_left:
                self.exit(d.reason, t)
        self._save()
        self.reconcile()
        if time.time() - self.t_beat >= HEARTBEAT_EVERY:
            self.t_beat = time.time()
            self.R.log("생존", 봉=str(t), z=round(float(row["z"]), 2), 상관=round(float(row["corr"]), 2),
                       포지션=None if self.sig.pos is None else self.sig.pos.dir, 거래=len(self.trades),
                       일손익원=round(self.sig.day_pnl), 신규중지=self.sig.halted or self.sig.blocked)

    def run(self):
        self.reconcile(force=True)
        try:
            while True:
                hhmm = now_kst().strftime("%H:%M")
                if hhmm >= self.a.stop_at and self.sig.pos is None and not self.exit_left:
                    break
                if "09:00" <= hhmm < "15:35":
                    try:
                        self.step()
                    except Exception as e:
                        self.R.log("루프오류", 오류=f"{type(e).__name__}: {str(e)[:250]}")
                time.sleep(POLL)
        except KeyboardInterrupt:
            self.R.log("중단", 내용="Ctrl+C")
            if self.sig.pos is not None:
                for _ in range(3):
                    self.exit("수동 중단", pd.Timestamp(now_kst().replace(tzinfo=None)))
                    if self.sig.pos is None:
                        break
        self._save()
        self.R.log("종료", 거래=len(self.trades), 일손익원=round(self.sig.day_pnl), 남은포지션=self.mine())


def load_history(F, nh, A, B, days: int) -> pd.DataFrame:
    """오늘 앞 `days` 거래일의 1분봉 (이 월물 그대로)."""
    out, d, got = [], now_kst().date() - dt.timedelta(days=1), 0
    tries = 0
    while got < days and tries < days * 3:
        tries += 1
        if d.weekday() < 5:
            day = d.strftime("%Y%m%d")
            fa, fb = nh.day_bars(F, A["sym"], day), nh.day_bars(F, B["sym"], day)
            if len(fa) and len(fb):
                out.append(pd.DataFrame({"a": fa["close"], "b": fb["close"]}).dropna())
                got += 1
            time.sleep(0.3)
        d -= dt.timedelta(days=1)
    return pd.concat(out).sort_index() if out else pd.DataFrame(columns=["a", "b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--paper-order", action="store_true", help="NH 국내 모의계좌 주문")
    mode.add_argument("--sim", action="store_true", help="실전 시세 호가로 종이 체결 (모의계좌를 못 쓸 때)")
    ap.add_argument("--sym-a"); ap.add_argument("--sym-b")
    ap.add_argument("--entry", type=float, default=2.0); ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=3.5); ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--loss-limit", type=float, default=300_000); ap.add_argument("--day-loss", type=float, default=600_000)
    ap.add_argument("--unit-b", type=int, default=1, help="하이닉스 계약 수 (삼성은 β 로 맞춤)")
    ap.add_argument("--fee-rate", type=float, default=0.00003, help="수수료율 편도 가정 (모름)")
    ap.add_argument("--quote-env", choices=("real", "paper"), default="real",
                    help="시세를 받을 키 (기본 실전). 실전 국내 키가 없고 모의 키에 시세 권한이 있으면 paper")
    ap.add_argument("--hist-days", type=int, default=7)
    ap.add_argument("--ou", action="store_true", help="README 3-7: 진입·청산·손절·창·보유를 오늘 OU 추정으로 (--entry 등 무시)")
    ap.add_argument("--passive", choices=("none", "a", "b", "auto"), default="none",
                    help="README 3-8: 지정가로 먼저 걸 다리. auto = 계약수×1틱이 큰 쪽(11/08 데이터 기준 삼성)")
    ap.add_argument("--passive-b", action="store_true", help="(예전 옵션) --passive b 와 같음")
    ap.add_argument("--passive-wait", type=float, default=30, help="하이닉스 지정가 기다리는 초")
    ap.add_argument("--trend-filter", action="store_true", help="README 3-9: 스프레드 추세 반대 방향 진입 막기 (--ou 필요)")
    ap.add_argument("--spread", choices=pc.SPREAD_MODES, default="beta",
                    help="README 3-11: beta(β 하루 고정) / ratio(β=1) / kalman(β 매 분 갱신) — 백테스트에서 고른 것 (--ou 필요)")
    ap.add_argument("--risk", action="store_true", help="README 3-10: 리스크 레이어로 단위 수 결정 (--ou 필요)")
    ap.add_argument("--risk-budget", type=float, default=300_000, help="거래당 꼬리 손실 예산(원)")
    ap.add_argument("--k-max", type=int, default=1, help="최대 단위 수 (1단위 = 하이닉스 1 + 삼성 β 맞춤). 기본 1 = 줄이기만")
    ap.add_argument("--stop-at", default="15:30")
    a = ap.parse_args()

    import nh_pair as nh
    F = nh.load_app(a.app_dir)
    nh.set_quote_env(a.quote_env)
    R = Rec(ROOT / now_kst().strftime("%Y%m%d"))
    if a.sym_a and a.sym_b:
        A, B = {"sym": a.sym_a, "name": nh.NAME_A, "mult": 10}, {"sym": a.sym_b, "name": nh.NAME_B, "mult": 10}
    else:
        rows = nh.master_rows(a.app_dir)
        A, B = nh.pick_contract(rows, nh.NAME_A, now_kst().date()), nh.pick_contract(rows, nh.NAME_B, now_kst().date())
    if A["mult"] != B["mult"]:
        sys.exit(f"두 종목 승수가 다름 ({A['mult']} / {B['mult']}) — 코드 수정 필요")
    pc.MULT = A["mult"]

    def quote(sym):
        q = nh.quote(F, sym)
        return {"last": q["last"], "bid": q["bid"], "ask": q["ask"]}

    if a.paper_order:
        broker, env = nh.NHPaperBroker(F), "모의계좌 주문 (시세는 실전 키)"
    elif a.sim:
        broker = pc.PaperFillBroker(lambda s: (quote(s)["bid"], quote(s)["ask"]))
        env = "시뮬레이션 — 실전 시세 호가로 종이 체결 (계좌 주문 없음)"
    else:
        broker, env = None, "신호만 기록 (주문 없음)"
    if a.risk and not a.ou:
        sys.exit("--risk 는 --ou 와 같이 써야 함 (σ_eq·꼬리 추정이 OU 계획에서 나옴)")
    if a.trend_filter and not a.ou:
        sys.exit("--trend-filter 는 --ou 와 같이 써야 함 (σ_eq·보유시간이 OU 추정에서 나옴)")
    R.log("시작", 환경=env, A=A, B=B, OU모드=a.ou, 스프레드=a.spread, 지정가다리=a.passive if a.passive != "none" else ("b" if a.passive_b else "none"), 추세필터=a.trend_filter, 리스크레이어=a.risk, 진입=a.entry, 청산=a.exit, 손절z=a.stop, 창=a.window,
          거래손실한도=a.loss_limit, 하루손실한도=a.day_loss)
    hist = load_history(F, nh, A, B, a.hist_days)
    R.log("과거봉", 일수=len(set(hist.index.date)) if len(hist) else 0, 봉=len(hist))
    Runner(a, R, A, B, lambda s: nh.day_bars(F, s, now_kst().strftime("%Y%m%d")), quote, broker, hist).run()


if __name__ == "__main__":
    main()
