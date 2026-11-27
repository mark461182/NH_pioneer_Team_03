"""6주차 장중 실행기 — 4주차 pair_spread.Runner 위에 rules6.Engine6(분할·손절·한도) + monitor6(실시간 감시)를 얹음.

  python runner6.py --app-dir <nh-futures-open-api 폴더> --paper-order          # NH 국내 모의계좌 주문 + 웹소켓 감시
  python runner6.py --app-dir <...> --sim                                       # 실전 시세 호가로 종이 체결 (계좌 주문 없음)
  python runner6.py --app-dir <...> --paper-order --no-stream                   # 웹소켓 없이 (감시 ④ 꺼짐)
  python runner6.py --app-dir <...> --paper-order --ack                         # 끈끈한 중지(거부·미체결·레그실패) 확인 후 재시작

바꾸지 않은 것 (4주차 그대로): 두 다리 주문(PairExecutor) · 레그 실패 되돌림 · 5분 잔고 대조 · 기록 형식
11/22 수정 (README 4-6 · 6절)
  - 규칙은 기본 설정 D (11/20). --rules e 면 rules6.improved() (설정 E): 고정 z 손절 · 비율 한도 · 갭/만기일 쉬기,
    낙폭 중단은 사람이 풀 때까지(--ack). 아래 수정들은 D·E 어느 쪽에도 적용됨
  - 진입 전 `주문가능수량` API 로 두 다리가 다 들어가는 단위만 (4주차에 있던 확인 — 6주차로 옮기며 빠졌던 것)
  - 누적손익·낙폭 상태·「사람 확인」 중지는 results/risk_state6.json 에 저장 → 날이 바뀌어도 이어짐
  - 실행기가 스스로 취소한 주문은 미체결 감시에서 뺌 (헛경보 방지)
  - 청산이 한 다리만 체결: 20초(또는 남은 다리 평가손실이 금액 손절의 절반) → 쉬지 않고 재시도 · 60초 → 신규 전면 중지(사람 확인)
  - 시세 웹소켓 끊김: 20초 신규 중지 → 60초 REST 간격 20초→5초 → 180초 2단위면 1단위로 축소
  - 틱 손절이 잡힌 시각 → 청산 주문까지의 지연(초)을 로그에 남김

6주차에 바뀐 것
  - 신호·청산 판단을 Signal → Engine6 (백테스트 backtest6.py 와 같은 객체)
  - 숫자: 보정 OU(호가 튐 분리) · 가격비 스프레드 · 추세 필터 · 시장가 (4주차 6-4 에서 비용 전 손익이 플러스였던 설정)
  - 루프 1초: 웹소켓 이상 점검·틱 손절은 1초마다, 1분봉 판단은 20초마다
  - 분할 진입·청산은 트랜치마다 따로 두 다리 주문

기록 results/<날짜>/ : log.jsonl · signals.csv · trades.csv(트랜치마다 한 줄) · risk_events.csv(규칙 작동) · state6.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

import pair_core as pc
import pair_spread as ps
import rules6 as r6
from backtest6 import fit_noise_days
from monitor6 import NHStream, Watch, WatchCfg

TICK = 1.0          # 감시 루프 (초)


class TrackingBroker:
    """주문번호가 나오는 순간 Watch 에 알림 (미체결·거부 감시용). 나머지는 원래 브로커 그대로."""

    def __init__(self, broker, watch: Watch):
        self._b, self._w = broker, watch

    def send(self, sym, side, qty, price=None):
        no = self._b.send(sym, side, qty) if price is None else self._b.send(sym, side, qty, price=price)
        self._w.track(no, sym, side, qty)
        return no

    def filled(self, no):
        q, px = self._b.filled(no)
        self._w.settle(no, q)
        return q, px

    def cancel(self, no, sym, side):
        try:
            return self._b.cancel(no, sym, side)
        finally:
            self._w.cancelled(no)          # 스스로 취소한 주문 — 남은 수량은 새 주문번호로 다시 감시

    def __getattr__(self, k):
        return getattr(self._b, k)


class Runner6(ps.Runner):
    def __init__(self, a, R, A, B, bars_today, quote, broker, hist, watch: Watch | None = None):
        self.watch = watch
        if getattr(a, "rules", "d") == "d":                # 11/20 설정 D (원 단위 한도·이동 z 손절·낙폭 뒤 자동 재개)
            self.rules = r6.Rules6(units_max=a.units_max)
        else:                                               # 11/22 설정 E
            self.rules = r6.improved(units_max=a.units_max, loss_stop_pct=a.loss_stop_pct, day_loss_pct=a.day_loss_pct,
                                     dd_resume="manual")
        if watch is not None and broker is not None:
            broker = TrackingBroker(broker, watch)
        self.eng = r6.Engine6(self.rules, mult=A["mult"])
        self.pending_close = None          # {"tids", "why", "left":[{sym,side,qty}], "fills":{sym:(q,avg)}}
        self.urgent = ""
        self.n_events = 0
        self.sticky = ""                   # 사람이 확인해야 풀리는 중지 사유 (날을 넘겨 유지)
        self.poll = ps.POLL
        self._fastpoll = self._shrunk = False
        self.carry_path = R.out.parent / "risk_state6.json"
        super().__init__(a, R, A, B, bars_today, quote, broker, hist)
        self.eng.p = self.p
        if watch is not None:
            watch.log, watch.block, watch.unblock = R.log, self._wblock, self._wunblock
            watch.urgent_exit = self._want_exit

    # ── Watch 콜백 ──
    def _now(self):
        return pd.Timestamp(ps.now_kst().replace(tzinfo=None))

    def _wblock(self, why):
        from monitor6 import STICKY
        if why.split(":")[0] in STICKY:
            self.sticky = why
        self.eng.block(self._now(), why)

    def _wunblock(self, why):
        self.eng.unblock(self._now(), why)

    def _want_exit(self, why):
        self.urgent = why

    def halt(self, why: str):                  # 4주차 레그실패·잔고불일치 → 엔진 신규 중지 (사람 확인)
        self.sticky = self.sticky or why
        self.eng.block(self._now(), why)
        if self.watch is not None:
            self.watch.issues.setdefault("사람확인", why)      # 끈끈한 항목으로 남김 → --ack 전엔 안 풀림
        self.R.log("신규중지", 이유=why)

    # ── 상태 ──
    def _save(self):
        s = {"engine": self.eng.to_state(), "pending_close": self.pending_close, "trades": len(self.trades),
             "watch_issues": dict(self.watch.issues) if self.watch else {}}
        (self.R.out / "state6.json").write_text(json.dumps(s, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        carry = {"engine": self.eng.carry_state(), "sticky": self.sticky, "saved": str(self._now())}
        carry["engine"]["blocked"] = self.sticky          # 끊김·시세정지 같은 자동 해제 항목은 날을 안 넘김
        self.carry_path.write_text(json.dumps(carry, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        self._flush_events()

    def _load(self):
        ack = getattr(self.a, "ack", False)
        if self.carry_path.exists():                      # ① 날을 넘겨 이어갈 것: 누적손익·낙폭 상태·사람 확인 중지
            c = json.loads(self.carry_path.read_text(encoding="utf-8"))
            self.eng.load_state(c.get("engine", {}))
            self.sticky = c.get("sticky", "")
        p = self.R.out / "state6.json"
        if p.exists():                                    # ② 오늘 다시 켠 경우: 포지션·그날 손익까지
            s = json.loads(p.read_text(encoding="utf-8"))
            self.eng.load_state(s.get("engine", {}))
            self.pending_close = s.get("pending_close")
            if self.pending_close:
                self.pending_close.setdefault("t0", time.time())
                self.pending_close.setdefault("ref", {})
            tp = self.R.out / "trades.csv"
            if tp.exists():
                self.trades = pd.read_csv(tp).to_dict("records")
        if ack:
            self.R.log("사람확인", 내용="--ack — 사람 확인 중지와 낙폭 중단을 풂", 풀린중지=self.sticky or None, 낙폭상태=self.eng.dd_mode)
            self.sticky, self.eng.blocked = "", ""
            self.eng.resume(self._now(), "--ack")
        else:
            self.eng.blocked = self.sticky
            if self.sticky and self.watch is not None:
                self.watch.issues["사람확인"] = self.sticky
        self.R.log("규칙", **{k: str(v) for k, v in self.rules.__dict__.items()})
        self.R.log("상태복구", 보유단위=self.eng.units, 방향=self.eng.dir, 남은청산=self.pending_close,
                   누적손익원=round(self.eng.equity), 고점대비원=round(self.eng.equity - self.eng.peak), 낙폭상태=self.eng.dd_mode,
                   신규중지=self.eng.blocked or self.eng.halted or None)

    def _flush_events(self):
        ev = self.eng.events[self.n_events:]
        for e in ev:
            self.R.append_csv("risk_events.csv", {**e, "t": str(e["t"])})
            self.R.log("규칙작동", **{k: v for k, v in e.items() if k != "t"}, 봉=str(e["t"]))
        self.n_events = len(self.eng.events)

    # ── 계좌 대조 (트랜치 합) ──
    def mine(self) -> dict[str, int]:
        out = {self.A["sym"]: 0, self.B["sym"]: 0}
        closing = set(self.pending_close["tids"]) if self.pending_close else set()
        for x in self.eng.tranches:
            if x.tid in closing:
                continue
            out[self.A["sym"]] += self.eng.dir * x.qa
            out[self.B["sym"]] += -self.eng.dir * x.qb
        if self.pending_close:
            for l in self.pending_close["left"]:
                out[l["sym"]] += -l["qty"] if l["side"] == "buy" else l["qty"]
        return out

    # ── 주문 ──
    def _open(self, o: r6.Order, row, t):
        qa_ref, qb_ref = self.quote(self.A["sym"]), self.quote(self.B["sym"])
        beta = float(row["beta"])
        if hasattr(self.b, "max_qty"):                     # 증거금: 두 다리가 다 들어가는 단위만 (주문가능수량 API)
            try:
                a_side = "buy" if o.dir > 0 else "sell"
                q1 = pc.hedge_qty(beta, qa_ref["last"], qb_ref["last"], replace(self.p, unit_b=1))
                ma = self.b.max_qty(self.A["sym"], a_side, qa_ref["last"])
                mb = self.b.max_qty(self.B["sym"], pc.opposite(a_side), qb_ref["last"])
                can = int(min(mb, ma // max(q1, 1)))
                self.R.log("증거금확인", 가능단위=can, 요청단위=o.units, 삼성가능=ma, 하이닉스가능=mb)
                if can < o.units:
                    o.units = can
            except Exception as e:
                self.R.log("주문가능수량실패", 오류=str(e)[:150], 내용="확인 없이 진행 — 모자라면 주문 거부로 드러남")
        if o.units <= 0:
            self.R.log("진입보류", 이유="증거금 부족 — 주문가능수량 0단위", 트랜치=o.tag)
            self.R.append_csv("trades.csv", {"구분": "증거금부족", "t": str(t), "트랜치": o.tag})
            return
        qa = pc.hedge_qty(beta, qa_ref["last"], qb_ref["last"], replace(self.p, unit_b=o.units))
        legs = self.legs_enter(o.dir, qa, o.units)
        self.R.log("진입신호", 트랜치=o.tag, 이유=o.why, dir=o.dir, 삼성=qa, 하이닉스=o.units, z=round(float(row["z"]), 2),
                   호가={"A": qa_ref, "B": qb_ref})
        if self.ex is None:
            self.eng.opened(t, o.units, qa, o.units, qa_ref["last"], qb_ref["last"], float(row["z"]), o.dir, o.tag)
            return
        t0 = time.time()
        res = self.ex.execute(legs, f"진입 {o.tag} {o.why}", unwind=True)
        by = {lf.leg.sym: lf for lf in res.fills}
        fa, fb = by[self.A["sym"]], by[self.B["sym"]]
        if not res.ok:
            self.halt(f"레그실패 — {res.note}")
            self.R.append_csv("trades.csv", {"구분": "레그실패", "t": str(t), "트랜치": o.tag,
                                             "A체결": f"{fa.filled}/{fa.leg.qty}", "B체결": f"{fb.filled}/{fb.leg.qty}", "메모": res.note})
            return
        self.eng.opened(t, o.units, fa.filled, fb.filled, fa.avg, fb.avg, float(row["z"]), o.dir, o.tag)
        self.R.log("진입", 트랜치=o.tag, 삼성=fa.filled, 하이닉스=fb.filled, A_fill=fa.avg, B_fill=fb.avg,
                   진입소요초=round(time.time() - t0, 1), 보유단위=self.eng.units)

    def _close(self, tids: tuple, why: str, t, z=float("nan")):
        gone = [x for x in self.eng.tranches if x.tid in tids]
        if not gone:
            return
        d = self.eng.dir
        if self.pending_close:
            legs = [pc.Leg(l["sym"], l["side"], l["qty"]) for l in self.pending_close["left"]]
            fills = self.pending_close["fills"]
        else:
            qa, qb = sum(x.qa for x in gone), sum(x.qb for x in gone)
            legs = [pc.Leg(self.B["sym"], "buy" if d > 0 else "sell", qb), pc.Leg(self.A["sym"], "sell" if d > 0 else "buy", qa)]
            fills = {}
        if self.ex is None:
            qa_, qb_ = self.quote(self.A["sym"]), self.quote(self.B["sym"])
            self._book(gone, why, t, qa_["last"], qb_["last"], z, virtual=True)
            return
        res = self.ex.execute(legs, f"청산 {why}", unwind=False)
        for lf in res.fills:
            if lf.filled:
                q0, a0 = fills.get(lf.leg.sym, (0, 0.0))
                fills[lf.leg.sym] = (q0 + lf.filled, (a0 * q0 + lf.avg * lf.filled) / (q0 + lf.filled))
        left = [{"sym": lf.leg.sym, "side": lf.leg.side, "qty": lf.leg.qty - lf.filled} for lf in res.fills if not lf.done]
        if left:
            old = self.pending_close or {}
            ref = old.get("ref") or {}
            for l in left:                                 # 한 다리 노출의 기준가 (처음 덜 체결된 순간)
                if l["sym"] not in ref:
                    try:
                        ref[l["sym"]] = self.quote(l["sym"])["last"]
                    except Exception:
                        pass
            self.pending_close = {"tids": list(tids), "why": why, "left": left, "fills": fills,
                                  "t0": old.get("t0", time.time()), "ref": ref,
                                  "fast": old.get("fast", False), "halted": old.get("halted", False)}
            if not old:
                self.R.log("경보", 내용=f"청산 덜 됨 {left} — 남은 다리만 다시 냄", 사유=why)
            return
        self.pending_close = None
        self._book(gone, why, t, fills[self.A["sym"]][1], fills[self.B["sym"]][1], z)

    def _book(self, gone, why, t, xa, xb, z, virtual=False):
        d, net_sum = self.eng.dir, 0.0
        for x in gone:
            gross = x.pnl(d, xa, xb, self.mult)
            fee = 2 * self.a.fee_rate * self.mult * (x.qa * (x.pa + xa) / 2 + x.qb * (x.pb + xb) / 2)
            net = gross - fee
            net_sum += net
            row = {"구분": "가상" if virtual else "거래", "트랜치": x.tag, "t_in": str(x.t_in), "t_out": str(t), "dir": d,
                   "삼성": x.qa, "하이닉스": x.qb, "z_in": round(x.z_in, 2), "A_in": x.pa, "B_in": x.pb, "A_out": xa, "B_out": xb,
                   "청산사유": why, "손익_비용전원": round(gross), "수수료가정원": round(fee), "순손익원": round(net)}
            self.R.append_csv("trades.csv", row)
            self.trades.append(row)
            self.R.log("청산", **row)
        self.eng.closed(t, tuple(x.tid for x in gone), net_sum, why, z)
        self._flush_events()

    # ── 한 바퀴 (20초마다) ──
    def step(self):
        f = self.features()
        if f is None or not len(f):
            self.R.log("시세없음", 내용="1분봉 빈 값 — 다음 루프")
            return
        row, t = f.iloc[-1], f.index[-1]
        if t.date() != self.eng.day:                        # 새 날: 갭·만기일이면 그날 신규 쉼
            ok = self.plan is not None and self.plan.ok
            self.eng.set_day(t.date(), self.p if ok else None, r6.day_skip(self.rules, t.date(), self._gap(f, t)) if ok else "")
        if self.pending_close:
            self._close(tuple(self.pending_close["tids"]), self.pending_close["why"], t)
        if t != self.last_bar:
            self.last_bar = t
            blk = frozenset(([+1] if bool(row.get("block_long", False)) else [])
                            + ([-1] if bool(row.get("block_short", False)) else []))
            z = float(row["z"]) if not self.no_trade else float("nan")
            orders = self.eng.on_bar(t, z, float(row["corr"]), float(row["a"]), float(row["b"]), blk,
                                     float(row["spread"]), float(row["sd"]))
            self.R.append_csv("signals.csv", {"t": str(t), "A": row["a"], "B": row["b"], "z": round(z, 3),
                                              "corr": round(float(row["corr"]), 3), "보유단위": self.eng.units,
                                              "주문": ";".join(f"{o.kind}:{o.tag or o.why}" for o in orders)})
            for o in orders:
                if o.kind == "open":
                    self._open(o, row, t)
                elif not self.pending_close:
                    self._close(o.tids, o.why, t, z)
            self._guard(row)
        self._save()
        self.reconcile()
        if time.time() - self.t_beat >= ps.HEARTBEAT_EVERY:
            self.t_beat = time.time()
            self.R.log("생존", 봉=str(t), z=round(float(row["z"]), 2), 보유단위=self.eng.units, 거래=len(self.trades),
                       일손익원=round(self.eng.day_pnl), 신규중지=self.eng.blocked or self.eng.halted or None,
                       실시간=None if self.watch is None else self.watch.state)

    @staticmethod
    def _gap(f: pd.DataFrame, t) -> float:
        """밤사이 스프레드 갭 (σ_eq 배): 오늘 첫 봉 − 전일 마지막 봉, 가격비 스프레드 ln(A) − ln(B)."""
        import numpy as np
        today, prev = f[f.index >= t.normalize()], f[f.index < t.normalize()]
        if not len(today) or not len(prev):
            return float("nan")
        sp = lambda r: float(np.log(r["a"]) - np.log(r["b"]))
        se = float(today["sd"].iloc[0])
        return (sp(today.iloc[0]) - sp(prev.iloc[-1])) / se if se > 0 else float("nan")

    def _guard(self, row):
        """틱 손절 기준을 Watch 에 넘김: μ = s − z·σ_eq (그 분의 이동평균)."""
        if self.watch is None:
            return
        if not self.eng.tranches or not math.isfinite(float(row["z"])):
            self.watch.set_guard(None)
            return
        se = float(row["sd"])
        self.watch.set_guard({"dir": self.eng.dir, "mu": float(row["spread"]) - float(row["z"]) * se, "se": se,
                              "entry": self.p.entry, "stop_gap": self.rules.stop_gap, "price_stop": self.rules.price_stop,
                              "mu0": self.eng.mu0 if self.rules.anchor_stop else None,
                              "loss_lim": self.eng.loss_limit() or 0, "mult": self.mult,
                              "tranches": [(x.qa, x.qb, x.pa, x.pb) for x in self.eng.tranches]})

    def _urgent(self):
        why, self.urgent = self.urgent, ""
        if self.eng.tranches and not self.pending_close:
            lag = round(time.time() - self.watch.fired_at, 2) if (self.watch is not None and self.watch.fired_at) else None
            self.R.log("틱손절실행", 사유=why, 보유단위=self.eng.units, 감지후_주문까지초=lag)
            t0 = time.time()
            self._close(tuple(x.tid for x in self.eng.tranches), why.replace("틱 ", ""), self._now())
            self.R.log("틱손절완료", 청산소요초=round(time.time() - t0, 1), 남은청산=bool(self.pending_close))
            if self.watch is not None:
                self.watch.set_guard(None)

    # ── 청산이 한 다리만 체결됐을 때 (한 종목 방향 베팅 상태) ──
    def _naked(self) -> int:
        """단계: 1 = 20초 안(다음 루프에 재시도) · 2 = 쉬지 않고 재시도 · 3 = + 신규 전면 중지(사람 확인)."""
        p = self.pending_close
        age = time.time() - p.get("t0", time.time())
        loss = 0.0
        for l in p["left"]:
            ref = p.get("ref", {}).get(l["sym"])
            px = self.watch.price.get(l["sym"]) if self.watch is not None else None
            if ref and px:                                  # 매수로 닫을 다리 = 지금 매도 포지션 → 오르면 손실
                loss += (px - ref if l["side"] == "buy" else ref - px) * l["qty"] * self.mult
        gone = [x for x in self.eng.tranches if x.tid in p["tids"]]       # 닫는 중인 트랜치 (단위 수만큼 기준을 키움)
        if self.rules.loss_stop_pct:
            full = self.rules.loss_stop_pct * sum(x.notional(self.mult) for x in gone)
        else:
            full = (self.rules.loss_stop or 300_000) * max(1, sum(x.units for x in gone))
        lim = 0.5 * full                                   # 금액 손절의 절반 — 1단위 15만 · 2단위 30만 (D)
        stage = 3 if age >= 60 else (2 if (age >= 20 or loss >= lim) else 1)
        if stage >= 2 and not p.get("fast"):
            p["fast"] = True
            self.R.log("한다리노출", 단계=2, 내용="쉬지 않고 재시도", 경과초=round(age), 남은다리=p["left"], 평가손실원=round(loss), 기준원=round(lim))
        if stage == 3 and not p.get("halted"):
            p["halted"] = True
            self.R.log("한다리노출", 단계=3, 내용="60초 — 신규 전면 중지, HTS 확인 필요", 남은다리=p["left"], 평가손실원=round(loss))
            self.halt(f"청산이 한 다리만 체결된 채 60초 — 남은 {p['left']}")
        return stage

    # ── 시세 웹소켓이 끊긴 채일 때 ──
    def _stream_down(self):
        down = self.watch.down_for() if self.watch is not None else 0.0
        if down < 60:
            if self._fastpoll:
                self.R.log("감시회복", 내용="REST 간격 20초로 되돌림")
            self.poll, self._fastpoll, self._shrunk = ps.POLL, False, False
            return
        if not self._fastpoll:
            self._fastpoll, self.poll = True, 5
            self.R.log("감시끊김", 단계=2, 내용="60초 — REST 간격 20초 → 5초", 끊긴초=round(down))
        if down >= 180 and not self._shrunk and self.eng.units >= 2 and not self.pending_close:
            self._shrunk = True
            last = max(self.eng.tranches, key=lambda x: x.tid)
            self.R.log("감시끊김", 단계=3, 내용="180초 — 2단위 → 1단위로 축소", 끊긴초=round(down), 닫는트랜치=last.tag)
            self._close((last.tid,), "감시끊김축소", self._now())

    def run(self):
        self.reconcile(force=True)
        t_step = 0.0
        try:
            while True:
                hhmm = ps.now_kst().strftime("%H:%M")
                if hhmm >= self.a.stop_at and not self.eng.tranches and not self.pending_close:
                    break
                if "09:00" <= hhmm < "15:35":
                    try:
                        if self.watch is not None:
                            self.watch.check()
                            if self.watch.need_refill:            # 다시 붙음 → 끊긴 동안은 REST 로 바로 다시 채움
                                self.watch.need_refill = False
                                self.R.log("다시채움", 내용="웹소켓 재연결 — 1분봉·잔고를 REST 로 즉시 다시 받음")
                                t_step = 0.0
                                self.reconcile(force=True)
                            self._stream_down()
                        if self.urgent:
                            self._urgent()
                        if self.pending_close and self._naked() >= 2:      # 한 다리 노출 — 20초 루프를 안 기다림
                            self._close(tuple(self.pending_close["tids"]), self.pending_close["why"], self._now())
                            self._save()
                        if time.time() - t_step >= self.poll:
                            t_step = time.time()
                            self.step()
                    except Exception as e:
                        self.R.log("루프오류", 오류=f"{type(e).__name__}: {str(e)[:250]}")
                time.sleep(TICK)
        except KeyboardInterrupt:
            self.R.log("중단", 내용="Ctrl+C")
            for _ in range(3):
                if not self.eng.tranches:
                    break
                self._close(tuple(x.tid for x in self.eng.tranches), "수동 중단", self._now())
        self._save()
        self.R.log("종료", 거래=len(self.trades), 일손익원=round(self.eng.day_pnl), 남은포지션=self.mine())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--paper-order", action="store_true")
    m.add_argument("--sim", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="웹소켓 감시 끄기")
    ap.add_argument("--ack", action="store_true",
                    help="거부·미체결·레그실패·한 다리 노출·낙폭 중단으로 멈춘 것을 HTS 에서 확인했음 → 풀고 시작")
    ap.add_argument("--rules", choices=("d", "e"), default="d",
                    help="d = 11/20 규칙(기본 — 백테스트에서 꼬리 손실이 더 작았음, README 4-6) · e = 11/22 개선 규칙")
    ap.add_argument("--units-max", type=int, default=2)
    ap.add_argument("--loss-stop-pct", type=float, default=0.013, help="금액 손절 (진입 명목 대비)")
    ap.add_argument("--day-loss-pct", type=float, default=0.026, help="하루 손실 한도 (1단위 명목 대비)")
    ap.add_argument("--fee-rate", type=float, default=0.00003)
    ap.add_argument("--quote-env", choices=("real", "paper"), default="real")
    ap.add_argument("--hist-days", type=int, default=7)
    ap.add_argument("--stop-at", default="15:30")
    a = ap.parse_args()
    # 4주차 Runner 가 읽는 칸 (6주차 설정으로 고정)
    a.entry, a.exit, a.stop, a.window, a.unit_b = 2.0, 0.5, 3.5, 120, 1
    a.loss_limit, a.day_loss = 300_000, 600_000          # 4주차 Signal 용 칸 — 6주차 엔진은 안 씀
    a.ou, a.spread, a.trend_filter, a.passive, a.passive_b, a.risk = True, "ratio", True, "none", False, False
    a.sym_a = a.sym_b = None
    pc.fit_ou_days = fit_noise_days                     # 4주차 6-4 보정 OU

    import nh_pair as nh
    F = nh.load_app(a.app_dir)
    nh.set_quote_env(a.quote_env)
    R = ps.Rec(ps.ROOT / ps.now_kst().strftime("%Y%m%d"))
    rows = nh.master_rows(a.app_dir)
    A, B = nh.pick_contract(rows, nh.NAME_A, ps.now_kst().date()), nh.pick_contract(rows, nh.NAME_B, ps.now_kst().date())
    pc.MULT = A["mult"]

    def quote(sym):
        q = nh.quote(F, sym)
        return {"last": q["last"], "bid": q["bid"], "ask": q["ask"]}

    if a.paper_order:
        broker, env = nh.NHPaperBroker(F), "모의계좌 주문 (시세는 실전 키)"
    elif a.sim:
        broker, env = pc.PaperFillBroker(lambda s: (quote(s)["bid"], quote(s)["ask"])), "시뮬레이션 (종이 체결)"
    else:
        broker, env = None, "신호만 기록"
    watch = None
    if not a.no_stream:
        watch = Watch(WatchCfg(), (A["sym"], B["sym"]), log=R.log)
    R.log("시작", 버전="6주차 (11/22 수정)", 환경=env, A=A, B=B, 실시간감시=not a.no_stream, 규칙세트=a.rules)
    hist = ps.load_history(F, nh, A, B, a.hist_days)
    run = Runner6(a, R, A, B, lambda s: nh.day_bars(F, s, ps.now_kst().strftime("%Y%m%d")), quote, broker, hist, watch)
    if watch is not None:
        NHStream(watch, (A["sym"], B["sym"]), quote_env=a.quote_env).start()
    run.run()


if __name__ == "__main__":
    main()
