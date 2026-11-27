"""6주차 재생 시연 — 지난 하루의 1분봉을 Engine6 + Watch 에 흘려 넣고, 장애 세 가지를 일부러 끼워 넣음.

  python demo6.py [--date 20260423]      → results/demo_<날짜>/log.jsonl · risk_events.csv

**실제 서버 기록이 아닙니다.** 모의계좌 실행(runner6.py --paper-order) 기록은 README 6-3 에 따로 붙입니다.
이 파일은 「규칙과 감시가 이 순서로 반응한다」를 키 없이 재현해 보이는 용도입니다.

틱 만들기: 1분봉마다 시가 → (롱 스프레드에 불리한 쪽 먼저) 고가/저가 → 종가 네 틱을 두 다리에 같이 흘림
끼워 넣는 장애 (시각은 그날 장중에 맞춰 자동으로 고름)
  ① 웹소켓 끊김 35초      → 20초에 「연결끊김」 신규 중지 → 다시 붙으면 해제 + 「다시채움」
  ② 하이닉스 체결 틱 150초 없음 → 120초에 「시세정지」 → 틱 오면 해제
  ③ 주문 거부 알림         → 「주문거부」 끈끈한 중지 → 그날 신규 없음 (사람 확인 전까지)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path

import pandas as pd

import pair_core as pc
import rules6 as r6
from backtest6 import CACHE, HERE, find_data, load
from monitor6 import Watch, WatchCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--reject-nth", type=int, default=0, help="n번째 진입 주문을 「거부」 로 바꿈 (0 = 안 함)")
    a = ap.parse_args()
    fp = Path(a.cache) if a.cache else sorted(CACHE.glob("plans_*.pkl"), key=lambda p: p.stat().st_size)[-1]   # 가장 긴 기간
    plans, F = pickle.loads(fp.read_bytes())
    df = load(find_data())
    if a.date is None:                                   # 6주차 D 에서 2차 진입 + 분할 청산이 있었던 마지막 날
        T = pd.read_csv(HERE / "results" / "trades_D.csv", dtype={"date": str})
        g = T.groupby("date")
        a.date = [d for d, x in g if (x["tag"] == "2차").any() and (x["exit"] == "분할청산1").any()][-1]
    d = a.date
    f = F[F.index.strftime("%Y%m%d") == d]
    o = df[df["date"] == d]
    pl = plans[pd.Timestamp(d)]
    out = HERE / "results" / f"demo_{d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").unlink(missing_ok=True)
    clock = {"t": f.index[0], "s": 0.0}

    def log(kind, at=None, **kw):
        rec = {"t": str(at if at is not None else clock["t"]), "kind": kind, **kw}
        with (out / "log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[{rec['t'][11:19]}] {kind:8s} " + " ".join(f"{k}={v}" for k, v in kw.items()))

    eng = r6.Engine6(r6.Rules6(), mult=10)
    eng.set_day(pd.Timestamp(d).date(), pl.params)
    urgent = []
    w = Watch(WatchCfg(), ("A", "B"), log=log, block=lambda why: eng.block(clock["t"], why),
              unblock=lambda why: eng.unblock(clock["t"], why), urgent_exit=urgent.append,
              clock=lambda: clock["s"], hhmm=lambda: clock["t"].strftime("%H:%M"))
    log("시작", 날짜=d, 진입a=pl.plan.entry, 수렴m=pl.plan.exit, 반감기분=round(pl.fit.half_life, 1), 최대보유분=pl.params.max_hold,
        내용="재생 시연 — 실제 서버 아님")
    w.on_state("connecting"); w.on_state("live")

    # 장애 시각: 첫 거래 뒤 첫 빈 시간대에
    T = pd.read_csv(HERE / "results" / "trades_D.csv", dtype={"date": str}, parse_dates=["t_in", "t_out"])
    Td = T[T["date"] == d]
    t_last = Td["t_out"].max()
    inj = {"disc": t_last + pd.Timedelta(minutes=5), "stale": t_last + pd.Timedelta(minutes=20)}
    n_open = 0
    stale_until = None
    pend = []
    n_ev = 0
    for i, (ts, r) in enumerate(f.iterrows()):
        bar = o.loc[ts] if ts in o.index else None
        clock["t"] = ts
        # 1) 다음 분 시가 체결
        for od in pend:
            pa, pb = float(bar["a_open"]), float(bar["b_open"])
            if od.kind == "open":
                qa = pc.hedge_qty(1.0, pa, pb, replace(eng.p, unit_b=od.units))
                no = f"D{i:04d}"
                w.track(no, "A", "buy" if od.dir > 0 else "sell", qa)
                n_open += 1
                if n_open == a.reject_nth:
                    w.on_order({"id": no, "status": "rejected", "errorCode": "(재생) 증거금 부족 가정"})
                    continue
                w.on_order({"id": no, "status": "filled", "filled": qa, "remaining": 0, "fillPrice": pa})
                eng.opened(ts, od.units, qa, od.units, pa, pb, od.z, od.dir, od.tag)
            else:
                x = [q for q in eng.tranches if q.tid in od.tids]
                if x:
                    net = sum(q.pnl(eng.dir, pa, pb, 10) - 2 * pc.round_trip_cost(q.qa, q.qb, pa, pb, pc.Costs(), 10) / 2 for q in x)
                    eng.closed(ts, tuple(q.tid for q in x), net, od.why, float(r["z"]))
        pend = []
        # 2) 틱 4개 (15초 간격)
        if bar is not None:
            seq_a = [bar["a_open"], bar["a_low"], bar["a_high"], bar["a_close"]]
            seq_b = [bar["b_open"], bar["b_high"], bar["b_low"], bar["b_close"]]
            for k in range(4):
                clock["t"], clock["s"] = ts + pd.Timedelta(seconds=15 * k), clock["s"] + 15
                if inj["disc"] <= clock["t"] < inj["disc"] + pd.Timedelta(seconds=35):
                    if w.state == "live":
                        w.on_state("reconnecting", "(재생) 서버가 연결을 닫음")
                    w.check()
                    continue
                if w.state != "live":
                    w.on_state("live")
                w.on_tick("A", float(seq_a[k]))
                if stale_until is None and clock["t"] >= inj["stale"]:
                    stale_until = clock["t"] + pd.Timedelta(seconds=150)
                if not (stale_until and clock["t"] < stale_until):
                    w.on_tick("B", float(seq_b[k]))
                w.check()
                if urgent:                              # 틱 손절 → 봉 끝을 안 기다리고 그 틱 가격으로
                    why = urgent.pop().replace("틱 ", "")
                    if eng.tranches:
                        x = list(eng.tranches)
                        pa_, pb_ = float(seq_a[k]), float(seq_b[k])
                        net = sum(q.pnl(eng.dir, pa_, pb_, 10) - pc.round_trip_cost(q.qa, q.qb, pa_, pb_, pc.Costs(), 10) for q in x)
                        eng.closed(clock["t"], tuple(q.tid for q in x), net, why, float("nan"))
                        w.set_guard(None)
                if w.need_refill:
                    w.need_refill = False
                    log("다시채움", 내용="재연결 — 실행기는 이 자리에서 REST 로 1분봉·잔고를 즉시 다시 받음")
        clock["t"] = ts
        # 3) 봉 판단
        blk = frozenset(([+1] if r["block_long"] else []) + ([-1] if r["block_short"] else []))
        for od in eng.on_bar(ts, float(r["z"]), float(r["corr"]), float(r["a"]), float(r["b"]), blk):
            od.z = float(r["z"])
            pend.append(od)
        if eng.tranches and math.isfinite(float(r["z"])):
            se = float(r["sd"])
            w.set_guard({"dir": eng.dir, "mu": float(r["spread"]) - float(r["z"]) * se, "se": se, "entry": eng.p.entry,
                         "stop_gap": 1.5, "loss_lim": 300_000 * eng.units, "mult": 10,
                         "tranches": [(q.qa, q.qb, q.pa, q.pb) for q in eng.tranches]})
        else:
            w.set_guard(None)
        for e in eng.events[n_ev:]:
            log("규칙작동", at=e["t"], **{k: v for k, v in e.items() if k != "t"})
        n_ev = len(eng.events)
    pd.DataFrame(eng.events).to_csv(out / "risk_events.csv", index=False, encoding="utf-8-sig")
    log("종료", 일손익원=round(eng.day_pnl), 남은이상=list(w.issues), 신규중지=eng.blocked or None)
    L = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    L.sort(key=lambda r: r["t"])                           # 봉 끝에 모아 적은 규칙 기록을 시각 순으로
    (out / "log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in L), encoding="utf-8")


if __name__ == "__main__":
    main()
