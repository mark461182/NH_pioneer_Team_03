"""6주차 — 위험관리 넣기 전·후 비교 백테스트 (같은 기간 · 같은 신호 · 같은 비용 · 같은 최대 2단위)

  python backtest6.py [--data data/pair_1m_full.csv] [--days 0]      # --days N: 마지막 N영업일만

전략 (바꾸지 않음): 4주차 6-4 「보정 OU · 시장가 · 추세필터」, 가격비 스프레드 s = ln(삼성) − ln(하이닉스)
  날마다 전날까지 2일 1분 스프레드로 OU(호가 튐 분리) 추정 → 비용을 넣은 분당 기대이익 최대 진입선 a·청산선 m
  창 = 반감기 × 5, 최대 보유 = 반감기 × 3, 추세 반대 진입 금지
체결 가정 (4주차와 같음): 주문 낸 분의 다음 분 시가, 다리마다 1틱 불리 + 수수료 편도 0.003%(가정)

비교하는 일곱 가지 — 바꾸는 것은 rules6.Rules6 하나뿐
  A  위험관리 없음           2단위 한 번에, 수렴·장마감으로만 청산
  A1 A + 가격 손절           |z| ≥ a + 1.5
  A2 A + 시간 손절           보유 ≥ 반감기 × 3
  A3 A + 금액 손절           평가손실 ≥ 단위당 30만 원
  B  4주차 안전장치          A1+A2+A3 + 상관 붕괴 + 하루 손실 60만
  C  B + 분할 진입·청산      1차 1단위 → a+0.5 에서 2차 1단위 / 중간선에서 절반 청산 → 이익 보호선
  D  6주차 전체              C + 하루 손절 2회 + 누적 낙폭 한도

숫자는 데이터를 보기 전에 정한 값 그대로 (README 3절). 아래 민감도 표는 「이 값 근처에서 결과가 뒤집히나」 확인용이고 고르는 데 쓰지 않음.

출력 results/compare.csv · results/compare_by_year.csv · results/trades_<arm>.csv · results/risk_events_D.csv
     results/sensitivity.csv · results/figs/*.png · results/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import ou_noise as on
import pair_core as pc
import rules6 as r6

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = HERE / "cache"


# ── 4주차 6-4 보정 OU (호가 튐 분리) — backtest_ou_noise.py 와 같은 함수 ──
def fit_noise_days(days, min_n: int = 200, min_len: int = 30):
    xs = [np.asarray(x, float) for x in days if len(np.asarray(x)) >= min_len]
    if sum(len(x) for x in xs) < min_n:
        return pc.OUFit(False, why="표본 부족")
    f, _ = on.fit_noise(on.autocov_days(xs))
    return f


def find_data() -> Path:
    """1분봉 원본 (약 38MB, 저장소엔 안 올림): week06/data → 4주차 폴더 results 순서로 찾음."""
    for p in (HERE / "data" / "pair_1m_full.csv", HERE.parent / "week04" / "results" / "pair_1m_full.csv",
              HERE.parent / "NH_pioneer_week04" / "results" / "pair_1m_full.csv"):
        if p.exists():
            return p
    raise SystemExit("pair_1m_full.csv 가 없음 — 4주차 README 2-2 순서 3 으로 받거나 data/ 에 넣어 주세요")


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"date": str, "time": str})
    d.index = pd.to_datetime(d["date"] + d["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    return d


def prepare(df: pd.DataFrame, tag: str):
    """OU 계획(날마다)과 z 표. 오래 걸려서(1,173일 약 7분) cache/ 에 저장."""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"plans_{tag}.pkl"
    if fp.exists():
        return pickle.loads(fp.read_bytes())
    pc.fit_ou_days = fit_noise_days
    base, c = pc.Params(), pc.Costs()
    pair = pd.concat({"a": df["a_close"], "b": df["b_close"]}, axis=1).dropna()
    V = pc.SpreadView(pair, "ratio", None, base)
    plans = pc.plan_days(df["a_close"], df["b_close"], base, c, view=V)
    feat = pc.features_ou(df["a_close"], df["b_close"], plans, trend_filter=True, view=V)
    fp.write_bytes(pickle.dumps((plans, feat)))
    return plans, feat


# ── 시뮬레이터 ────────────────────────────────────────────────
def simulate(df: pd.DataFrame, F: pd.DataFrame, plans: dict, rules: r6.Rules6, c: pc.Costs = pc.Costs(),
             mult: int = pc.MULT, keep_events: bool = False):
    """Engine6 가 낸 주문을 다음 분 시가에 체결. 반환: (트랜치 청산표, 포지션 단위 표, 엔진)."""
    eng = r6.Engine6(rules, mult=mult)
    idx = F.index
    A_o = df["a_open"].reindex(idx).values
    B_o = df["b_open"].reindex(idx).values
    z, corr, pa, pb, beta = (F[k].values for k in ("z", "corr", "a", "b", "beta"))
    bl, bs = F["block_long"].values, F["block_short"].values
    spr, sdv = F["spread"].values, F["sd"].values
    day = idx.normalize()
    n = len(F)
    rows, pend, skip = [], [], False
    ep_id, ep_of = 0, {}                     # 포지션(에피소드) 번호 — 분할 묶음을 한 거래로 셈
    s_mkt = c.slip_ticks

    def fill_close(i, tids, why, xa, xb, zz):
        nonlocal ep_id
        gone = [x for x in eng.tranches if x.tid in tids]
        net_sum = 0.0
        for x in gone:
            gross = x.pnl(eng.dir, xa, xb, mult)
            cost = (pc.leg_cost(x.qa, x.pa, c, mult, s_mkt) + pc.leg_cost(x.qa, xa, c, mult, s_mkt)
                    + pc.leg_cost(x.qb, x.pb, c, mult, s_mkt) + pc.leg_cost(x.qb, xb, c, mult, s_mkt))
            net = gross - cost
            net_sum += net
            rows.append({"date": idx[i].strftime("%Y%m%d"), "ep": ep_of[x.tid], "tid": x.tid, "tag": x.tag,
                         "shadow": x.shadow, "dir": eng.dir, "units": x.units, "qa": x.qa, "qb": x.qb, "t_in": x.t_in, "t_out": idx[i],
                         "z_in": round(x.z_in, 2), "pa_in": x.pa, "pb_in": x.pb, "pa_out": xa, "pb_out": xb,
                         "hold_min": (idx[i] - x.t_in).total_seconds() / 60, "exit": why,
                         "gross": round(gross), "cost": round(cost), "net": round(net)})
        eng.closed(idx[i], tids, net_sum, why, zz)

    for i in range(n):
        new_day = i == 0 or day[i] != day[i - 1]
        last = i == n - 1 or day[i + 1] != day[i]
        if new_day:
            pl = plans.get(day[i])
            skip = not (pl is not None and pl.ok)
            gap = (spr[i] - spr[i - 1]) / sdv[i] if (i > 0 and not skip and sdv[i] > 0) else float("nan")
            eng.set_day(day[i].date(), pl.params if not skip else None,
                        "" if skip else r6.day_skip(rules, day[i].date(), gap))
        # 1) 걸린 주문 체결 (다음 분 시가)
        for o in pend:
            if o.kind == "open":
                qa = pc.hedge_qty(o.beta, A_o[i], B_o[i], replace(eng.p, unit_b=o.units))
                if qa > 0:
                    if not eng.tranches:
                        ep_id += 1
                    eng.opened(idx[i], o.units, qa, o.units, A_o[i], B_o[i], o.z, o.dir, o.tag)
                    ep_of[eng.tranches[-1].tid] = ep_id
            else:
                live = tuple(t for t in o.tids if any(x.tid == t for x in eng.tranches))
                if live:
                    fill_close(i, live, o.why, A_o[i], B_o[i], o.z)
        pend = []
        if last:                                          # 하루 끝 — 남은 것은 그 분 종가로
            if eng.tranches:
                fill_close(i, tuple(x.tid for x in eng.tranches), "데이터끝", pa[i], pb[i], z[i])
            continue
        if skip and not eng.tranches:
            continue
        blk = frozenset(([+1] if bl[i] else []) + ([-1] if bs[i] else []))
        zz = z[i] if not skip else float("nan")
        for o in eng.on_bar(idx[i], zz, corr[i], pa[i], pb[i], blk, spr[i], sdv[i]):
            if o.kind == "open" and not math.isfinite(beta[i]):
                continue
            o.beta, o.z = beta[i], zz
            pend.append(o)
    T = pd.DataFrame(rows)
    return T, eng


# ── 지표 ─────────────────────────────────────────────────────
def stats(T: pd.DataFrame, days: list) -> dict:
    if len(T) and "shadow" in T:
        T = T[~T["shadow"].astype(bool)]              # 가상 매매(낙폭 중단 중 재검증)는 손익에서 뺌
    if T.empty:
        return {"거래": 0}
    daily = T.groupby("date")["net"].sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    sh = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else float("nan")
    ep = T.groupby("ep").agg(net=("net", "sum"), gross=("gross", "sum"), units=("units", "sum"),
                             exp=("hold_min", lambda s: float((s * T.loc[s.index, "units"]).sum())))
    worst5 = daily.nsmallest(max(1, len(daily) // 20))           # 하위 5% 날의 평균 = 일간 ES95
    return {"거래(포지션)": int(len(ep)), "청산건(트랜치)": int(len(T)),
            "비용전원": round(float(T.gross.sum())), "비용원": round(float(T.cost.sum())),
            "순손익원": round(float(T.net.sum())), "일간샤프": round(float(sh), 2),
            "최대낙폭원": round(float((eq - eq.cummax()).min())),
            "최악의하루원": round(float(daily.min())), "일간ES95원": round(float(worst5.mean())),
            "최악의거래원": round(float(ep.net.min())), "거래당순손익원": round(float(ep.net.mean())),
            "승률": round(float((ep.net > 0).mean()), 3),
            "평균단위": round(float(ep.units.mean()), 2),
            "노출_단위분": round(float(ep.exp.sum())),
            "노출1만단위분당_비용전원": round(float(T.gross.sum() / max(ep.exp.sum(), 1) * 1e4)),
            "청산사유": T["exit"].value_counts().to_dict()}


ARMS = {
    "A 위험관리 없음": r6.no_risk(),
    "A1 A+가격손절": replace(r6.no_risk(), price_stop=True),
    "A2 A+시간손절": replace(r6.no_risk(), time_stop=True),
    "A3 A+금액손절": replace(r6.no_risk(), loss_stop=300_000),
    "B 4주차 안전장치": r6.week4(),
    "C B+분할": replace(r6.week4(), split_entry=True, split_exit=True, protect=True),
    "D 6주차 전체": r6.Rules6(),
    # ── 11/22 개선: D 에 하나씩 쌓음 (값은 결과를 보기 전에 정함, README 4-6) ──
    "E1 D+고정z손절": replace(r6.Rules6(), anchor_stop=True),
    "E2 E1+비율한도": replace(r6.Rules6(), anchor_stop=True, loss_stop_pct=0.013, day_loss_pct=0.026, dd_half_pct=0.066,
                          dd_stop_pct=0.11, dd_recover_pct=0.044),
    "E3 E2+재검증재개": r6.improved(gap_skip=None, skip_expiry=False),
    "E 개선 전체": r6.improved(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--days", type=int, default=0, help="마지막 N영업일만 (0 = 전부)")
    ap.add_argument("--no-sens", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    df = load(Path(a.data) if a.data else find_data())
    days = sorted(df["date"].unique())
    if a.days:
        days = days[-a.days:]
        df = df[df["date"].isin(days)]
    tag = f"{days[0]}_{days[-1]}"
    plans, F = prepare(df, tag)
    print(f"데이터 {days[0]} ~ {days[-1]} {len(days)}일 · 준비 {time.time() - t0:.0f}초", flush=True)
    okd = [p for p in plans.values() if p.ok]
    OUT.mkdir(exist_ok=True)
    rows, yrs, summ, allT = [], [], {}, {}
    for name, R in ARMS.items():
        T, eng = simulate(df, F, plans, R)
        allT[name] = T
        s = stats(T, days)
        rows.append({"설정": name, **{k: v for k, v in s.items() if k != "청산사유"}})
        summ[name] = {"규칙": asdict(R), **s}
        key = name.split(" ")[0]
        T.to_csv(OUT / f"trades_{key}.csv", index=False, encoding="utf-8-sig")
        if key in ("D", "E"):
            E = pd.DataFrame(eng.events)
            E.to_csv(OUT / f"risk_events_{key}.csv", index=False, encoding="utf-8-sig")
            summ[name]["작동횟수"] = E["규칙"].value_counts().to_dict()
        if len(T) and T["shadow"].any():
            sh = T[T["shadow"]]
            summ[name]["가상매매"] = {"건": int(sh["ep"].nunique()), "가상순손익원": int(sh["net"].sum()),
                                  "마지막상태": eng.dd_mode, "가상중": eng.shadow}
        for y in sorted({d[:4] for d in days}):
            dy = [d for d in days if d[:4] == y]
            Ty = T[T["date"].str[:4] == y] if len(T) else T
            sy = stats(Ty, dy)
            yrs.append({"설정": name, "연도": y, **{k: sy.get(k, 0) for k in ("거래(포지션)", "비용전원", "순손익원", "최대낙폭원", "최악의하루원")}})
        print(f"{name:16s} {s.get('순손익원', 0):>12,} 낙폭 {s.get('최대낙폭원', 0):>12,}  ({time.time() - t0:.0f}초)", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yrs).to_csv(OUT / "compare_by_year.csv", index=False, encoding="utf-8-sig")
    print(C.to_string(index=False))

    sens = []
    if not a.no_sens:                                   # 민감도: D 에서 값 하나씩만 바꿈 (고르지 않음)
        for k, vals in (("stop_gap", (1.0, 2.0)), ("add_gap", (0.3, 1.0)), ("loss_stop", (200_000, 500_000)),
                        ("day_loss", (400_000, 1_000_000)), ("day_stops", (1, 3))):
            for v in vals:
                T, _ = simulate(df, F, plans, replace(r6.Rules6(), **{k: v}))
                s = stats(T, days)
                sens.append({"바꾼값": k, "값": v, **{q: s.get(q) for q in ("거래(포지션)", "비용전원", "순손익원", "최대낙폭원", "최악의하루원", "일간ES95원")}})
        for k, vals in (("loss_stop_pct", (0.008, 0.02)), ("dd_stop_pct", (0.07, 0.15)), ("shadow_n", (10, 30)),
                        ("gap_skip", (1.5, 3.0))):
            for v in vals:
                T, _ = simulate(df, F, plans, r6.improved(**{k: v}))
                s = stats(T, days)
                sens.append({"바꾼값": "E:" + k, "값": v, **{q: s.get(q) for q in ("거래(포지션)", "비용전원", "순손익원", "최대낙폭원", "최악의하루원", "일간ES95원")}})
        S = pd.DataFrame(sens)
        S.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
        print(S.to_string(index=False))

    info = {"데이터": [days[0], days[-1]], "영업일": len(days), "거래가능일": len(okd),
            "반감기_중앙분": round(float(np.median([p.fit.half_life for p in okd])), 1),
            "진입선a_중앙": float(np.median([p.plan.entry for p in okd])),
            "청산선m_중앙": float(np.median([p.plan.exit for p in okd])),
            "최대보유_중앙분": float(np.median([p.params.max_hold for p in okd])),
            "비용가정": {"슬리피지틱_다리마다_편도": pc.Costs().slip_ticks, "수수료율_편도": pc.Costs().fee_rate}}
    (OUT / "summary.json").write_text(json.dumps({"설정": info, "비교": summ}, ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    try:
        import figs6
        figs6.make(df, F, plans, allT, days, OUT / "figs")
    except Exception as e:
        print("그림 생략:", e)
    print(f"끝 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
